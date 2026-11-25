"""Matched sibling branches and source-null interventions from the two simulators."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic

import numpy as np
from .tasks import ACTIONS, BASE, COUNTS, TASKS, create


def scene(job):
    task, split, seed = job
    rng = np.random.default_rng(seed)
    prefix = [(int(rng.choice([0,1,2,3,9])), 0.) for _ in range(int(rng.integers(0,4)))]
    if split.startswith('near'):
        prefix = []
    ref = int(rng.choice([0,1,2,3,9]))
    noise = rng.choice([-.04,0.,.04],2)
    env = create(task,seed,split)
    def restore_slow(history):
        nonlocal env
        env.close()
        env = create(task,seed,split)
        for a,n in history:
            env.run(ACTIONS[a],n)
    try:
        for a,n in prefix:
            env.run(ACTIONS[a],n)
        initial, ends, edges, null_ends, null_source = [],[],[],[],[]
        for t in range(2):
            start = env.obs()
            initial.append(start)
            panel, labels, removed, sources = [],[],[],[]
            # Rebuild the seeded simulator for each sibling, including warm contacts,
            # actuator state and the entire prefix. This also isolates collision filters.
            for a, action in enumerate(ACTIONS):
                restore_slow(prefix)
                z,e = env.run(action,noise[t])
                panel.append(z)
                labels.append(e)
                source = int(rng.integers(3))
                sources.append(source)
                if split.endswith('train'):
                    restore_slow(prefix)
                    znull,_ = env.run(action,noise[t],source)
                    removed.append(znull[:3])
            ends.append(panel)
            edges.append(labels)
            null_source.append(sources)
            if removed:
                null_ends.append(removed)
            if t==0:
                restore_slow(prefix)
                actual,_=env.run(ACTIONS[ref],noise[t])
                np.testing.assert_allclose(actual,panel[ref],atol=6e-4,rtol=1e-5)
                prefix.append((ref,noise[t]))
        return dict(seed=seed,state=initial,end=ends,edge=edges,ref=ref,physics=env.physics,
                    protocol=env.protocol,physical_value=env.mass,execution=noise,
                    null_source=null_source,null_pose=null_ends)
    finally:
        env.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',choices=TASKS,required=True)
    ap.add_argument('--splits',nargs='+',default=['train','val','test'])
    ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--count',type=int)
    ap.add_argument('--out',type=Path,default=Path('data/crossbench'))
    args=ap.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    for split in args.splits:
        n=args.count or COUNTS[split]
        start=BASE[split]
        tic=monotonic()
        with ProcessPoolExecutor(args.workers) as pool:
            rows=list(pool.map(scene,((args.task,split,i) for i in range(start,start+n))))
        out={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        for k,v in out.items():
            if np.issubdtype(v.dtype,np.floating):
                out[k]=v.astype(np.float32)
        np.savez_compressed(args.out/f'{args.task}-{split}.npz',**out)
        print(args.task,split,n,round(monotonic()-tic,1),flush=True)

if __name__=='__main__':
    main()
