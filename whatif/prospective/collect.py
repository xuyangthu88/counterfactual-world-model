"""One observed factual stroke, then a new, independently disturbed set of sibling actions."""
from concurrent.futures import ProcessPoolExecutor
import argparse
from pathlib import Path
import numpy as np
from whatif.shared.actions import menu

DATA=Path('data/prospective')
OUT=Path('results/prospective')
START={'fit':311000,'val':321000,'test':331000,'controlval':341000,'control':351000}


def collect(job):
    from whatif.shared.sim import Scene
    from whatif.shared.returns import physics
    task,split,seed=job
    rng=np.random.default_rng(seed)
    envsplit={'fit':'train','val':'transfer_val','test':'test'}[split]
    near=split=='fit' and task=='pusht' and rng.random()<.30
    pre=[]
    env=Scene(task,seed,envsplit,near)
    for _ in range(int(rng.integers(0,7 if task=='pusht' else 3))):
        st=env.state()
        ranking=np.argsort(-physics(st,menu(st)))
        i=int(ranking[rng.choice([0,0,1,2,4])]);pre.append(i);env.push(i)
    s0=env.state();a0=menu(s0)
    b=int(rng.choice([0,1,2,4,7]))
    u0,u1=rng.choice([-.04,0.,.04],size=2)
    s,_,_,_=env.push(b,noise=u0)
    free=env.latest_free
    measure=np.array([len(free),np.mean(free) if free else 0.])
    a=menu(s);truth=env.env.physics;group=env.env.protocol
    env.close()
    end=[];edges=[];peaks=[]
    for j in range(12):
        env=Scene(task,seed,envsplit,near)
        for ix in pre:env.push(ix)
        observed=env.push(b,noise=u0)[0]
        if j==0:np.testing.assert_allclose(observed,s,rtol=1e-5,atol=6e-4)
        z,e,_,p=env.push(j,noise=u1)
        end.append(z);edges.append(e);peaks.append(p);env.close()
    return dict(seed=seed,previous=s0,previous_actions=a0,b=b,state=s,cmd=a,measurement=measure,
                end=end,edge=edges,peak=peaks,ref=int(rng.choice([0,1,2,4,7])),protocol=group,physics=truth)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task');ap.add_argument('split');ap.add_argument('count',type=int);ap.add_argument('--workers',type=int,default=16)
    a=ap.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        rows=list(pool.map(collect,[(a.task,a.split,i) for i in range(START[a.split],START[a.split]+a.count)]))
    np.savez_compressed(DATA/f'{a.task}-{a.split}.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
    print(a.task,a.split,len(rows),flush=True)
