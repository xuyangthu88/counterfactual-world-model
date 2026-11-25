import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .actions import DATA,START,menu
from .sim import Scene


def sample(job):
    task,split,seed=job
    rng=np.random.default_rng(seed)
    near=task=='pusht' and split in ('train','val') and rng.random()<.32
    history=[]
    env=Scene(task,seed,split,near)
    count=rng.integers(0,8 if task=='pusht' else 3)
    for _ in range(count):
        a=int(rng.choice([0,1,1,2,4,7]))
        env.push(a);history.append(a)
    state=env.state();cmd=menu(state);physical=env.env.physics;protocol=env.env.protocol
    ref=int(rng.choice([0,1,2,4,7]));noise=float(rng.choice([-.04,0.,.04]))
    env.close()
    def branch(i,null=-1):
        e=Scene(task,seed,split,near)
        try:
            for a in history:e.push(a)
            return e.push(i,noise=noise,null=null)
        finally:e.close()
    end=[];edge=[];steps=[];peak=[];yn=[];source=rng.integers(3,size=12)
    for i in range(12):
        z,g,n,c=branch(i)
        end.append(z);edge.append(g);steps.append(n);peak.append(c)
        if split=='train':
            zn=branch(i,int(source[i]))[0];yn.append(zn[:3])
    return dict(seed=seed,state=state,cmd=cmd,end=end,edge=edge,peak=peak,steps=steps,physics=physical,
                protocol=protocol,ref=ref,noise=noise,source=source,nullpose=yn)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task');ap.add_argument('split');ap.add_argument('count',type=int);ap.add_argument('--workers',type=int,default=24)
    a=ap.parse_args();start=START[a.split]
    with ProcessPoolExecutor(a.workers) as ex:
        rows=list(ex.map(sample,[(a.task,a.split,i) for i in range(start,start+a.count)]))
    d={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(DATA/f'{a.task}-{a.split}.npz',**d)
    print(a.task,a.split,len(rows),flush=True)
