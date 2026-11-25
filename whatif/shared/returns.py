"""Supervise the value of a physical continuation with realized, fixed-policy pushes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .actions import DATA,menu,rotate,wrap
from .sim import Scene


def physics(s,cmd):
    dom=s[13];known=s[18]
    goal=rotate(s[9:11]-s[:2],-s[2])+dom*(rotate(np.array([0.,.45]),s[11]-s[2])-[0.,.45])
    gain=(.66+.50*s[17]*known)*dom+(1-dom)*.95
    drift=rotate(s[14:16],-s[2])*(1+3*s[17]*known)
    delta=cmd[:,2:4]*cmd[:,4:5]*gain+drift
    theta=cmd[:,5]*cmd[:,4]*gain/(.50-.2791*dom)
    ang=wrap(s[11]-s[2]-theta-s[16]*(1+3*s[17]*known))
    return -np.sqrt(((goal-delta)**2).sum(-1)+(.47*dom*ang)**2+1e-8)-.012*cmd[:,6]


def run(job):
    task,split,seed=job
    rng=np.random.default_rng(seed)
    near=task=='pusht' and rng.random()<.35
    env=Scene(task,seed,split,near)
    try:
        # Roll in with the deployable controller, occasionally exploring nearby contacts.
        for _ in range(rng.integers(0,12 if task=='pusht' else 5)):
            s=env.state();order=np.argsort(-physics(s,menu(s)))
            i=order[rng.choice([0,0,0,1,2,4])]
            env.push(i)
        s=env.state();cmd=menu(s)
        order=np.argsort(-physics(s,cmd))
        i=order[rng.choice([0,0,1,2,3,4,6,9])]
        z,_,_,peak=env.push(i,stop=True)
        post=z.copy(); future=[]
        for _ in range(3):
            if z[12]<=(.9+.05*z[13]):
                a=physics(z,menu(z)).argmax()
                z,_,_,peak=env.push(a,stop=True)
            future.append(z.copy())
        return post,np.asarray(future),env.env.physics,seed
    finally:env.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task');ap.add_argument('split');ap.add_argument('count',type=int);ap.add_argument('--workers',type=int,default=24)
    a=ap.parse_args();start={'train':271000,'val':281000}[a.split]
    with ProcessPoolExecutor(a.workers) as ex:
        rows=list(ex.map(run,[(a.task,a.split,i) for i in range(start,start+a.count)]))
    out=dict(zip(['state','future','physics','seed'],map(np.asarray,zip(*rows))))
    np.savez_compressed(DATA/f'return-{a.task}-{a.split}.npz',**out)
    print(a.task,a.split,len(rows),flush=True)
