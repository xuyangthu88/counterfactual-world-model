"""Paired native six-macro returns under one common continuation policy."""
import argparse
import multiprocessing as mp
import numpy as np
from .native import Simulator,N,A
from .proposals import Proposal,ROOT

HORIZON=6
SPLITS={'train':(1423000,1024),'val':(1523000,160),'test':(1623000,192)}


def sample(item):
    domain,seed,split=item
    actor=Proposal()
    rng=np.random.default_rng(seed)
    sim=Simulator(domain,seed,split)
    history=[]
    try:
        x,goal=sim.observe()
        role=sim.role
        f0,f1,fa=x,x,np.zeros((N,A),np.float32)
        for _ in range(1+seed%18):
            actions,_=actor.candidates(x,role,goal,rng,8)
            k=0 if rng.random()<.80 else rng.integers(1,8)
            new,_,_=sim.execute(actions[k],stop=True,landmarks=True)
            if sim.success:
                break
            history.append(actions[k])
            f0,fa,x=x,actions[k],new
            f1=x
    finally:
        sim.close()
    candidates,priors=actor.candidates(x,role,goal,rng,24)
    ix=np.r_[0,rng.choice(np.arange(1,24),3,replace=False)]
    commands=candidates[ix]
    values=[]
    for cmd in commands:
        sim=Simulator(domain,seed,split)
        try:
            for previous in history:
                sim.execute(previous,landmarks=True)
            current,_=sim.observe()
            if not np.allclose(current,x,atol=1.e-5):
                raise RuntimeError('paired history mismatch')
            peak=0.
            action=cmd
            for j in range(HORIZON):
                current,_,r=sim.execute(action,stop=True,landmarks=True)
                peak=max(peak,float(r[2]))
                if sim.success:
                    break
                action=actor.candidates(current,role,goal,rng,1)[0][0]
            values.append([sim.success,peak,float(r[1])])
        finally:
            sim.close()
    return dict(f0=f0,fa=fa,f1=f1,x=x,role=role,goal=goal,a=commands,prior=priors[ix],
                same=0.,outcome=values,seed=seed)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('domain',choices=['causalworld','pusht'])
    p.add_argument('--workers',type=int,default=16)
    args=p.parse_args()
    folder=ROOT/'data/mechanism/continuations'
    folder.mkdir(parents=True,exist_ok=True)
    for split,(base,n) in SPLITS.items():
        with mp.Pool(args.workers,maxtasksperchild=32) as pool:
            rows=pool.map(sample,[(args.domain,base+j,split) for j in range(n)],chunksize=2)
        np.savez_compressed(folder/f'{args.domain}-{split}.npz',
                 **{k:np.asarray([r[k] for r in rows],dtype=np.int64 if k=='seed' else np.float32) for k in rows[0]})
        print(args.domain,split,len(rows),flush=True)


if __name__=='__main__':
    main()

