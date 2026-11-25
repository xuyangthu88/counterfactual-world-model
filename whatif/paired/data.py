"""Matched state, contact, reward and success labels at early and late decision points."""
import argparse
from functools import partial
import multiprocessing as mp

import numpy as np
from whatif.mechanism.collect import tower
from whatif.mechanism.native import N,A,Simulator
from whatif.mechanism.proposals import Proposal,ROOT

DATA=ROOT/'data/paired'
OUT=ROOT/'results/paired'
SPLITS={'train':(1823000,2048),'val':(1923000,240),'test':(2023000,256)}
TEST_CONTROL=2123000
HORIZON=12
BUDGET=36


def proposals(actor,state,role,goal,rng):
    act,prior=actor.candidates(state,role,goal,rng,24)
    # Sample coherent translations of both landmarks as well as changes of end point.
    for k in range(12,24):
        base=0 if k%2==0 else rng.integers(0,4)
        act[k]=act[base]
        prior[k]=prior[base]
        node=act[k,:,0].argmax()
        scale=.15 if k<18 else .35
        shift=rng.normal(0,scale,3)
        act[k,node,1:4]+=shift
        act[k,node,4:7]+=shift+rng.normal(0,scale*.6,3)
    return act,prior


def push(item):
    domain,seed,split=item
    actor=Proposal()
    rng=np.random.default_rng(seed)
    sim=Simulator(domain,seed,split)
    history=[]
    try:
        x,goal=sim.observe()
        role=sim.role
        f0,fa,fe=x,np.zeros((N,A),np.float32),np.zeros((N,N),np.float32)
        # Half of histories follow the reference exactly. The rest explore nearby commands.
        for _ in range(1+seed%35):
            cmd,_=proposals(actor,x,role,goal,rng)
            idx=rng.integers(1,24) if seed%2 and rng.random()<.18 else 0
            new,edge,_=sim.execute(cmd[idx],landmarks=True,stop=True)
            if sim.success:
                break
            history.append(cmd[idx])
            f0,fa,fe,x=x,cmd[idx],edge,new
    finally:
        sim.close()
    cmd,prior=proposals(actor,x,role,goal,rng)
    chosen=[0,rng.integers(1,4),rng.integers(4,12),rng.integers(12,18),
            rng.integers(18,24)]
    remaining=BUDGET-len(history)
    outputs=[]
    for first in cmd[chosen]:
        sim=Simulator(domain,seed,split)
        try:
            for a in history:
                sim.execute(a,landmarks=True)
            replay,_=sim.observe()
            if not np.allclose(x,replay,atol=1e-5):
                raise RuntimeError(f'mismatched branch {domain} {seed}')
            y,edge,r=sim.execute(first,landmarks=True,stop=True)
            peak,last=float(r[2]),float(r[1])
            obs=y
            for _ in range(1,min(remaining,HORIZON)):
                if sim.success:
                    break
                modal=actor.candidates(obs,role,goal,rng,1)[0][0]
                obs,_,rr=sim.execute(modal,landmarks=True,stop=True)
                peak,last=max(peak,float(rr[2])),float(rr[1])
            outputs.append((y,edge,r,[sim.success,peak,last]))
        finally:
            sim.close()
    y,e,r,outcome=map(np.stack,zip(*outputs))
    return dict(f0=f0,fa=fa,f1=x,fe=fe,x=x,role=role,goal=goal,a=cmd[chosen],
                y=y,e=e,r=r,prior=prior[chosen],outcome=outcome,remaining=remaining,seed=seed)


def save(path,rows):
    np.savez_compressed(path,**{k:np.asarray([row[k] for row in rows],
                      dtype=np.int64 if k=='seed' else np.float32) for k in rows[0]})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('domain',choices=['causalworld','pusht','cophy'])
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True)
    with mp.Pool(args.workers,maxtasksperchild=48) as workers:
        for split,(base,n) in SPLITS.items():
            if args.domain=='cophy':
                n={'train':2048,'val':128,'test':192}[split]
                seeds=range(base,base+n)
                rows=workers.map(tower,seeds,chunksize=3)
                save(DATA/f'cophy-{split}.npz',rows)
                if split!='train':
                    effects=workers.map(partial(tower,deletion=True),seeds,chunksize=2)
                    np.savez_compressed(DATA/f'excision-{split}.npz',seed=[a for a,_ in effects],
                                        impact=[b for _,b in effects])
            else:
                items=[(args.domain,base+i,split) for i in range(n)]
                rows=workers.map(push,items,chunksize=2)
                save(DATA/f'{args.domain}-{split}.npz',rows)
            print(args.domain,split,n,flush=True)


if __name__=='__main__':
    main()
