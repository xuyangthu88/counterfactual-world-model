"""Retrieve three-macro options from the original training demonstrations and achieved goals."""
import argparse
from functools import lru_cache
import multiprocessing as mp
import numpy as np
from scipy.spatial import cKDTree
from whatif.mechanism.proposals import Proposal,local_view,ROOT
from whatif.mechanism.native import Simulator,N,D,planar,roles
from .data import DATA,OUT,BUDGET,HORIZON,save

LENGTH=3
SPLITS={'train':(2423000,2048),'val':(2523000,224),'test':(2623000,256)}


def make_bank():
    from whatif.mechanism.fit_proposal import read_demo
    state,act,ends=read_demo(ROOT/'work/mechanism/pusht-demos.zip')
    xs,ys=[],[]
    def append(x,role,g,node,offsets):
        ids,rot,keys=local_view(x,role,g)
        xs.append(keys[np.flatnonzero(ids==node)[0]])
        ys.append(offsets@rot)
    role=roles([0],[1],[2,3,4,5])
    goal=planar([0,0],np.pi/4)
    for ep,(start,end) in enumerate(zip(np.r_[0,ends[:-1]],ends)):
        if ep%10==9:
            continue
        for t in range(start,end):
            x=np.zeros((N,D),np.float32)
            x[0]=planar((state[t,2:4]-256)/100,state[t,4])
            x[1]=planar((state[t,:2]-256)/100)
            xy=(act[np.minimum(t+np.arange(3,24,4),end-1)]-state[t,:2])/100
            append(x,role,goal,1,np.column_stack([xy,np.zeros(6)]))
    with np.load(ROOT/'data/mechanism/fine/causalworld-train.npz') as d:
        for x,role,goal,a,y,r in zip(*(d[k] for k in ('x','role','goal','a','y','r'))):
            err=y[:,0]-goal
            score=r[:,1]-.35*np.sqrt((err[:,:3]**2).sum(1)+.25*(err[:,3:]**2).sum(1))
            best=score.argmax()
            for i,cmd in enumerate(a):
                node=cmd[:,0].argmax()
                position=np.zeros(3)
                marks=[]
                for dest in cmd[node,1:].reshape(2,3):
                    for step in range(12):
                        diff=dest-position
                        position=position+diff*min(1.,.12/max(1e-9,np.linalg.norm(diff)))
                        if step%4==3:
                            marks.append(position.copy())
                append(x,role,y[i,0],node,np.array(marks))
                if i==best:
                    append(x,role,goal,node,np.array(marks))
    keys=np.asarray(xs,np.float32)
    vals=np.asarray(ys,np.float32)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'bank.npz',keys=keys,offsets=vals,spread=keys.std(0).clip(.12,None))
    print('training examples',len(keys),flush=True)


class Bank:
    def __init__(self):
        with np.load(OUT/'bank.npz') as d:
            self.spread=d['spread']
            self.tree=cKDTree(d['keys']/self.spread)
            self.vectors=d['offsets']
        self.actor=Proposal()

    def plans(self,x,role,goal):
        ids,rot,features=local_view(x,role,goal)
        distance,neighbors=self.tree.query(features/self.spread,k=8)
        row=distance[:,0].argmin()
        node=ids[row]
        ds=distance[row]
        paths=self.vectors[neighbors[row]]
        # A distinct neighbor among the eight nearest observations, plus their local mean.
        distinct=np.linalg.norm((paths[:,-1]-paths[0,-1]),axis=-1).argmax()
        weights=1/(.05+ds)
        average=np.average(paths,axis=0,weights=weights)
        local=np.stack([paths[0],paths[distinct],average])
        absolute=local@rot.T+x[node,:3]
        return node,absolute.astype(np.float32)

    def command(self,choice,step,x,role,goal,node,absolute):
        if choice==0:
            return self.actor.candidates(x,role,goal,None,1)[0][0]
        cmd=np.zeros((N,7),np.float32)
        cmd[node]=np.r_[1,(absolute[choice-1,step*2:step*2+2]-x[node,:3]).ravel()]
        return cmd


@lru_cache(maxsize=1)
def bank():
    return Bank()


def sample(item):
    dom,seed,split=item
    model=bank()
    rng=np.random.default_rng(seed)
    sim=Simulator(dom,seed,split)
    history=[]
    try:
        x,goal=sim.observe()
        role=sim.role
        f0,fa=x,np.zeros((N,7),np.float32)
        for _ in range(1+seed%35):
            if not history or len(history)%LENGTH==0:
                node,plan=model.plans(x,role,goal)
                choice=rng.integers(1,4) if rng.random()<.3 else 0
            action=model.command(choice,len(history)%LENGTH,x,role,goal,node,plan)
            new,_,_=sim.execute(action,stop=True,landmarks=True)
            if sim.success:
                break
            history.append(action)
            f0,fa,x=x,action,new
    finally:
        sim.close()
    node,plans=model.plans(x,role,goal)
    outcomes=[]
    commands=[]
    for choice in range(4):
        sim=Simulator(dom,seed,split)
        try:
            for a in history:
                sim.execute(a,landmarks=True)
            obs,_=sim.observe()
            if not np.allclose(x,obs,atol=1e-5):
                raise RuntimeError(f'{dom} {seed}: unmatched history')
            peak=0.
            for step in range(min(HORIZON,BUDGET-len(history))):
                cmd=model.command(choice if step<LENGTH else 0,step,obs,role,goal,node,plans)
                if step==0:
                    commands.append(cmd)
                obs,_,r=sim.execute(cmd,landmarks=True,stop=True)
                peak=max(peak,float(r[2]))
                if sim.success:
                    break
            outcomes.append([sim.success,peak,float(r[1])])
        finally:
            sim.close()
    return dict(f0=f0,fa=fa,f1=x,x=x,goal=goal,role=role,same=0.,a=commands,node=node,
                plans=plans,remaining=BUDGET-len(history),outcome=outcomes,seed=seed)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('operation',choices=['bank','pusht','causalworld'])
    p.add_argument('--workers',type=int,default=24)
    args=p.parse_args()
    if args.operation=='bank':
        return make_bank()
    with mp.Pool(args.workers,maxtasksperchild=48) as workers:
        for split,(base,n) in SPLITS.items():
            rows=workers.map(sample,[(args.operation,base+i,split) for i in range(n)],chunksize=2)
            save(DATA/f'segment-{args.operation}-{split}.npz',rows)
            print(args.operation,split,n,flush=True)


if __name__=='__main__':
    main()
