"""Matched two-decision native continuations with the frozen Stage 23 policy bank."""
import argparse
from functools import partial
import json
import multiprocessing as mp
import numpy as np
from whatif.feedback.native import command, DOMAINS, ROOT
from whatif.mechanism.collect import tower
from whatif.mechanism.native import Simulator
from whatif.paired.data import save

DATA=ROOT/'data/recourse'
OUT=ROOT/'results/recourse'
BANK=np.asarray(json.loads((ROOT/'results/feedback/expanded-bank.json').read_text())['bank'])
MIDDLE=9
START={'train':(5100000,1536),'val':(5200000,160),'test':(5300000,192),'control':(5400000,384)}

def rollout(item):
    dom,seed,split,first,second,mode=item
    sim=Simulator(dom,seed,split)
    try:
        x,goal=sim.observe()
        role=sim.role
        peak=0.
        def advance(x,w):
            a=command(x,role,goal,w)
            y,e,r=sim.execute(a,landmarks=True,stop=True)
            return a,y,e,r
        a0,start,e0,r0=advance(x,None)
        fact=(x,a0,start)
        record=dict(f0=x,fa=a0,f1=start,fe=e0,x=start,goal=goal,role=role,seed=seed,
                    alive=float(not sim.success))
        if mode=='first':
            return record
        obs=start
        last,native,peak=r0[1],r0[0],r0[2]
        hist=[fact]
        mid_edge=np.zeros_like(e0)
        early_y,early_e,early_r=start,mid_edge.copy(),r0
        for step in range(2,MIDDLE+1):
            if not sim.success:
                a,y,e,r=advance(obs,BANK[first])
                fact=(obs,a,y)
                obs=y
                mid_edge=np.maximum(mid_edge,e)
                last,native,peak=r[1],r[0],max(peak,r[2])
                if step==2:
                    early_y,early_e,early_r=y,e,r
            if step in (5,MIDDLE):
                hist.append(fact)
        record.update(first=first,mid=obs,mide=mid_edge,midscore=np.array([sim.success,peak,last,native]),
                      early_y=early_y,early_e=early_e,early_r=early_r)
        record.update(zip(('hx','ha','hy'),map(np.stack,zip(*hist))))
        if mode=='middle':
            return record
        a1=command(obs,role,goal,BANK[second])
        y1,e1,r1=obs,np.zeros_like(e0),np.array([native,last,last])
        cumulative=np.zeros_like(e0)
        for step in range(MIDDLE+1,37):
            if sim.success:
                break
            a,y,e,r=advance(obs,BANK[second])
            obs=y
            last,native,peak=r[1],r[0],max(peak,r[2])
            cumulative=np.maximum(cumulative,e)
            if step==MIDDLE+1:
                y1,e1,r1=obs,e,r
        outcome=np.array([sim.success,peak,last])
        if mode=='score':
            return outcome
        record.update(late_a=a1,late_y=y1,late_e=e1,late_r=r1,final=obs,final_e=cumulative,
                      outcome=outcome,native=native)
        return record
    finally:
        sim.close()

def row(item):
    dom,seed,split,first=item
    group=[rollout((dom,seed,split,first,b,'panel')) for b in range(len(BANK))]
    record=group[0].copy()
    if any(not np.allclose(v['hy'],record['hy'],atol=1e-5) for v in group):
        raise ValueError('mismatched midpoint replay')
    for key in ('late_a','late_y','late_e','late_r','final','final_e','outcome','native'):
        record[key]=np.stack([b[key] for b in group])
    return record

def collect(pool,dom):
    for split,(base,n) in START.items():
        if split=='control':
            save(DATA/f'{dom}-observe.npz',pool.map(rollout,[(dom,s,split,0,0,'first') for s in range(base,base+n)]))
            continue
        items=[]
        for seed in range(base,base+n):
            k=3 if split=='train' else (3 if split=='val' else 4)
            ids=np.random.default_rng(seed).choice(len(BANK),k,replace=False)
            items.extend((dom,seed,split,int(a)) for a in ids)
        results=pool.map(row,items,chunksize=1)
        save(DATA/f'{dom}-{split}.npz',results)
        print(dom,split,n,len(results),flush=True)

def towers(pool):
    for split,base,n in [('train',5500000,640),('test',5600000,192)]:
        seeds=range(base,base+n)
        save(DATA/f'cophy-{split}.npz',pool.map(tower,seeds,chunksize=2))
        tags=pool.map(partial(tower,deletion=True),seeds,chunksize=2)
        np.savez_compressed(DATA/f'excision-{split}.npz',seed=[s for s,_ in tags],impact=[d for _,d in tags])
        print('towers',split,n,flush=True)

def run_control(pool,phase):
    base,count=START['control']
    plan=json.loads((OUT/('first.json' if phase=='middle' else 'second.json')).read_text())
    result={}
    for dom,arms in plan.items():
        if phase=='middle':
            keys=sorted({(base+i,a) for sequence in arms.values() for i,a in enumerate(sequence)})
            rows=pool.map(rollout,[(dom,s,'control',a,0,'middle') for s,a in keys],chunksize=2)
            save(DATA/f'{dom}-live.npz',rows)
        else:
            keys=sorted({(base+i,a,b) for sequence in arms.values() for i,(a,b) in enumerate(sequence)})
            observed=dict(zip(keys,pool.map(rollout,[(dom,s,'control',a,b,'score') for s,a,b in keys],chunksize=2)))
            result[dom]={key:[observed[base+i,a,b].tolist() for i,(a,b) in enumerate(sequence)]
                         for key,sequence in arms.items()}
        print(dom,phase,len(keys),flush=True)
    if phase=='execute':
        (OUT/'episodes.json').write_text(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=[*DOMAINS,'cophy','middle','execute'])
    p.add_argument('--workers',type=int,default=40)
    args=p.parse_args()
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    with mp.Pool(args.workers,maxtasksperchild=64) as pool:
        if args.job in DOMAINS:
            collect(pool,args.job)
        elif args.job=='cophy':
            towers(pool)
        else:
            run_control(pool,args.job)
