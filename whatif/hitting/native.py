"""Measured potential outcomes at two real decision points, using the frozen policy bank."""
import argparse
import json
import multiprocessing as mp
from functools import partial
import numpy as np
from whatif.feedback.native import ROOT,DOMAINS,command
from whatif.recourse.native import BANK
from whatif.mechanism.native import Simulator
from whatif.mechanism.collect import tower
from whatif.paired.data import save

DATA=ROOT/'data/hitting'
OUT=ROOT/'results/hitting'
SPLITS={'world':(7100000,2048), 'policy':(7200000,1280),'val':(7300000,256),'test':(7400000,320)}
CONTROL=(7500000,768)
TIMES=(1,4,8,16,27)
NPROG=8

def run(item):
    dom,seed,split,a,b,mode=item
    sim=Simulator(dom,int(seed),split)
    try:
        start,goal=sim.observe()
        role=sim.role
        obs=start
        peak=0.
        hit=37
        hist=[]
        path={}
        edge_sum=np.zeros((len(role),len(role)),np.float32)
        for t in range(1,37):
            if not sim.success:
                previous=obs
                weights=None if t==1 else BANK[a if t<=9 else b]
                act=command(obs,role,goal,weights)
                obs,edge,r=sim.execute(act,landmarks=True,stop=True)
                peak=max(peak,float(r[2]))
                if sim.success:
                    hit=t
                last,native=float(r[1]),float(r[0])
                fact=(previous,act,obs)
                edge_sum=np.maximum(edge_sum,edge)
            if t==1:
                first=dict(f0=start,fa=act,f1=obs,fe=edge,goal=goal,role=role,seed=seed,alive=float(not sim.success))
                if mode=='initial':
                    return first
                edge_sum=np.zeros_like(edge_sum)
            if t in (1,5,9):
                hist.append(fact)
            if t==9:
                mid=dict(first=a,x=obs,mid_alive=float(not sim.success),midscore=np.array([peak,last,native]))
                mid.update(zip(('hx','ha','hy'),map(np.stack,zip(*hist))))
                if mode=='reach':
                    return dict(seed=seed,contact=edge_sum.copy())
                if mode=='mid':
                    return dict(**first,**mid)
                edge_sum=np.zeros_like(edge_sum)
            if t in (2,10):
                path['early' if t==2 else 'late']=(obs,edge.copy() if hit>=t else np.zeros_like(edge),r.copy())
            if mode=='target' and t in (2,5,9,17,36):
                path[('initial',t)]=(obs.copy(),native)
            if t>=10 and t-9 in TIMES:
                path[t]=(obs.copy(),native,edge_sum.copy())
        outcome=np.array([float(sim.success),peak,last],np.float32)
        if mode=='score':
            return outcome
        final=dict(hit=hit,outcome=outcome,final=obs,native=native)
        if mode=='target':
            traj,rewards=map(np.stack,zip(*[path[('initial',t)] for t in (2,5,9,17,36)]))
            return dict(**final,traj=traj,rewards=rewards)
        y,rew,edges=map(np.stack,zip(*[path[9+t] for t in TIMES]))
        ey,ee,er=path['early']
        ly,le,lr=path['late']
        return dict(**first,**mid,**final,y=ly,e=le,r=lr,ey=ey,ee=ee,er=er,traj=y,rewards=rew,edges=edges)
    finally:
        sim.close()

def panel(item):
    dom,s,split,a=item
    group=[run((dom,s,split,a,b,'panel')) for b in range(len(BANK))]
    r=group[0].copy()
    if any(not np.allclose(x['hy'],r['hy'],atol=1e-5) for x in group):
        raise ValueError('midpoint replay mismatch')
    for k in ('y','e','r','traj','rewards','edges','hit','outcome','final','native'):
        r[k]=np.stack([v[k] for v in group])
    return r

def collect(pool,dom):
    for split,(base,n) in SPLITS.items():
        items=[(dom,s,'train' if split in ('world','policy') else split,int(a))
               for s in range(base,base+n) for a in np.random.default_rng(s).choice(len(BANK),2,replace=False)]
        save(DATA/f'late-{dom}-{split}.npz',pool.map(panel,items,chunksize=1))
        print('late',dom,split,n,flush=True)
    base,n=CONTROL
    save(DATA/f'{dom}-initial.npz',pool.map(run,[(dom,s,'control',0,0,'initial') for s in range(base,base+n)]))

def early(pool,job):
    for split,(base,n) in SPLITS.items():
        base+=1000000
        for dom in DOMAINS:
            phase='train' if split in ('world','policy') else split
            if job=='prepare':
                items=[(dom,s,phase,a,0,'mid') for s in range(base,base+n) for a in range(len(BANK))]
                save(DATA/f'mid-{dom}-{split}.npz',pool.map(run,items,chunksize=2))
            else:
                chosen=np.load(OUT/f'continuation-{dom}-{split}.npy').reshape(n,len(BANK))
                items=[(dom,s,phase,a,b,'target') for i,s in enumerate(range(base,base+n))
                       for a,b in [(0,0),*enumerate(chosen[i])]]
                raw=pool.map(run,items,chunksize=2)
                grouped=[{k:np.stack([r[k] for r in raw[j:j+NPROG]]) for k in raw[j]}
                         for j in range(0,len(raw),NPROG)]
                for i,row in enumerate(grouped):
                    row['seed']=base+i
                save(DATA/f'early-{dom}-{split}.npz',grouped)
            print(job,dom,split,n,flush=True)

def contacts(pool):
    for split,(base,n) in SPLITS.items():
        base+=1000000
        for dom in DOMAINS:
            phase='train' if split in ('world','policy') else split
            rows=pool.map(run,[(dom,s,phase,a,0,'reach') for s in range(base,base+n)
                              for a in range(len(BANK))],chunksize=3)
            save(DATA/f'contact-{dom}-{split}.npz',rows)
            print('contact',dom,split,n,flush=True)

def cophy(pool):
    for split,base,n in [('train',9100000,512),('test',9200000,224)]:
        seeds=range(base,base+n)
        save(DATA/f'cophy-{split}.npz',pool.map(tower,seeds,chunksize=2))
        effects=pool.map(partial(tower,deletion=True),seeds,chunksize=2)
        np.savez_compressed(DATA/f'excision-{split}.npz',seed=[s for s,_ in effects],impact=[v for _,v in effects])
        print('cophy',split,n,flush=True)

def control(pool,phase):
    base,n=CONTROL
    path=OUT/('first.json' if phase=='observe' else 'second.json')
    plans=json.loads(path.read_text())
    result={}
    for dom,arms in plans.items():
        if phase=='observe':
            keys=sorted({(base+i,p[0]) for seq in arms.values() for i,p in enumerate(seq)})
            save(DATA/f'{dom}-live.npz',pool.map(run,[(dom,s,'control',a,0,'mid') for s,a in keys],chunksize=2))
        else:
            keys=sorted({(base+i,a,b) for seq in arms.values() for i,(a,b) in enumerate(seq)})
            values=dict(zip(keys,pool.map(run,[(dom,s,'control',a,b,'score') for s,a,b in keys],chunksize=2)))
            result[dom]={k:[values[base+i,a,b].tolist() for i,(a,b) in enumerate(seq)] for k,seq in arms.items()}
        print(phase,dom,len(keys),flush=True)
    if phase=='execute':
        (OUT/'episodes.json').write_text(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=[*DOMAINS,'cophy','prepare','targets','contact','observe','execute'])
    p.add_argument('--workers',type=int,default=56)
    args=p.parse_args()
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    with mp.Pool(args.workers,maxtasksperchild=64) as pool:
        if args.job in DOMAINS:
            collect(pool,args.job)
        elif args.job=='cophy':
            cophy(pool)
        elif args.job in ('prepare','targets'):
            early(pool,args.job)
        elif args.job=='contact':
            contacts(pool)
        else:
            control(pool,args.job)
