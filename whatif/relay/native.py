"""Interval contact graphs and matched potential outcomes in the native simulators."""
import argparse
import json
import multiprocessing as mp
from functools import partial
import numpy as np
from whatif.mechanism.collect import tower
from whatif.mechanism.native import Simulator
from whatif.feedback.native import ROOT,DOMAINS,command
from whatif.recourse.native import BANK
from whatif.hitting.native import run as execute
from whatif.paired.data import save

DATA=ROOT/'data/relay'
OUT=ROOT/'results/relay'
SPLITS={'train':(11000000,2048),'policy':(11100000,1024),'val':(11200000,224),'test':(11300000,320)}
CONTROL=(11400000,1024)
VALIDATION=(11800000,320)
FOLLOWUP=(12100000,1024)
ENDS={'early':(2,5,9,17,36),'late':(10,13,17,25,36)}
TICKS=(1,2,4,6,8)

def branch(dom,seed,split,phase,first,arm,continuation=None):
    sim=Simulator(dom,seed,split)
    try:
        start=1 if phase=='early' else 9
        ends=ENDS[phase]
        x,goal=sim.observe()
        role=sim.role
        recent=[]
        traj,contacts,rewards,valid=[],[],[],[]
        short=[]
        seen=np.zeros((len(role),len(role)),np.float32)
        count=0
        peak=0.
        hit=37
        for t in range(1,37):
            if not sim.success:
                before=x
                idx=0 if t==1 else first if t<=start else arm
                if continuation is not None and t>9:
                    idx=continuation
                a=command(x,role,goal,BANK[idx])
                tracing=(t==start+1)
                def record(y,e,r):
                    short.append((y,e,r))
                x,edge,r=sim.execute(a,stop=True,landmarks=True,on_tick=record if tracing else None)
                peak=max(peak,r[2])
                if sim.success:
                    hit=t
                fact=before,a,x,edge
                if t>start:
                    np.maximum(seen,edge,out=seen)
                    count+=1
            if t in (1,5,9) and t<=start:
                recent.append(fact)
            if t==start:
                row=dict(seed=seed,x=x,goal=goal,role=role,first=first,alive=float(not sim.success))
                row.update(zip(('hx','ha','hy','he'),map(np.stack,zip(*recent))))
            if t==start+1:
                row.update(a=command(row['x'],role,goal,BANK[arm]),r1=r.copy())
            if t in ends:
                traj.append(x)
                contacts.append(seen.copy())
                rewards.append(r[0])
                valid.append(count>0)
                seen.fill(0)
                count=0
        st,se,sv=[],[],[]
        prev=0
        for end in TICKS:
            group=short[prev:end]
            st.append(short[min(end,len(short))-1][0] if short else row['x'])
            se.append(np.maximum.reduce([z[1] for z in group]) if group else np.zeros_like(seen))
            sv.append(bool(group))
            prev=end
        row.update(hit=hit,outcome=[float(sim.success),peak,r[1]],traj=traj,contact=contacts,rewards=rewards,
                   valid=valid,short_state=st,short_edge=se,short_valid=sv)
        return row
    finally:
        sim.close()

def panel(item):
    dom,seed,split,phase=item
    rng=np.random.default_rng(seed)
    first=int(rng.integers(len(BANK))) if phase=='late' else 0
    branches=[branch(dom,seed,split,phase,first,k) for k in range(len(BANK))]
    row=branches[0].copy()
    for other in branches[1:]:
        if not np.array_equal(row['hy'],other['hy']):
            raise ValueError('native branches have different prefixes')
    for k in ('a','r1','outcome','hit','traj','contact','rewards','valid','short_state','short_edge','short_valid'):
        row[k]=np.stack([r[k] for r in branches])
    return row

def collect(pool,dom,only=None):
    for split,(base,n) in SPLITS.items():
        if only and split!=only:
            continue
        for phase in ENDS:
            offset=25000*(phase=='late')
            env_split='train' if split=='policy' else split
            rows=pool.map(panel,[(dom,s+offset,env_split,phase) for s in range(base,base+n)],chunksize=2)
            save(DATA/f'{dom}-{phase}-{split}.npz',rows)
            print(dom,phase,split,n,flush=True)

def controls(pool,job,validation=False,followup=False):
    base,n=FOLLOWUP if followup else VALIDATION if validation else CONTROL
    prefix='backup-' if followup else 'val-' if validation else ''
    split='val' if validation else 'control'
    if job!='initial':
        plans=json.loads((OUT/(prefix+('first.json' if job=='observe' else 'second.json'))).read_text())
    out={}
    for dom in DOMAINS:
        if job=='initial':
            rows=pool.map(execute,[(dom,s,split,0,0,'initial') for s in range(base,base+n)])
            save(DATA/f'{prefix}{dom}-initial.npz',rows)
        elif job=='observe':
            keys=sorted({(base+i,a) for seq in plans[dom].values() for i,(a,_) in enumerate(seq)})
            rows=pool.map(execute,[(dom,s,split,a,0,'mid') for s,a in keys],chunksize=2)
            save(DATA/f'{prefix}{dom}-mid.npz',rows)
        else:
            keys=sorted({(base+i,a,b) for seq in plans[dom].values() for i,(a,b) in enumerate(seq)})
            vs=dict(zip(keys,pool.map(execute,[(dom,s,split,a,b,'score') for s,a,b in keys],chunksize=2)))
            out[dom]={k:[vs[base+i,a,b].tolist() for i,(a,b) in enumerate(seq)] for k,seq in plans[dom].items()}
        print(dom,job,flush=True)
    if job=='execute':
        (OUT/(prefix+'episodes.json')).write_text(json.dumps(out))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=[*DOMAINS,'cophy','initial','observe','execute'])
    p.add_argument('--workers',type=int,default=36)
    p.add_argument('--only',choices=SPLITS)
    p.add_argument('--validation',action='store_true')
    p.add_argument('--followup',action='store_true')
    args=p.parse_args()
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    with mp.Pool(args.workers,maxtasksperchild=64) as pool:
        if args.job in DOMAINS:
            collect(pool,args.job,args.only)
        elif args.job=='cophy':
            for split,n,base in [('train',640,11500000),('val',144,11600000),('test',256,11700000)]:
                rows=pool.map(partial(tower,trace=True),range(base,base+n),chunksize=3)
                save(DATA/f'cophy-{split}.npz',rows)
                print(split,n,flush=True)
        else:
            controls(pool,args.job,args.validation,args.followup)
