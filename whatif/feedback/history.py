"""Paired native continuations after an observed, common feedback trajectory."""
import argparse
import json
import multiprocessing as mp
import numpy as np
from .native import DATA,OUT,DOMAINS,episode,save

SPLITS = {'train':(4300000,2304),'val':(4400000,192),'test':(4500000,192),'control':(3300000,192)}
PREFIX = 18

def search(pool):
    start = np.array(json.loads((OUT/'expanded-bank.json').read_text())['bank'])
    weights = list(start)
    rng = np.random.default_rng(271)
    def measure(params,seed,n):
        items = [(d,s,'train',p,'score',PREFIX) for p in params for d in DOMAINS for s in range(seed,seed+n)]
        y = np.stack(pool.map(episode,items,chunksize=3)).reshape(len(params),2,n,3)
        return (y@[1.,.5,.25]).mean(2),y[...,0].mean(2)
    log = []
    for line,mix in enumerate(([.85,.15],[.15,.85])):
        center = start[line].copy()
        std = np.array([.38,.4,.16,.16,.14,.3,.4,.4,.25,.2,.2])
        for gen in range(3):
            noise = rng.normal(size=(7,len(center)))
            candidate = np.clip(center+np.r_[noise,-noise]*std,-1.3,1.3)
            score,succ = measure(candidate,4700000+1000*(3*line+gen),40)
            order = np.argsort(-(score@mix))
            elite = candidate[order[:4]]
            center = elite.mean(0)
            std = .65*std+.35*elite.std(0)
            weights.extend([candidate[order[0]],center.copy()])
            log.append(dict(line=line,gen=gen,params=candidate.tolist(),score=score.tolist(),success=succ.tolist()))
    params=np.stack(weights)
    score,succ=measure(params,4800000,160)
    ids=[0]
    for rank in range(len(params)):
        for mix in ([.95,.05],[.05,.95],[.5,.5]):
            ix=int(np.argsort(-(score@mix))[rank])
            if ix not in ids:
                ids.append(ix)
            if len(ids)==7:
                result=dict(bank=params[ids].tolist(),ids=ids,params=params.tolist(),score=score.tolist(),
                            screen_success=succ.tolist(),history=log)
                (OUT/'history-bank.json').write_text(json.dumps(result,indent=1))
                print('midpoint training search',succ[ids],flush=True)
                return

def collect(pool):
    params=np.array(json.loads((OUT/'history-bank.json').read_text())['bank'])
    for split,(base,n) in SPLITS.items():
        for dom in DOMAINS:
            path=DATA/f'history-{dom}-{split}.npz'
            if split=='control':
                items=[(dom,s,split,None,'observe',PREFIX) for s in range(base,base+n)]
                save(path,pool.map(episode,items,chunksize=3))
                continue
            items=[(dom,s,split,w,'panel',PREFIX) for s in range(base,base+n) for w in params]
            raw=pool.map(episode,items,chunksize=3)
            rows=[]
            for index in range(0,len(raw),len(params)):
                group=raw[index:index+len(params)]
                row=group[0].copy()
                for k in ('a','y','e','r','future','contact','outcome','native'):
                    row[k]=np.stack([g[k] for g in group])
                if any(not np.allclose(g['hy'],row['hy'],atol=1e-5) for g in group):
                    raise ValueError('different measured histories')
                rows.append(row)
            save(path,rows)
            print(dom,split,n,flush=True)

def execute(pool):
    params=np.array(json.loads((OUT/'history-bank.json').read_text())['bank'])
    choices=json.loads((OUT/'history-choices.json').read_text())
    base,_=SPLITS['control']
    result={}
    for dom,arms in choices.items():
        keys=sorted({(base+i,j) for vs in arms.values() for i,j in enumerate(vs)})
        items=[(dom,s,'control',params[j],'score',PREFIX) for s,j in keys]
        observed=dict(zip(keys,pool.map(episode,items,chunksize=2)))
        result[dom]={arm:[observed[base+i,j].tolist() for i,j in enumerate(vs)] for arm,vs in arms.items()}
    (OUT/'history-episodes.json').write_text(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=['collect','execute'])
    p.add_argument('--workers',type=int,default=72)
    args=p.parse_args()
    with mp.Pool(args.workers,maxtasksperchild=80) as pool:
        if args.job=='collect':
            search(pool)
            collect(pool)
        else:
            execute(pool)
