"""Measure the unmodified factual policy as an explicit potential outcome."""
import argparse
import json
import multiprocessing as mp
import numpy as np
from .native import DATA,OUT,DOMAINS,rollout,save
from .backup_data import SPLITS

BASE=6100000
COUNT=512
def run(pool,job):
    for dom in DOMAINS:
        if job=='labels':
            for split,(base,n) in SPLITS.items():
                items=[(dom,s,split,0,0,'panel') for s in range(base,base+n)]
                rows=[{k:r[k] for k in ('seed','final','outcome','native')}
                      for r in pool.map(rollout,items,chunksize=2)]
                save(DATA/f'anchor-{dom}-{split}.npz',rows)
        elif job=='observe':
            items=[(dom,s,'control',0,0,'first') for s in range(BASE,BASE+COUNT)]
            save(DATA/f'{dom}-anchor-observe.npz',pool.map(rollout,items,chunksize=2))
        elif job=='middle':
            selections=json.loads((OUT/'anchor-first.json').read_text())[dom]
            keys=sorted({(BASE+i,a) for seq in selections.values() for i,(a,_) in enumerate(seq)})
            rows=pool.map(rollout,[(dom,s,'control',a,0,'middle') for s,a in keys],chunksize=2)
            save(DATA/f'{dom}-anchor-live.npz',rows)
        else:
            selections=json.loads((OUT/'anchor-second.json').read_text())[dom]
            keys=sorted({(BASE+i,a,b) for seq in selections.values() for i,(a,b) in enumerate(seq)})
            obs=dict(zip(keys,pool.map(rollout,[(dom,s,'control',a,b,'score') for s,a,b in keys],chunksize=2)))
            results={arm:[obs[BASE+i,a,b].tolist() for i,(a,b) in enumerate(seq)] for arm,seq in selections.items()}
            (OUT/f'anchor-{dom}-episodes.json').write_text(json.dumps(results))
        print(dom,job,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=['labels','observe','middle','execute'])
    p.add_argument('--workers',type=int,default=64)
    args=p.parse_args()
    with mp.Pool(args.workers,maxtasksperchild=72) as pool:
        run(pool,args.job)
