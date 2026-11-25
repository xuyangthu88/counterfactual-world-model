"""Realized continuation returns for a fixed learned second decision."""
import argparse
import multiprocessing as mp
import numpy as np
from whatif.paired.data import save
from .native import BANK,DATA,OUT,DOMAINS,rollout

SPLITS={'train':(5700000,1792),'val':(5800000,192),'test':(5900000,192)}
def job(pool,phase):
    for split,(base,n) in SPLITS.items():
        for dom in DOMAINS:
            path=DATA/f'backup-{dom}-{split}.npz'
            keys=[(s,a) for s in range(base,base+n) for a in range(len(BANK))]
            if phase=='observe':
                rows=pool.map(rollout,[(dom,s,split,a,0,'middle') for s,a in keys],chunksize=3)
                save(path,rows)
            else:
                selected=np.load(OUT/f'continuation-{dom}-{split}.npy')
                items=[(dom,s,split,a,int(b),'panel') for (s,a),b in zip(keys,selected,strict=True)]
                raw=pool.map(rollout,items,chunksize=2)
                rows=[{k:r[k] for k in ('seed','first','outcome','native','final')} for r in raw]
                save(DATA/f'target-{dom}-{split}.npz',rows)
            print(phase,dom,split,n,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['observe','targets'])
    p.add_argument('--workers',type=int,default=72)
    a=p.parse_args()
    with mp.Pool(a.workers,maxtasksperchild=80) as pool:
        job(pool,a.phase)
