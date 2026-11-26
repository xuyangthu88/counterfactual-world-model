"""Measure the eight complete programs under the validation-selected Stage 26 continuation."""
import argparse
import multiprocessing as mp
import numpy as np
from whatif.hitting.native import DATA as OLD,run
from whatif.paired.data import save
from .native import DATA,OUT,DOMAINS,BANK

SIZES={'world':2048,'policy':1024,'val':224}
def middles(dom,key):
    return DATA/f'backup-mid-{dom}.npz' if key=='test' else OLD/f'mid-{dom}-{key}.npz'

def prepare(workers):
    with mp.Pool(workers,maxtasksperchild=64) as pool:
        for dom in DOMAINS:
            jobs=[(dom,s,'test',a,0,'mid') for s in range(12200000,12200256) for a in range(len(BANK))]
            save(middles(dom,'test'),pool.map(run,jobs,chunksize=2))

def continuation(test=False):
    import torch
    from whatif.hitting.data import load_np,take
    from .data import common
    from .model import restore
    from .policy import features,load
    net,policy=restore('dynamic','cuda'),load('cuda')
    with torch.no_grad():
        for key,n in ({'test':256} if test else SIZES).items():
            for dom in DOMAINS:
                r=load_np(middles(dom,key))
                r={k:v[:n*len(BANK)] for k,v in r.items()}
                d=common(r,dom,'late','cuda',live=True)
                ix=torch.arange(len(d['x']),device='cuda')
                choice=torch.cat([policy(features(net,take(d,i)),'law')[0].argmax(1) for i in ix.split(32)]).cpu().numpy()
                choice=np.where(r['mid_alive'],choice,r['first']).astype(np.int64)
                np.save(OUT/f'follow-{dom}-{key}.npy',choice.reshape(n,len(BANK)))
                print('choices',dom,key,n,flush=True)

def measure(workers,hold=False,test=False):
    with mp.Pool(workers,maxtasksperchild=64) as pool:
        for key,n in ({'test':256} if test else SIZES).items():
            for dom in DOMAINS:
                with np.load(middles(dom,key)) as f:
                    seeds=f['seed'][::len(BANK)][:n]
                second=np.tile(np.arange(len(BANK)),(n,1)) if hold else np.load(OUT/f'follow-{dom}-{key}.npy')
                split=key if key in ('val','test') else 'train'
                jobs=[(dom,int(s),split,a,int(b),'target') for s,bs in zip(seeds,second) for a,b in
                      (list(enumerate(bs)) if hold else [(0,0),*enumerate(bs)])]
                rows=pool.map(run,jobs,chunksize=2)
                k=len(BANK)+int(not hold)
                archive=[dict(seed=int(s),**{j:np.stack([r[j] for r in rows[t*k:(t+1)*k]]) for j in ('traj','rewards','outcome','hit')})
                         for t,s in enumerate(seeds)]
                prefix='hold' if hold else 'backup'
                save(DATA/f'{prefix}-{dom}-{key}.npz',archive)
                print('measured',dom,key,n,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=['prepare','choose','measure','hold'])
    p.add_argument('--workers',type=int,default=60)
    p.add_argument('--test',action='store_true')
    a=p.parse_args()
    if a.job=='prepare':
        prepare(a.workers)
    elif a.job=='choose':
        continuation(a.test)
    else:
        measure(a.workers,a.job=='hold',a.test)
