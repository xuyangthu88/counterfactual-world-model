"""Window contacts for the actual held-out contingent programs, replaying their selected suffix."""
import argparse
import multiprocessing as mp
import numpy as np
from whatif.paired.data import save
from .native import DOMAINS,OUT,DATA,branch

def trace(item):
    dom,seed,second=item
    rows=[branch(dom,seed,'test','early',0,a,continuation=int(b)) for a,b in [(0,0),*enumerate(second)]]
    return dict(seed=seed,contact=np.stack([r['contact'] for r in rows]),valid=np.stack([r['valid'] for r in rows]),
                outcome=np.stack([r['outcome'] for r in rows]))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=48)
    with mp.Pool(p.parse_args().workers,maxtasksperchild=48) as pool:
        for dom in DOMAINS:
            suffix=np.load(OUT/f'follow-{dom}-test.npy')
            with np.load(DATA/f'backup-{dom}-test.npz') as f:
                rows=pool.map(trace,[(dom,int(s),v) for s,v in zip(f['seed'],suffix)],chunksize=2)
                if not np.allclose(np.stack([r['outcome'] for r in rows]),f['outcome'],atol=1e-6):
                    raise ValueError('contingent replay mismatch')
            save(DATA/f'backup-contact-{dom}.npz',rows)
            print(dom,len(rows),flush=True)
