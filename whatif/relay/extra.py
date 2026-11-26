"""Fresh physical interventions reserved for the trajectory readout comparison."""
import argparse
import multiprocessing as mp
from functools import partial
from whatif.mechanism.collect import tower
from whatif.paired.data import save
from .native import DOMAINS,DATA,panel

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=60)
    args=p.parse_args()
    with mp.Pool(args.workers,maxtasksperchild=64) as pool:
        base=11900000
        save(DATA/'cophy-extra.npz',pool.map(partial(tower,trace=True),range(base,base+256),chunksize=2))
        for dom in DOMAINS:
            rows=pool.map(panel,[(dom,s,'test','late') for s in range(base+10000,base+10320)],chunksize=2)
            save(DATA/f'{dom}-late-extra.npz',rows)
            print(dom,len(rows),flush=True)
