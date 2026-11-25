"""Measure endpoint response to deleting one physical pair interaction in BlocktowerCF."""
from functools import partial
import multiprocessing as mp
import numpy as np
from .collect import ROOT,SPLITS,tower


if __name__ == '__main__':
    for split in ('val','test'):
        start,n = SPLITS[split]
        with mp.Pool(16) as pool:
            rows = pool.map(partial(tower,deletion=True),range(start,start+n))
        ids,impact = zip(*rows)
        np.savez_compressed(ROOT/f'data/mechanism/excision-{split}.npz',seed=ids,impact=impact)
        print(split,n,flush=True)
