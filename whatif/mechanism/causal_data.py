"""Training intervention labels and a new held-out CoPhy physical-structure set."""
from functools import partial
import multiprocessing as mp
import numpy as np
from .collect import ROOT,SPLITS,tower


def main():
    directory=ROOT/'data/mechanism/directed'
    directory.mkdir(parents=True,exist_ok=True)
    with mp.Pool(20) as pool:
        for split,(start,n) in [('train',SPLITS['train']),('test',(1323000,160))]:
            ids=range(start,start+n)
            rows=pool.map(partial(tower,deletion=True),ids)
            np.savez_compressed(directory/f'excision-{split}.npz',seed=[s for s,_ in rows],
                                impact=[y for _,y in rows])
            if split=='test':
                data=pool.map(tower,ids)
                np.savez_compressed(directory/'cophy-test.npz',**{k:np.asarray([r[k] for r in data],
                                    dtype=np.int64 if k=='seed' else np.float32) for k in data[0]})
            print(split,n,flush=True)


if __name__=='__main__':
    main()
