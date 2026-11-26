"""Add native step labels and matched collision replays to the original scenarios."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from whatif.effects.collect import DATA as COARSE_DATA, DOMAINS, ROOT, items, pushing
from whatif.mechanism.collect import tower
from whatif.mechanism.native import Simulator

PROTOCOL = json.loads(Path(__file__).with_name('protocol.json').read_text())
DATA = ROOT/'data/shortstep'
OUT = ROOT/'results/shortstep'


def collect_tower(seed, split):
    times = np.array(PROTOCOL['cophy_clock'])
    row = tower(seed, height=5 if split=='transfer' else None, trajectory_pairs=True, windows=tuple(times))
    ends = np.searchsorted(times, (6,12,18,24,30))
    row['native_pose'] = np.asarray(row['traj'])
    row['native_edge'] = np.asarray(row['contact'])
    row['native_off'] = np.asarray(row['off'])
    row['fact'] = row['fact'][np.r_[0,ends+1]]
    row['traj'] = row['native_pose'][:,ends]
    row['contact'] = np.stack([row['native_edge'][:,a:b+1].max(1)
                               for a,b in zip(np.r_[0,ends[:-1]+1],ends)],1)
    row['off'] = row['native_off'][:,:,ends]
    return row


def collect_push(domain, seed, split):
    env_split = 'test' if split=='transfer' else 'val'
    row = pushing(domain,seed,env_split,native_trace=True)
    prefix = row.pop('prefix')
    rng = np.random.default_rng(seed ^ 19973)
    n = len(row['x'])
    i,j = np.tril_indices(n,-1)
    valid = (j==0)&row['role'][i].any(-1)
    touched = row['native_edge'][:,:,i,j].any(1)
    weight = (1+4*touched.any(0)+6*(touched.any(0)!=touched.all(0)))*valid
    pair = int(rng.choice(len(i),p=weight/weight.sum()))
    edited = []
    for action in row['a']:
        sim = Simulator(domain,seed,env_split)
        try:
            for p in prefix:
                sim.execute(p,landmarks=True)
            sim.disable_object_contact((i[pair],j[pair]))
            points = []
            sim.execute(action,landmarks=True,on_tick=lambda x,e,r: points.append(x[:,:3]))
            edited.append(np.stack(points))
        finally:
            sim.close()
    row['native_off'] = row['native_pose'][...,:3]-np.stack(edited)
    row['removed_pair'] = pair
    return row


def sample(item):
    domain,seed,split = item
    return collect_tower(seed,split) if domain=='cophy' else collect_push(domain,seed,split)


def write(data, domain, split, name=None):
    name = name or (PROTOCOL['test_split'] if split=='transfer' else split)
    fields = ['seed','native_pose','native_edge','native_off',
              *([] if domain=='cophy' else ['native_reward','removed_pair'])]
    old = ['seed','x','a','role','goal','fa','fact','fe','traj','contact',
           *(['off'] if domain=='cophy' else ['rewards'])]
    coarse = {k:np.asarray([d[k] for d in data]) for k in old}
    path = COARSE_DATA/f'{domain}-{name}.npz'
    if split=='transfer':
        np.savez_compressed(path,**coarse)
    else:
        with np.load(path) as archived:
            for k,v in coarse.items():
                if not np.allclose(archived[k],v,rtol=0,atol=1e-5):
                    raise ValueError(f'Coarse replay disagreed with {path.name} at {k}')
    path = DATA/f'{domain}-{name}.npz'
    np.savez_compressed(path,**{k:np.asarray([d[k] for d in data]) for k in fields})
    print(path.relative_to(ROOT),len(data),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('domain',choices=DOMAINS)
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--split',choices=('train','val','transfer'))
    p.add_argument('--transfer-base',type=int)
    p.add_argument('--transfer-name')
    a=p.parse_args()
    if (a.transfer_base is None)!=(a.transfer_name is None) or (a.transfer_name and a.split!='transfer'):
        p.error('A named transfer split needs both --transfer-base and --transfer-name with --split transfer')
    DATA.mkdir(exist_ok=True)
    transfer_base = PROTOCOL['layout_base'] if a.transfer_base is None else a.transfer_base
    with mp.Pool(a.workers,maxtasksperchild=32) as pool:
        for split in ([a.split] if a.split else ('train','val','transfer')):
            base = transfer_base if split=='transfer' else None
            rows=pool.map(sample,items(a.domain,split,base),chunksize=2)
            write(rows,a.domain,split,a.transfer_name)
