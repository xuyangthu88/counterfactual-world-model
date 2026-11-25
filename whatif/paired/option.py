"""Native closed-loop selection of the commitment duration and minimum switch gain."""
import argparse
import json
import multiprocessing as mp
from itertools import product
import numpy as np
from .data import OUT

VAL_START,TEST_START=2223000,2323000
CONFIGS=[dict(cost=c,hold=h,prior=.05,risk=.15) for c,h in product([.06,.16],[5,11])]
CONFIRM=['actor','stage21','marginal','pair_law','pair_kl','combined_law','combined_kl']


def choose(o,prior,policy,config,batched=False):
    from .decision import statistics,KL_WEIGHT
    import torch
    v=statistics(o)
    p=o['mass']
    if policy=='marginal':
        marginal=o['marginal'].sigmoid()
        absolute=marginal[...,0]+.2*(.5*marginal[...,1]+.25*marginal[...,2])
        score=absolute-absolute[:,:1]
    else:
        score=p[...,2]-p[...,1]+.2*(p*o['mu']).sum(-1)
        if policy.endswith('_kl'):
            score=score+KL_WEIGHT*v['kl']
        if policy!='pair_mean':
            score=score.masked_fill(v['loss']>config['risk'],-torch.inf)
    score=score+config['prior']*prior.clamp(min=-6)-config['cost']
    score[:,0]=0
    selected=score.argmax(-1)
    return selected if batched else selected.item()


def run(phase,workers,sim):
    from .control import batch
    if phase=='validation':
        cases=[('pair_law',k,c) for k,c in enumerate(CONFIGS)]
        start,n=VAL_START,48
    else:
        chosen=json.loads((OUT/'option-selection.json').read_text())['selected']
        cases=[(p,0,chosen) for p in CONFIRM]
        start,n=TEST_START,128
    jobs=[(d,p,start+i,8,sim,dict(config=c,id=k)) for i in range(0,n,8)
          for d in ('causalworld','pusht') for p,k,c in cases]
    rows=[]
    with mp.get_context('fork').Pool(workers) as pool:
        for part in pool.imap_unordered(batch,jobs):
            rows.extend(part)
            print(phase,len(rows),flush=True)
    rows.sort(key=lambda r:(r['domain'],r['policy'],r['id'],r['seed']))
    (OUT/f'option-{phase}.json').write_text(json.dumps(rows,separators=(',',':')))
    if phase=='validation':
        summaries=[]
        for k,cfg in enumerate(CONFIGS):
            group=[r for r in rows if r['id']==k]
            summaries.append(dict(config=cfg,success=sum(r['success'] for r in group),
                                  terminal=float(np.mean([r['terminal'] for r in group]))))
        best=max(summaries,key=lambda s:(s['success'],s['terminal']))
        (OUT/'option-selection.json').write_text(json.dumps(dict(selected=best['config'],validation=summaries),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['validation','test'])
    p.add_argument('--sim-python',required=True)
    p.add_argument('--workers',type=int,default=20)
    args=p.parse_args()
    run(args.phase,args.workers,args.sim_python)
