"""Independent closed-loop cohort for observing evidence about the shared response."""
import json
import argparse
import multiprocessing as mp
from whatif.paired.data import save
from .native import DATA,OUT,DOMAINS,run

BASE,N=7600000,1024

def collect(job,workers):
    results={}
    with mp.Pool(workers,maxtasksperchild=64) as pool:
        for dom in DOMAINS:
            if job=='initial':
                rows=pool.map(run,[(dom,s,'control',0,0,'initial') for s in range(BASE,BASE+N)])
                save(DATA/f'evidence-initial-{dom}.npz',rows)
            elif job=='mid':
                ids=json.loads((OUT/'evidence-first.json').read_text())[dom]
                rows=pool.map(run,[(dom,BASE+i,'control',max(k-1,0),0,'mid') for i,k in enumerate(ids)])
                save(DATA/f'evidence-mid-{dom}.npz',rows)
            else:
                arms=json.loads((OUT/'evidence-second.json').read_text())[dom]
                keys=sorted({(BASE+i,a,b) for seq in arms.values() for i,(a,b) in enumerate(seq)})
                v=dict(zip(keys,pool.map(run,[(dom,s,'control',a,b,'score') for s,a,b in keys],chunksize=2)))
                results[dom]={m:[v[BASE+i,a,b].tolist() for i,(a,b) in enumerate(seq)] for m,seq in arms.items()}
            print(job,dom,flush=True)
    if job=='execute':
        (OUT/'evidence-episodes.json').write_text(json.dumps(results))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=['initial','mid','execute'])
    p.add_argument('--workers',type=int,default=56)
    a=p.parse_args()
    collect(a.job,a.workers)
