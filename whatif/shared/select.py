"""Two finalists per comparator on source validation; freeze on fresh transfer validation."""
import argparse
import csv
import json
from collections import defaultdict
from .actions import OUT

FAMILY={'graph':'graph','cemgraph':'graph','returngraph':'graph','pets':'pets','returnpets':'pets','cophy':'cophy','physics':'physics'}


def select(mode):
    stats={}
    for task in ('causalworld','pusht'):
        pool=defaultdict(list)
        paths=sorted(OUT.glob(('val-' if mode=='shortlist' else 'confirm-')+'*.csv'))
        if not paths:paths=[OUT/'validation.csv']
        for f in paths:
            if f.name.endswith('-steps.csv'):continue
            for r in csv.DictReader(f.open()):
                if f.name=='validation.csv' and r['split']!=('source' if mode=='shortlist' else 'transfer'):continue
                if r['task']==task:
                    pool[FAMILY[r['policy']],float(r['weight']),int(r['look'])].append(r)
        stats[task]={}
        for model in ('graph','pets','cophy','physics'):
            candidates=[]
            for (family,w,h),rows in pool.items():
                if family!=model:continue
                n=len(rows)
                if len({r['seed'] for r in rows})!=n:raise ValueError('Duplicated layout/config')
                if n!=(48 if mode=='shortlist' else 64):raise ValueError((model,w,h,n))
                q=sum(.60*int(r['success'])+.35*float(r['coverage'])-.05*int(r['steps'])/(420 if task=='causalworld' else 280) for r in rows)/n
                candidates.append((q,w,h))
            if not candidates:raise ValueError((model,task))
            candidates.sort(reverse=True)
            stats[task][model]=[[v[1],v[2]] for v in candidates[:2]] if mode=='shortlist' else list(candidates[0][1:])
    (OUT/('shortlist.json' if mode=='shortlist' else 'planning.json')).write_text(json.dumps(stats,indent=2)+'\n')
    print(stats)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['shortlist','freeze']);select(ap.parse_args().mode)
