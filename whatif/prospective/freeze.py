import csv
import json
from .collect import OUT


def freeze():
    rows=[]
    for p in sorted(OUT.glob('v-*.csv')):
        if not p.name.endswith('-steps.csv'):
            with p.open() as f:rows.extend(csv.DictReader(f))
    if not rows:
        with (OUT/'validation.csv').open() as f:rows=list(csv.DictReader(f))
    def score(task,method):
        group=[r for r in rows if (r['task'],r['policy'])==(task,method)]
        assert len({r['seed'] for r in group})==len(group)==64
        cap=420 if task=='causalworld' else 280
        return sum(.60*int(r['success'])+.35*float(r['coverage'])-.05*int(r['steps'])/cap for r in group)/len(group)
    tasks=['causalworld','pusht']
    result=dict(model=max(['mass','gain'],key=lambda p:sum(score(t,p) for t in tasks)))
    for t in tasks:
        result[t]={'pets':max(['pets5','pets15'],key=lambda p:score(t,p)),
                   'validation':{p:score(t,p) for p in ['mass','gain','pets5','pets15']}}
    (OUT/'freeze.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':freeze()
