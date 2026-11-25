"""Freeze one planning weight on separate closed-loop validation scenes."""
import csv
import json
from .models import OUT, TASKS


def select():
    with (OUT/'value-validation.csv').open() as f:
        rows=list(csv.DictReader(f))
    last={(r['task'],r['policy'],r['scene']):r for r in rows}
    policies=['ours','value0.4','value0.8']
    scores={}
    for policy in policies:
        tasks=[]
        for name in TASKS:
            group=[r for (t,p,_),r in last.items() if t==name and p==policy]
            tasks.append(sum(float(r['coverage'])+int(r['success']) for r in group)/(2*len(group)))
        scores[policy]=sum(tasks)/len(tasks)
    chosen=max(policies,key=scores.get)
    extra=max(policies[1:],key=scores.get)
    (OUT/'value_selection.json').write_text(json.dumps(dict(policy=chosen,extra=extra,scores=scores),indent=2)+'\n')
    print(extra)


if __name__=='__main__':
    select()
