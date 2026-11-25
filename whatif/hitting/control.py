"""Prospective, real two-decision rollout with fixed programs and shared downstream controls."""
import argparse
import json
import torch
from whatif.recourse.anchor import restore as saved_anchor,full_feature
from whatif.recourse.fit import restore as saved_world
from .native import OUT,CONTROL,DOMAINS
from .data import data,take
from .fit import restore
from .policy import selector,MODES,features
from .continuations import winner

READOUT=('unpaired','direct',*MODES)

def scores(joint,marginal,policy,row,early):
    f=features(joint,row,early)
    raw=marginal(row,early)[0]['p']
    values=dict(unpaired=1-raw[...,-1],direct=f['m'][...,0])
    values.update({m:policy(f,m) for m in MODES})
    return {m:z.argmax(1).tolist() for m,z in values.items()}

def plans(index,mode):
    return [[max(k-1,0),mode if k else 'hold'] for k in index]

@torch.no_grad()
def run(first):
    torch.set_num_threads(2)
    dev='cuda'
    early_mode,late_mode=winner(True),winner(False)
    joint=restore('joint',first,dev)
    marginal=restore('marginal',first,dev)
    policy=selector(first,dev)
    old=saved_world(dev)
    records={}
    if first:
        anchor=saved_anchor(dev)
    else:
        orders=json.loads((OUT/'first.json').read_text())
    for dom in DOMAINS:
        d=data(dom,'early' if first else 'late','control',dev,live=True)
        r={}
        for ix in torch.arange(len(d['seed']),device=dev).split(48):
            row=take(d,ix)
            choices=scores(joint,marginal,policy,row,first)
            if first:
                batch={'first_'+m:plans(choices[m],late_mode) for m in READOUT}
                batch.update({'both_'+m:plans(choices[m],m) for m in READOUT})
                batch.update({'late_'+m:plans(choices[early_mode],m) for m in READOUT})
                batch['selected']=plans(choices[early_mode],late_mode)
                de=dict(row,a=row['a'][:,1:])
                full=full_feature(old,de)
                z=old.encode(de,8)[0]
                k=anchor.decisions(torch.cat([full,z],1))['anchor_marginal'].argmax(1)
                batch.update(stage24=plans(k.tolist(),'stage24'),actor=[[0,'hold']]*len(ix),fixed=[[1,'hold']]*len(ix))
                for m,q in batch.items():
                    r.setdefault(m,[]).extend(q)
            else:
                choices['stage24']=old.last_scores(row,row['first'])['mean'].argmax(1).tolist()
                for j,(s,a) in enumerate(zip(row['seed'].tolist(),row['first'].tolist())):
                    r[s,a]={m:v[j] for m,v in choices.items()}
        if first:
            records[dom]=r
        else:
            base,_=CONTROL
            records[dom]={m:[[a,a if mode=='hold' else r[base+i,a][mode]] for i,(a,mode) in enumerate(seq)]
                          for m,seq in orders[dom].items()}
    name='first' if first else 'second'
    (OUT/f'{name}.json').write_text(json.dumps(records))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['first','second'])
    run(p.parse_args().phase=='first')
