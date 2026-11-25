"""Choose using only observed prefixes. Native workers subsequently execute the selected branches."""
import argparse
import json
import torch
from whatif.feedback.fit import restore as previous_world
from whatif.feedback.model import preference
from .fit import restore,indices
from .backup import restore as backup_world
from .load import read,take,early,late
from .native import OUT,DOMAINS,BANK,START

@torch.no_grad()
def first():
    net=restore('cuda')
    backup=backup_world('cuda')
    old=previous_world('cuda')
    results={}
    for dom in DOMAINS:
        d=read(dom,'observe','cuda')
        picks={}
        for ix in torch.arange(len(d['seed']),device='cuda').split(20):
            row=take(d,ix)
            de=early(row)
            values=net.first_scores(de)
            z,_=net.encode(de,8)
            values.update(backup.scores(backup(z)))
            i,j=indices(len(BANK),'cuda')
            o=old(dict(de,weights=net.bank[None].expand(len(ix),-1,-1)))
            p=old.contrasts(o,i,j,row['role'][...,0])[2]
            values['stage23']=preference(old,o,i,j,p)['mean']
            for mode,v in values.items():
                selected=torch.where(row['alive'].bool(),v.argmax(1),0)
                picks.setdefault(mode,[]).extend(selected.tolist())
        picks['actor']=[0]*len(d['seed'])
        picks['fixed']=[1]*len(d['seed'])
        results[dom]=picks
    (OUT/'first.json').write_text(json.dumps(results))

@torch.no_grad()
def second():
    net=restore('cuda')
    firsts=json.loads((OUT/'first.json').read_text())
    base,n=START['control']
    results={}
    for dom in DOMAINS:
        d=read(dom,'live','cuda')
        lookup={(int(s),int(a)):j for j,(s,a) in enumerate(zip(d['seed'].tolist(),d['first'].tolist()))}
        scores={}
        for ix in torch.arange(len(d['seed']),device='cuda').split(48):
            row=take(d,ix)
            for m,v in net.last_scores(late(row),row['first']).items():
                b=torch.where(row['midscore'][:,0].bool(),row['first'],v.argmax(-1))
                scores.setdefault(m,[]).extend(b.tolist())
        picks={}
        for mode,ids in firsts[dom].items():
            if mode in ('actor','fixed','stage23'):
                picks[mode]=[[a,a] for a in ids]
            else:
                last='mean' if mode.startswith('backup_') else mode
                picks[mode]=[[a,scores[last][lookup[base+i,a]]] for i,a in enumerate(ids)]
        results[dom]=picks
    (OUT/'second.json').write_text(json.dumps(results))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['first','second'])
    torch.set_num_threads(2)
    (first if p.parse_args().stage=='first' else second)()
