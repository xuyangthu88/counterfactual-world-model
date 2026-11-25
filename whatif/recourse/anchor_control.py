"""Evaluate original-policy preservation and replanning on a fresh registered cohort."""
import argparse
import json
import torch
from whatif.feedback.fit import restore as previous
from whatif.feedback.model import preference
from .anchor import restore as anchored,full_feature,data
from .backup import restore as backed
from .fit import restore,indices
from .load import read,take,early,late
from .native import OUT,DOMAINS,BANK
from .anchor_native import BASE

@torch.no_grad()
def select():
    net=anchored('cuda')
    records={}
    for split in ('val','test'):
        records[split]={}
        for dom in DOMAINS:
            x,y,v=data(dom,split,'cuda')
            s=y[...,0]
            ret=s+.5*y[...,1]+.25*y[...,2]
            p,atom,_=net.paired(x,net(x))
            rmse=((p[:,1:]@net.delta-ret[:,1:]+ret[:,:1]).square().mean().sqrt().item())
            label=ret[:,1:]-ret[:,:1]
            dist=p[:,1:]
            crps=((dist*(net.delta-label[...,None]).abs()).sum(-1)
                  -.5*((dist@(net.delta[:,None]-net.delta).abs())*dist).sum(-1)).mean().item()
            arms={}
            for k,values in net.decisions(x).items():
                choice=values.argmax(1)
                success=s.gather(1,choice[:,None])[:,0]
                arms[k]=dict(success=int(success.sum()),rescued=int((success>s[:,0]).sum()),
                             broken=int((success<s[:,0]).sum()),G=ret.gather(1,choice[:,None]).mean().item(),
                             original=int((choice==0).sum()))
            from sklearn.metrics import average_precision_score
            mass=p[:,1:].unflatten(-1,(3,17)).sum(-1)
            events={}
            for name,i,yes in [('rescue',2,s[:,1:]>s[:,:1]),('break',0,s[:,1:]<s[:,:1])]:
                prob=mass[...,i]
                events[name]=dict(rate=yes.float().mean().item(),ap=float(average_precision_score(
                    yes.flatten().cpu(),prob.flatten().cpu())),brier=(yes.float()-prob).square().mean().item())
            records[split][dom]=dict(reference=int(s[:,0].sum()),arms=arms,return_rmse=rmse,return_crps=crps,
                                     events=events)
        if split=='val':
            scores=records['val']
            records['selected']=max(scores[DOMAINS[0]]['arms'],key=lambda a:(
                sum(scores[d]['arms'][a]['success'] for d in DOMAINS),sum(scores[d]['arms'][a]['G'] for d in DOMAINS)))
    (OUT/'anchor-panel.json').write_text(json.dumps(records,indent=2))

@torch.no_grad()
def first():
    net=restore('cuda')
    backup=backed('cuda')
    anchor=anchored('cuda')
    old=previous('cuda')
    result={}
    for dom in DOMAINS:
        d=read(dom,'anchor-observe','cuda')
        selections={}
        for ix in torch.arange(len(d['x']),device='cuda').split(24):
            row=take(d,ix)
            de=early(row)
            z,_=net.encode(de,8)
            full=full_feature(net,row)
            plans={}
            for mode,v in anchor.decisions(torch.cat([full,z],1)).items():
                index=v.argmax(1)
                # Program zero is the actual original-policy continuation.
                plans[mode]=list(zip((index-1).clamp_min(0).tolist(),(index!=0).int().tolist()))
            plans['anchor_both_marginal']=plans['anchor_marginal']
            direct=net.first_scores(de)
            for name,v in [(key,direct[key]) for key in ('pooled','marginal','mean','law','kl')]+[('backup_mean',backup.scores(backup(z))['backup_mean'])]:
                plans[name]=[(a,1) for a in v.argmax(1).tolist()]
            o=old(dict(de,weights=net.bank[None].expand(len(ix),-1,-1)))
            i,j=indices(len(BANK),'cuda')
            p=old.contrasts(o,i,j,row['role'][...,0])[2]
            plans['stage23']=[(a,0) for a in preference(old,o,i,j,p)['mean'].argmax(1).tolist()]
            plans['actor']=[(0,0)]*len(ix)
            plans['fixed']=[(1,0)]*len(ix)
            for key,v in plans.items():
                selections.setdefault(key,[]).extend(v)
        result[dom]=selections
    (OUT/'anchor-first.json').write_text(json.dumps(result))

@torch.no_grad()
def second():
    net=restore('cuda')
    first=json.loads((OUT/'anchor-first.json').read_text())
    result={}
    for dom in DOMAINS:
        d=read(dom,'anchor-live','cuda')
        selected={}
        for ix in torch.arange(len(d['x']),device='cuda').split(64):
            row=take(d,ix)
            scores=net.last_scores(late(row),row['first'])
            for mode in ('marginal','mean','law','kl'):
                b=torch.where(row['midscore'][:,0].bool(),row['first'],scores[mode].argmax(1))
                selected.update({(s,a,mode):j for s,a,j in zip(row['seed'].tolist(),row['first'].tolist(),b.tolist())})
        result[dom]={}
        for k,v in first[dom].items():
            mode='marginal' if k=='anchor_both_marginal' else (k if k in ('marginal','law','kl') else 'mean')
            result[dom][k]=[[a,selected[BASE+i,a,mode] if can_replan else a] for i,(a,can_replan) in enumerate(v)]
    (OUT/'anchor-second.json').write_text(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=['select','first','second'])
    torch.set_num_threads(2)
    {'select':select,'first':first,'second':second}[p.parse_args().job]()
