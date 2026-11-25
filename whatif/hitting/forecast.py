"""Held-out future, paired differences, timing dependence, and matched policy outcomes."""
import json
import math
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from .data import data,take
from .native import OUT,DOMAINS
from .fit import restore,pairs
from .policy import selector,features,MODES

def crps(err,sd):
    z=err/sd
    return err*(2*torch.special.ndtr(z)-1)+sd*(math.sqrt(2/math.pi)*(-z.square()/2).exp()-1/math.sqrt(math.pi))

def auc_event(pred,real):
    f=real.float()
    return dict(rate=f.mean().item(),mean=pred.mean().item(),brier=(pred-f).square().mean().item(),
                ap=float(average_precision_score(real.flatten().cpu(),pred.flatten().cpu())))

def law_metrics(law,target,i,j):
    n=law.size(-1)
    obs=target[:,i]*n+target[:,j]
    nll=-law.flatten(-2).gather(-1,obs[...,None]).clamp_min(1e-9).log().mean()
    axis=torch.arange(n,device=law.device)
    assign=F.one_hot((axis[:,None]-axis[None,:]+n-1).flatten(),2*n-1).to(law)
    pdf=law.flatten(-2)@assign
    labels=F.one_hot(target[:,i]-target[:,j]+n-1,2*n-1)
    ranked=(pdf.cumsum(-1)-labels.cumsum(-1)).square().sum(-1).mean()
    si,sj=target[:,i]<n-1,target[:,j]<n-1
    # Orient rescue/break consistently as changing from program 0 to program j.
    ref=i==0
    resc=auc_event(law[:,ref,-1,:-1].sum(-1),~si[:,ref]&sj[:,ref])
    brk=auc_event(law[:,ref,:-1,-1].sum(-1),si[:,ref]&~sj[:,ref])
    return dict(nll=nll.item(),difference_crps=ranked.item(),rescue=resc,broken=brk)

@torch.no_grad()
def evaluate(kind,dom,early):
    dev='cuda'
    phase='early' if early else 'late'
    d=data(dom,phase,'test',dev)
    net=restore(kind,early,dev)
    s=selector(early,dev) if kind=='joint' else None
    parts=[]
    unit=100 if dom=='pusht' else 65
    k=d['a'].size(1)
    i,j=pairs(k,dev)
    methods={}
    for ix in torch.arange(len(d['x']),device=dev).split(48):
        row=take(d,ix)
        o,_=net(row,early)
        chosen={'direct':(1-o['p'][...,-1]).argmax(1)}
        if s is not None:
            f=features(net,row,early)
            chosen.update({m:s(f,m).argmax(1) for m in MODES})
        for m,idx in chosen.items():
            methods.setdefault(m,[]).append(idx)
        mu=o['mu']
        p=o['p']
        independent=p[:,i,:,None]*p[:,j,None,:]
        law=net.head.coupling(o,i,j) if kind=='joint' else independent
        dm,sd=net.head.normal(o,i,j,independent=kind!='joint')
        _,sid=net.head.normal(o,i,j,independent=True)
        parts.append(dict(mu=mu,dm=dm,sd=sd,sid=sid,p=p,law=law,ind=independent,score=o['score']))
    r={k:torch.cat([v[k] for v in parts]) for k in parts[0]}
    p,law,ind=r['p'],r['law'],r['ind']
    bin=d['bin']
    out=d['outcome']
    y=d['value']
    err=r['mu']-y
    de=r['dm']-(y[:,i]-y[:,j])
    target=out[...,0]
    success=1-p[...,-1]
    linear=lambda v:v.detach().cpu().tolist()
    stats=dict(layouts=len(d['seed'].unique()),states=len(y),branches=k,
        hit_nll=-p.gather(-1,bin[...,None]).clamp_min(1e-9).log().mean().item(),
        success_brier=(success-target).square().mean().item(),
        success_effect_rmse=((success[:,i]-success[:,j])-(target[:,i]-target[:,j])).square().mean().sqrt().item(),
        peak_last_rmse=linear((r['score']-out[...,1:]).square().mean((0,1)).sqrt()),
        pos=linear(unit*err[...,:3].square().sum(-1).mean((0,1)).sqrt()),
        pose_mae=linear(err[...,:9].abs().mean((0,1,3))),
        reward=linear(err[...,9].square().mean((0,1)).sqrt()),
        effect=linear(unit*de[...,:3].square().sum(-1).mean((0,1)).sqrt()),
        reward_effect=linear(de[...,9].square().mean((0,1)).sqrt()))
    for name,width in [('paired',r['sd']),('independent',r['sid'])]:
        loss=crps(de,width)
        stats[name]=dict(crps_xyz=linear(unit*loss[...,:3].mean((0,1,3))),
                         crps_reward=linear(loss[...,9].mean((0,1))))
    stats['law']=law_metrics(law,bin,i,j)
    stats['independence']=law_metrics(ind,bin,i,j)
    stats['max_marginal_error']=max((law.sum(-1)-p[:,i]).abs().max().item(),(law.sum(-2)-p[:,j]).abs().max().item())
    arms={}
    for m,choices in methods.items():
        idx=torch.cat(choices)
        reached=target.gather(1,idx[:,None])[:,0]
        arms[m]=dict(success=int(reached.sum()),rescued=int((reached>target[:,0]).sum()),broken=int((reached<target[:,0]).sum()),
                     original=int(target[:,0].sum()))
    stats['selections']=arms
    return stats

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['early','late'])
    phase=p.parse_args().phase
    torch.set_num_threads(2)
    records={dom:{k:evaluate(k,dom,phase=='early') for k in ('joint','marginal')} for dom in DOMAINS}
    (OUT/f'{phase}-forecast.json').write_text(json.dumps(records,indent=2,allow_nan=False))
