"""Learn the effect of recourse at its actual time using matched hold/observe-and-replan labels."""
import argparse
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import mlp
from whatif.hitting.data import load_np,take
from whatif.hitting.fit import pairs
from .native import BANK,DATA,OUT,DOMAINS,ENDS
from .data import common
from .model import restore,coupled,difference
from .fit import mixture_nll
from .policy import features,Decision,MODES,loss as decision_loss
from .backup_data import middles

def attach(net):
    net.core.recourse_source=mlp(24+9+9+11,24,64).to(next(net.parameters()).device)
    net.backup_heads=nn.ModuleDict({k:copy.deepcopy(getattr(net,k)) for k in ('lift','regime','hit','point','emission','score')})
    return net

def world(dev):
    net=attach(restore('dynamic-full',dev))
    net.load_state_dict(torch.load(OUT/'recourse.pt',map_location=dev)['model'])
    return net.eval()

def programs(d):
    d=dict(d)
    for k in ('a','program'):
        d[k]=torch.cat([d[k],d[k]],1)
    k=d['a'].size(1)//2
    d['adaptive']=d['a'].new_tensor([0]*k+[1]*k).expand(len(d['a']),-1)
    return d

def subset(d,idx):
    return {k:(v[:,idx] if k in ('a','program','adaptive','value','outcome','bin') else v) for k,v in d.items()}

def menu(d):
    k=len(BANK)
    return subset(d,[0,*range(k,2*k)])

def records(dom,split,dev):
    r=load_np(DATA/f'backup-{dom}-{split}.npz')
    hold=load_np(DATA/f'hold-{dom}-{split}.npz')
    n=len(r['seed'])
    with np.load(middles(dom,split)) as f:
        raw={k:f[k][:n*len(BANK):len(BANK)] for k in ('seed','role','goal','f0','fa','f1','alive')}
    if not (np.array_equal(r['seed'],raw['seed']) and np.array_equal(hold['seed'],r['seed'])):
        raise ValueError('unaligned branch labels')
    raw.update(x=raw['f1'],hx=raw['f0'][:,None],ha=raw['fa'][:,None],hy=raw['f1'][:,None])
    d=programs(common(raw,dom,'early',dev,live=True))
    def join(k):
        return torch.as_tensor(np.concatenate([hold[k],r[k][:,1:]],1),device=dev)
    role=d['role'][:,None,None,:,0:1]
    x=(d['x']*role[:,0,0]).sum(1)/role[:,0,0].sum(1).clamp_min(1)
    traj=(join('traj')*role).sum(-2)/role.sum(-2).clamp_min(1)
    rot=torch.stack([x[:,3:6],x[:,6:],torch.cross(x[:,3:6],x[:,6:],dim=-1)],-1)
    pose=((traj-x[:,None,None]).unflatten(-1,(3,3))@rot[:,None,None]).flatten(-2)
    d.update(value=torch.cat([pose,join('rewards')[...,None]],-1),outcome=join('outcome'),
             bin=torch.searchsorted(x.new_tensor(ENDS['early']),join('hit')).long())
    return d

def response(net,d):
    o,_=net(d)
    i,j=pairs(d['a'].size(1),d['a'].device)
    y=d['value']
    target=d['bin']
    s=d['outcome'][...,0]
    e=o['mu']-y
    point=lambda v:v[...,:9].square().mean()+1.5*v[...,9].square().mean()
    base=point(e)+.55*point(e[:,i]-e[:,j])+.25*e.abs().mean()+1.6*F.mse_loss(o['score'],d['outcome'][...,1:])
    timing=-o['p'].gather(-1,target[...,None]).clamp_min(1e-8).log().mean()
    joint=coupled(o,i,j)
    neg=joint[:,:,-1,:-1].sum(-1)
    pos=joint[:,:,:-1,-1].sum(-1)
    law=torch.stack([neg,1-(neg+pos),pos],-1)
    observed=(1+s[:,i]-s[:,j]).long()
    delta=-law.gather(-1,observed[...,None]).clamp_min(1e-8).log().mean()
    p=o['timing'].gather(-1,target[:,:,None,None].expand(-1,-1,o['w'].size(1),1)).squeeze(-1)
    cell=-(o['w'].clamp_min(1e-8).log()+p.clamp_min(1e-8).log().sum(1)).logsumexp(1).mean()/s.size(1)
    dm,ds,w=difference(o,i,j,detach_mean=True)
    sd=(o['tau'].square()+o['load'].square()).sqrt()
    marginal=mixture_nll(y,o['mu'].detach().unsqueeze(3)+o['offset'],sd,w).mean()
    effect=mixture_nll(y[:,i]-y[:,j],dm,ds,w).mean()
    return base+timing+.7*cell+.8*delta+.11*marginal+.16*effect

def fit():
    torch.manual_seed(271)
    dev='cuda'
    net=attach(restore('dynamic-full',dev))
    net.requires_grad_(False)
    net.core.recourse_source.requires_grad_(True)
    net.backup_heads.requires_grad_(True)
    tr,val=[[records(dom,split,dev) for dom in DOMAINS] for split in ('world','val')]
    val=[menu(d) for d in val]
    params=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=.00085)
    steps=1400
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00006)
    best=float('inf')
    hist=[]
    for step in range(1,steps+1):
        idx=torch.randperm(len(BANK),device=dev)[:4]
        cols=torch.cat([idx,idx+len(BANK)])
        parts=[subset(take(d,torch.randint(len(d['x']),(24,),device=dev)),cols) for d in tr]
        row={k:torch.cat([q[k] for q in parts]) for k in parts[0]}
        obj=response(net,row)
        opt.zero_grad()
        obj.backward()
        torch.nn.utils.clip_grad_norm_(params,4.)
        opt.step()
        sched.step()
        if step%100==0:
            with torch.no_grad():
                values=[sum(len(ix)*response(net,take(d,ix)).item() for ix in torch.arange(len(d['x']),device=dev).split(24))/len(d['x']) for d in val]
            hist.append([step,values])
            if np.mean(values)<best:
                best=np.mean(values)
                torch.save(dict(step=step,model=net.state_dict()),OUT/'recourse.pt')
            print('world',step,float(np.mean(values)),flush=True)
    (OUT/'recourse-fit.json').write_text(json.dumps(hist))

def read_policy(dev):
    net=Decision().to(dev)
    net.load_state_dict(torch.load(OUT/'recourse-selector.pt',map_location=dev)['model'])
    return net.eval()

def train_policy():
    torch.manual_seed(271)
    dev='cuda'
    net=world(dev)
    def encode(dom,split):
        d=menu(records(dom,split,dev))
        rows=[features(net,take(d,ix)) for ix in torch.arange(len(d['x']),device=dev).split(20)]
        return {k:torch.cat([q[k] for q in rows]) for k in rows[0]}|dict(s=d['outcome'][...,0],outcome=d['outcome'])
    tr,val=[[encode(dom,split) for dom in DOMAINS] for split in ('policy','val')]
    rank=Decision().to(dev)
    x=torch.cat([d['m'].flatten(0,1) for d in tr])
    rank.center.copy_(x.mean(0))
    rank.scale.copy_(x.std(0).clamp_min(.05))
    rank.kl_scale.copy_(torch.cat([d['kl'].flatten() for d in tr]).mean().clamp_min(.005))
    opt=torch.optim.AdamW(rank.parameters(),lr=.00065)
    steps=640
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00006)
    best=float('inf')
    hist=[]
    for step in range(1,steps+1):
        ds=[take(d,torch.randint(len(d['s']),(48,),device=dev)) for d in tr]
        row={k:torch.cat([s[k] for s in ds]) for k in ds[0]}
        obj=sum(decision_loss(rank,row,m) for m in MODES)/len(MODES)
        opt.zero_grad()
        obj.backward()
        opt.step()
        sch.step()
        if step%40==0:
            with torch.no_grad():
                v=np.mean([decision_loss(rank,d,m).item() for d in val for m in MODES])
            hist.append([step,v])
            if v<best:
                best=v
                torch.save(dict(step=step,model=rank.state_dict()),OUT/'recourse-selector.pt')
    rank=read_policy(dev)
    choices={}
    for mode in ('direct',*MODES):
        evals=[]
        for d in val:
            a=(d['m'][...,0] if mode=='direct' else rank(d,mode)[0]).argmax(1)
            y=d['outcome'][torch.arange(len(a),device=dev),a]
            evals.append(y.mean(0).tolist())
        choices[mode]=evals
    selected=max(choices,key=lambda m:(sum(r[0] for r in choices[m]),sum(r[1]+r[2] for r in choices[m])))
    (OUT/'recourse-selection.json').write_text(json.dumps(dict(mode=selected,validation=choices,loss=hist),indent=1))
    print('selected',selected,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=['world','policy'])
    torch.set_num_threads(2)
    (fit if p.parse_args().job=='world' else train_policy)()
