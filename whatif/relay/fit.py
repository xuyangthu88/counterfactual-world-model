"""Matched budgets for rebuilding, holding the initial graph, and disabling paired-law loss."""
import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
from whatif.mechanism.model import labeled
from whatif.mechanism.train import avg
from whatif.hitting.fit import restore as stage25,pairs
from whatif.hitting.data import take
from .native import DOMAINS,OUT
from .data import dataset,sample
from .model import World,coupled,difference

SOURCES=[('cophy','short')]+[(d,p) for p in ('short','early','late') for d in DOMAINS]
def data(split,dev):
    return [dataset(d,p,split,dev) for d,p in SOURCES]

def mixed(ds):
    parts=[sample(d,n,4) for d,n in zip(ds,[14,8,8,15,15,15,15])]
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}

def mixture_nll(truth,mu,sd,w):
    log= torch.log(w.clamp_min(1e-9))[:,None,None,:,None]-sd.log()-.5*((truth.unsqueeze(-2)-mu)/sd).square()
    return -log.logsumexp(-2)

def objective(net,d,paired):
    o,g=net(d)
    i,j=pairs(d['a'].size(1),d['x'].device)
    mover=d['role'][:,None,None,:,0:1]
    present=d['role'].sum(-1)[:,None,None,:,None]
    obj=present+3*mover
    atom=(1-d['feedback'])
    # Point losses directly fit conditional means and supervise intermediate motion.
    err=g['trajectory']-d['traj']
    loss_pose=avg(F.smooth_l1_loss(g['trajectory'],d['traj'],beta=.25,reduction='none')+.12*err.square(),obj)
    pair_pose=avg((err[:,i]-err[:,j]).square(),mover)
    valid=labeled(d['role'])[:,None,None]*d['valid'][...,None,None]
    edge=g['edge_time']
    target=d['contact']
    ce=F.binary_cross_entropy_with_logits(edge,target,reduction='none')
    current=valid[:,:,1:]*d['valid'][:,:,:-1,None,None]
    change=(target[:,:,1:]-target[:,:,:-1]).abs()
    contact=avg(ce,valid)+.3*avg(ce[:,:,1:],current*change)
    discord=target[:,i]-target[:,j]
    rank=F.softplus(-(edge[:,i]-edge[:,j])*discord)
    contact+=.25*avg(rank,valid[:,i]*valid[:,j]*discord.abs())
    tower=d['measured'][:,None,None,None,None]*valid
    causal=avg(F.binary_cross_entropy_with_logits(g['causal_time'],(d['effects']>.1).float(),reduction='none'),tower)
    strength=avg(F.smooth_l1_loss(g['impact_time'],d['effects'].log1p(),beta=.2,reduction='none'),tower)
    factual=avg(F.binary_cross_entropy_with_logits(g['fact'],d['fe'],reduction='none'),labeled(d['role']))
    short_reward=avg((g['r']-d['r']).square(),(atom*d['reward_mask'])[:,None,None])
    (dm,_),_=net.core.contrast(g,i,j)
    atomdiff=avg((dm-(d['y'][:,i]-d['y'][:,j])).square(),atom[:,None,None,None]*mover[:,:,0])
    physical=(loss_pose+.28*pair_pose+.65*contact+.65*causal+.16*strength+.1*factual+.65*short_reward+
              .24*atomdiff+.12*net.core.reconstruction(d)+.002*g['kl']+.001*g['sparse'])
    idx=d['feedback'].bool()
    if not idx.any():
        return physical
    o={k:v[idx] for k,v in o.items()}
    target=d['bin'][idx]
    values=d['value'][idx]
    e=o['mu']-values
    mu=o['mu'].detach().unsqueeze(-2)+o['offset']
    sd=(o['tau'].square()+o['load'].square()).sqrt()
    time=-o['p'].gather(-1,target[...,None]).clamp_min(1e-9).log().mean()
    rms=lambda v:v[...,:9].square().mean()+1.5*v[...,9].square().mean()
    points=rms(e)+.55*rms(e[:,i]-e[:,j])+.25*e.abs().mean()
    nll=mixture_nll(values,mu,sd,o['w']).mean()
    forecast=time+1.6*F.mse_loss(o['score'],d['outcome'][idx,...,1:])+points+.11*nll
    if paired:
        # All branches in the same layout share exactly one response regime.
        prob=o['timing'].gather(-1,target[:,:,None,None].expand(-1,-1,o['w'].size(1),1)).squeeze(-1)
        joint=-(o['w'].clamp_min(1e-9).log()+prob.clamp_min(1e-9).log().sum(1)).logsumexp(1).mean()/target.size(1)
        dm,ds,w=difference(o,i,j,detach_mean=True)
        effect=mixture_nll(values[:,i]-values[:,j],dm,ds,w).mean()
        forecast=forecast+.7*joint+.16*effect
    return physical+forecast

def validation(net,ds):
    scores=[]
    with torch.no_grad():
        for d in ds:
            chunks=[]
            for ix in torch.arange(len(d['x']),device=d['x'].device).split(32):
                v=take(d,ix)
                chunks.append(len(ix)*objective(net,v,False).item())
            scores.append(sum(chunks)/len(d['x']))
    return scores

def train(kind,steps=2400):
    dev='cuda'
    torch.manual_seed(271)
    torch.set_num_threads(2)
    train_rows,val=data('train',dev),data('val',dev)
    net=World(kind).to(dev)
    old=stage25('joint',False,dev)
    net.core.load_state_dict(old.core.state_dict(),strict=False)
    net.scene.load_state_dict(old.scene.state_dict())
    net.lift.load_state_dict(old.lift.state_dict())
    inherited=set(old.core.state_dict())
    groups=[[],[]]
    for name,p in net.named_parameters():
        slow=name.startswith('core.') and name[5:] in inherited
        groups[int(slow)].append(p)
    opt=torch.optim.AdamW([dict(params=groups[0],lr=.00115),dict(params=groups[1],lr=.00030)],weight_decay=.008)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.000055)
    best=float('inf')
    hist=[]
    for step in range(1,steps+1):
        net.train()
        row=mixed(train_rows)
        loss=objective(net,row,kind!='unpaired')
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),4.)
        opt.step()
        sched.step()
        if step%200==0:
            net.eval()
            scores=validation(net,val)
            criterion=float(np.mean(scores))
            if criterion<best:
                best=criterion
                torch.save(dict(step=step,model=net.state_dict()),OUT/f'{kind}.pt')
            hist.append(dict(step=step,scores=scores,criterion=criterion))
            (OUT/f'{kind}-fit.json').write_text(json.dumps(hist,indent=1))
            print(kind,step,round(criterion,5),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['dynamic','fixed','unpaired'])
    train(p.parse_args().kind)
