"""Joint success outcomes and conditional score changes from local graph rollouts."""
import json
from itertools import product
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import mlp,pool,tile
from whatif.mechanism.proposals import WEIGHTS,MODES
from whatif.mechanism.train import SEED,sub
from .data import BUDGET,OUT
from .fit import current,world

KL_WEIGHT=.5
POLICIES=['marginal','pair_mean','pair_law','pair_kl']


class ModalAction(nn.Module):
    """Batched execution of the fixed proposal's first mode on imagined observations."""
    def __init__(self):
        super().__init__()
        with np.load(WEIGHTS) as w:
            for k in w.files:
                self.register_buffer(k,torch.tensor(w[k]))

    def forward(self,state,role,goal):
        target=pool(state,role[...,0])
        ex=F.normalize(target[:,3:6],dim=-1)
        ey=target[:,6:9]
        ey=F.normalize(ey-(ex*ey).sum(-1,keepdim=True)*ex,dim=-1)
        rot=torch.stack([ex,ey,torch.cross(ex,ey,dim=-1)],-1)
        own=(state[...,:3]-target[:,None,:3])@rot
        dest=((goal[:,:3]-target[:,:3])[:,None]@rot)[:,0]
        orientation=(goal[:,3:].reshape(-1,2,3)@rot).flatten(1)
        cond=torch.cat([dest,orientation,target[:,:3],ex,ey,role[...,1].sum(-1,keepdim=True)/3],-1)
        z=(torch.cat([own,cond[:,None].expand(-1,state.size(1),-1)],-1)-self.center)/self.spread
        for j in range(4):
            z=F.linear(z,getattr(self,'w'+str(j)),getattr(self,'b'+str(j)))
            if j<3:
                z=F.silu(z)
        z=z.unflatten(-1,(MODES,7))
        logits=z[...,0].masked_fill(~role[...,1,None].bool(),-torch.inf)
        chosen=logits.flatten(1).argmax(-1)
        node=chosen//MODES
        idx=torch.arange(len(z),device=z.device)
        shift=z[idx,node,chosen%MODES,1:].reshape(-1,2,3)@rot.transpose(1,2)
        cmd=torch.zeros((*state.shape[:2],7),device=z.device)
        cmd[idx,node]=torch.cat([torch.ones_like(shift[:,0,:1]),shift.flatten(1)],-1)
        return cmd


def trajectory_features(net,actor,d):
    b,k=d['a'].shape[:2]
    role=d['role']
    now=pool(d['x'],role[...,0])
    goal=d['goal']
    context=torch.cat([now-goal,goal,pool(d['f1']-d['f0'],role[...,0]),
                       d['remaining'][:,None]/BUDGET],-1)
    parts=[tile(context,k)]
    # Retain the actual factual evidence for all imagined transitions.
    fixed={key:tile(d[key],k) for key in ('role','goal','f0','fa','f1','same')}
    o=net(d)
    for t in range(3):
        if t:
            cmd=actor(state,fixed['role'],fixed['goal'])
            if 'plans' in d:
                rows=torch.arange(b,device=state.device)[:,None]
                choices_idx=torch.arange(1,k,device=state.device)[None,:]
                node=d['node'].long()[:,None]
                pos=state.reshape(b,k,*state.shape[1:])[rows,choices_idx,node,:3]
                relative=d['plans'][:,:,2*t:2*t+2]-pos[:,:,None]
                cmd=cmd.reshape(b,k,*cmd.shape[1:])
                cmd[:,1:]=0
                cmd[rows,choices_idx,node]=torch.cat([torch.ones_like(pos[...,:1]),relative.flatten(-2)],-1)
                cmd=cmd.flatten(0,1)
            o=net(dict(fixed,x=state,a=cmd[:,None]))
        state=o['y'].reshape(b*k,*d['x'].shape[1:])
        h=o['h'].reshape(b*k,state.size(1),-1)
        fixed_node=fixed['role'][...,2:3].bool()
        state=torch.where(fixed_node,tile(d['x'],k),state)
        moved=pool(state,fixed['role'][...,0])
        parts.extend([moved-fixed['goal'],pool(h,fixed['role'][...,0]),o['r'].reshape(b*k,3)])
    return torch.cat(parts,-1).reshape(b,k,-1)


class SuccessLaw(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.register_buffer('center',torch.zeros(dim))
        self.register_buffer('spread',torch.ones(dim))
        self.enc=mlp(dim,80,112)
        self.marginal=mlp(80,3,80)
        self.pair=mlp(240,11,112)

    def forward(self,x,i,j):
        z=self.enc((x-self.center)/self.spread)
        l,r=z[:,i],z[:,j]
        def branch(a,b):
            return self.pair(torch.cat([a,b,a-b],-1))
        fwd,rev=branch(l,r),branch(r,l)
        discord=(fwd[...,0]+rev[...,0])*.5
        rescue=(fwd[...,1]-rev[...,1])*.5
        both=(fwd[...,2]+rev[...,2])*.5
        # Label order 00, 01, 10, 11. Reverse swaps the middle two components.
        flip=[0,2,1,3]
        mu=.75*((fwd[...,3:7]-rev[...,3:7][...,flip])*.5).tanh()
        scale=.012+.16*F.softplus((fwd[...,7:]+rev[...,7:][...,flip])*.5)
        p,q,s=discord.sigmoid(),rescue.sigmoid(),both.sigmoid()
        mass=torch.stack([(1-p)*(1-s),p*(1-q),p*q,(1-p)*s],-1)
        marginal=self.marginal(z)
        return dict(marginal=marginal,discord=discord,rescue=rescue,both=both,
                    mass=mass,mu=mu,scale=scale)


def targets(outcome):
    success,peak,last=outcome.unbind(-1)
    return success,.5*peak+.25*last


def objective(net,x,y):
    i,j=torch.triu_indices(x.size(1),x.size(1),1,device=x.device)
    o=net(x,i,j)
    s,aux=targets(y)
    different=(s[:,i]!=s[:,j]).float()
    bce=F.binary_cross_entropy_with_logits
    def masked(v,mask):
        return (v*mask).sum()/mask.sum().clamp_min(1)
    event=(2*s[:,i]+s[:,j]).long()[...,None]
    mu=o['mu'].gather(-1,event).squeeze(-1)
    width=o['scale'].gather(-1,event).squeeze(-1)
    diff=aux[:,i]-aux[:,j]
    return (bce(o['marginal'][...,0],s)+F.mse_loss(o['marginal'][...,1:].sigmoid(),y[...,1:])
            +bce(o['discord'],different)
            +masked(bce(o['rescue'],s[:,i],reduction='none'),different)
            +masked(bce(o['both'],s[:,i],reduction='none'),1-different)
            +.22*((diff-mu).abs()/width+width.log()).mean())


def statistics(o):
    p=o['mass']
    q=o['rescue'].sigmoid().clamp(1e-7,1-1e-7)
    signed_kl=(p[...,1]+p[...,2])*torch.sign(q-.5)*(np.log(2.)+q*q.log()+(1-q)*(1-q).log())
    score=o['marginal'].sigmoid()
    return dict(marginal=score[...,0]+.5*score[...,1]+.25*score[...,2],
                mean=p[...,2]-p[...,1]+(p*o['mu']).sum(-1),loss=p[...,1],kl=signed_kl)


def choices(o,prior,policy,config):
    v=statistics(o)
    if policy=='marginal':
        score=v['marginal']-v['marginal'][:,:1]
    else:
        score=v['mean']
        if policy=='pair_kl':
            score=score+KL_WEIGHT*v['kl']
        if policy in ('pair_law','pair_kl'):
            score=score.masked_fill(v['loss']>config['risk'], -torch.inf)
    score=score+config['prior']*prior.clamp(min=-6.)-config['cost']
    score[:,0]=0.
    return score.argmax(-1)


@torch.no_grad()
def panel(net,actor,split,device,segments=False):
    xx,yy,pr=[],[],[]
    for dom in ('causalworld','pusht'):
        if segments:
            from .data import DATA
            with np.load(DATA/f'segment-{dom}-{split}.npz') as f:
                d={k:torch.as_tensor(f[k],device=device).float() for k in f.files}
            d['prior']=torch.zeros_like(d['a'][:,:,0,0])
        else:
            d=current(dom,split,device)
        for ix in torch.arange(len(d['x']),device=device).split(40):
            xx.append(trajectory_features(net,actor,sub(d,ix)))
        yy.append(d['outcome'])
        pr.append(d['prior'])
    return torch.cat(xx),torch.cat(yy),torch.cat(pr)


def decision_head(segments=False):
    ckpt=torch.load(OUT/('segment.pt' if segments else 'decision.pt'),map_location='cpu')
    head=SuccessLaw(len(ckpt['center']))
    head.load_state_dict(ckpt)
    return head.eval()


def fit(device,segments=False):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    net,actor=world().to(device),ModalAction().to(device)
    x,y,_=panel(net,actor,'train',device,segments)
    vx,vy,vp=panel(net,actor,'val',device,segments)
    head=SuccessLaw(x.size(-1)).to(device)
    head.center.copy_(x.mean((0,1)))
    head.spread.copy_(x.flatten(0,1).std(0).clamp_min(.035))
    opt=torch.optim.AdamW(head.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,75,eta_min=.0001)
    history=[]
    best=float('inf')
    for epoch in range(75):
        for ids in torch.randperm(len(x),device=device).split(96):
            loss=objective(head,x[ids],y[ids])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),4.)
            opt.step()
        sched.step()
        if epoch%3==2:
            with torch.no_grad():
                error=objective(head,vx,vy).item()
            history.append([epoch+1,error])
            if error<best:
                best=error
                torch.save({k:v.cpu() for k,v in head.state_dict().items()},OUT/('segment.pt' if segments else 'decision.pt'))
    head=decision_head(segments).to(device)
    with torch.no_grad():
        i=torch.arange(x.size(1),device=device)
        o=head(vx,i,torch.zeros_like(i))
        success,aux=targets(vy)
        g=success+aux
        options=[]
        grid=product([0.],[0.,.025,.07,.14],[.1,.2,.4]) if segments else product([.015,.05,.075],[.0,.025,.055],[.1,.2,.35])
        for prior,cost,risk in grid:
            conf=dict(prior=prior,cost=cost,risk=risk)
            if segments:
                from .option import choose
                picked=choose(o,vp,'pair_law',conf,batched=True)
                criterion=success+.2*aux
            else:
                picked=choices(o,vp,'pair_law',conf)
                criterion=g
            gain=(criterion.gather(1,picked[:,None])[:,0]-criterion[:,0]).mean().item()
            options.append(dict(**conf,validation_gain=gain))
        # Fixed, shared search. Break equal gains in favor of stricter trust in the reference.
        chosen=max(options,key=lambda c:(c['validation_gain'],c['cost'],c['prior'],-c['risk']))
    (OUT/('segment-selection.json' if segments else 'selection.json')).write_text(json.dumps(dict(history=history,config=chosen,grid=options),indent=2))
    print('selected',chosen,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu')
    p.add_argument('--segments',action='store_true')
    args=p.parse_args()
    fit(args.device,args.segments)
