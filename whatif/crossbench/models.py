"""A contact mechanism shared by both benchmarks, plus published state interfaces."""
import ast
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .tasks import ACTIONS, TASKS

ROOT = Path('work/stage7/upstream')
OUT = Path('results/crossbench')
ID = [0,1,2,3,4,5,6,7,8,12,14,15,16]


def wrap(x):
    return torch.atan2(x.sin(),x.cos())


def difference(x,y):
    raw=x-y
    return torch.cat((raw[...,:2],wrap(raw[...,2:3]),raw[...,3:]),dim=-1)


def panel(split,device,tasks=TASKS):
    rows=[]
    for task in tasks:
        splits = [split]
        if task=='pusht' and split in ('train','val'):
            splits.append('near'+split)
        for part in splits:
            with np.load(Path('data/crossbench')/f'{task}-{part}.npz') as z:
                d=dict(z)
                if task=='causalworld':
                    d['physics']=np.digitize(d['physical_value'],[.025,.035])
                rows.append(d)
    return {k:torch.from_numpy(np.concatenate([d[k] for d in rows])).to(device) for k in rows[0]}


def actions(s):
    return torch.as_tensor(ACTIONS,device=s.device,dtype=s.dtype)


def expand_state(s,a):
    return s.unsqueeze(-2).expand(*s.shape[:-1],a.shape[-2],s.shape[-1])


def mlp(d,o,h=96):
    return nn.Sequential(nn.Linear(d,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,o))


def geometry(s,a):
    # s,a have identical leading axes. The scene-specific quantities remain observable.
    diff=s[...,9:11]-s[...,:2]
    phi=torch.atan2(diff[...,1],diff[...,0])+a[...,0]
    f=torch.stack((phi.cos(),phi.sin()),dim=-1)
    side=torch.stack((-f[...,1],f[...,0]),dim=-1)
    rel=s[...,3:9].reshape(*s.shape[:-1],3,2)-s[...,:2].unsqueeze(-2)
    along=(rel*f.unsqueeze(-2)).sum(-1)
    across=(rel*side.unsqueeze(-2)).sum(-1)
    domain=s[...,13:14]
    role=torch.stack([torch.zeros_like(domain[...,0]),domain[...,0],domain[...,0]],dim=-1)
    ang=s[...,2]-phi
    base=torch.cat((s[...,:2],diff,ang.cos()[...,None],ang.sin()[...,None],
                    wrap(s[...,2:3]-s[...,11:12]).sin(),wrap(s[...,2:3]-s[...,11:12]).cos(),
                    s[...,12:17],f, a),dim=-1)
    sites=torch.stack((along,across,role),dim=-1)
    return base,sites,f,side


def geometry_cover(xy, yaw, s):
    """Area of the known goal covered by a predicted pose, in normalized units."""
    goal = s[..., 9:11]
    size = .5 * (yaw.cos().abs() + yaw.sin().abs())
    relative = xy - goal
    hi = (relative + size[..., None]).clamp(max=.5)
    lo = (relative - size[..., None]).clamp(min=-.5)
    square = (hi - lo).clamp_min(0.).prod(-1)

    # Equal-area quadrature points from the two rectangles used by native PushT.
    # x, y represent the body origin, as in Pymunk's local_to_world transform.
    spacing = .075
    points = [(x0+(i+.5)*spacing, y0+(j+.5)*spacing)
              for x0, y0, nx, ny in ((-.6, 0., 16, 4), (-.15, .3, 4, 12))
              for i in range(nx) for j in range(ny)]
    grid = xy.new_tensor(points)
    goal_yaw = s[..., 11:12]
    gx = grid[:, 0] * goal_yaw.cos() - grid[:, 1] * goal_yaw.sin() - relative[..., 0:1]
    gy = grid[:, 0] * goal_yaw.sin() + grid[:, 1] * goal_yaw.cos() - relative[..., 1:2]
    c, sn = yaw[..., None].cos(), yaw[..., None].sin()
    x, y = (c*gx + sn*gy).abs(), -sn*gx + c*gy
    tee = (((x<=.6) & (y>=0) & (y<=.3)) |
           ((x<=.15) & (y>=.3) & (y<=1.2))).float().mean(-1)
    return torch.where(s[..., 13]>.5, tee, square)


def utility(cov, xy, yaw, s):
    dist=(xy-s[...,9:11]).norm(dim=-1)
    turn=(1-(yaw-s[...,11]).cos())*s[...,13]
    threshold=.90+.05*s[...,13]
    return cov+.28*torch.sigmoid((cov-threshold)/.035)-.09*dist-.05*turn


class ContactModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge=mlp(23,5)
        self.drift=mlp(20,3)
        self.cover=mlp(25,1)
    def forward(self,s,a,k,u,remove=None):
        base,sites,f,side=geometry(s,a)
        common=torch.cat((base,k[...,None],u[...,None]),dim=-1)
        per_site=torch.cat((common.unsqueeze(-2).expand(*sites.shape[:-1],common.shape[-1]),sites),dim=-1)
        edge=self.edge(per_site)
        logit=edge[...,0]
        weight=logit.sigmoid()
        if remove is not None:
            weight=weight*remove
        passed=(edge[...,1:]*weight[...,None]).sum(-2)
        local=self.drift(common)*.15+passed[...,:3]
        xy=local[...,0:1]*f+local[...,1:2]*side
        delta=torch.cat((xy,local[...,2:3]),dim=-1)
        closeness=(s[...,:2]+xy-s[...,9:11]).norm(dim=-1,keepdim=True)
        cover=self.cover(torch.cat((common,passed,closeness),dim=-1)).squeeze(-1).sigmoid()
        return delta,cover,logit


def model_batch(model,s,a,k,u,remove=None):
    z=expand_state(s,a)
    return model(z,a,k[:,None].expand_as(a[...,0]),u[:,None].expand_as(a[...,0]),remove)


def graph_loss(model,d,idx,t,supervised=True):
    s=d['state'][idx,t]
    y=d['end'][idx,t]
    a=actions(s).expand(len(s),-1,-1)
    k=d['physics'][idx].float()-1
    u=d['execution'][idx,t]/.04
    pred,cover,logits=model_batch(model,s,a,k,u)
    target=difference(y[...,:3],s[:,None,:3])
    pose=F.smooth_l1_loss(difference(pred,target),torch.zeros_like(target),beta=.09)
    z=s[:,None].expand_as(y)
    up=utility(cover,z[...,:2]+pred[...,:2],z[...,2]+pred[...,2],z)
    ut=utility(y[...,12],y[...,:2],y[...,2],z)
    gap=up-ut
    rank=F.smooth_l1_loss(gap[:,:,None]-gap[:,None,:],torch.zeros_like(gap[:,:,None]-gap[:,None,:]),beta=.08)
    # Same factual episode, same latent physics and disturbance on both action branches.
    row=torch.arange(len(idx),device=s.device)
    ref=d['ref'][idx]
    observed_effect=difference(y[...,:3],y[row,ref,:3][:,None])
    effect=difference(pred,pred[row,ref][:,None])
    effect_error=difference(effect,observed_effect)
    paired_pose=F.smooth_l1_loss(effect_error,torch.zeros_like(effect_error),beta=.09)
    loss=pose+.55*paired_pose+.45*F.mse_loss(cover,y[...,12])+.28*rank
    if d['null_pose'].numel():
        src=d['null_source'][idx,t]
        keep=1-F.one_hot(src,3).float()
        masked=model_batch(model,s,a,k,u,keep)[0]
        nt=difference(d['null_pose'][idx,t],s[:,None,:3])
        loss=loss+.4*F.smooth_l1_loss(difference(masked,nt),torch.zeros_like(nt),beta=.09)
        if not supervised:
            impact=difference(target,nt).norm(dim=-1)>.025
            return loss+.18*F.binary_cross_entropy_with_logits(logits.gather(-1,src[...,None]).squeeze(-1),impact.float())
    if supervised:
        loss=loss+.22*F.binary_cross_entropy_with_logits(logits,d['edge'][idx,t].float())
    return loss


def particles(model,s):
    a=actions(s)
    pair=torch.cartesian_prod(s.new_tensor([-1.,0.,1.]),s.new_tensor([-1.,0.,1.]))
    shape=(len(s),len(a),len(pair))
    z=s[:,None,None].expand(*shape,-1)
    cmd=a[None,:,None].expand(*shape,-1)
    return model(z,cmd,pair[:,0].expand(shape),pair[:,1].expand(shape))


def belief(delta0,ref,observed,sigma):
    i=torch.arange(len(ref),device=ref.device)
    error=difference(delta0[i,ref],observed[:,None])
    return (-.5*(error/sigma).square().sum(-1)).softmax(-1)


def fresh(w):
    return w.reshape(-1,3,3).sum(-1).repeat_interleave(3,dim=-1)/3


def cophy_class():
    # Execute the published module's state path. Remove only its unused image
    # encoder constructor and imports; the original GCN, GRUs and gates run verbatim.
    path=ROOT/'cophy/cf_learning/model.py'
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if getattr(n,'name',None) in ('aggreg_E','CoPhyNet')]
    init=next(n for n in nodes[1].body if getattr(n,'name',None)=='__init__')
    init.body=[n for n in init.body if not (isinstance(n,ast.Assign) and
              any(getattr(t,'attr','')=='derendering' for t in n.targets))]
    space={'torch':torch,'nn':nn}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),space)
    return space['CoPhyNet']


def tokens(s,a):
    base,sites,f,side=geometry(s,a)
    # One moving target and visible command/context tokens in the published 3-channel interface.
    src=s[...,3:9].reshape(*s.shape[:-1],3,2)
    obj=torch.cat((s[...,:2],wrap(s[...,2:3])/np.pi),dim=-1)
    src=torch.cat((src,torch.zeros_like(src[...,:1])),dim=-1)
    goal=s[...,9:12].clone()
    goal[...,2]=wrap(goal[...,2])/np.pi
    command=torch.cat((s[...,:2]+f*a[...,2:3]+side*a[...,1:2],a[...,2:3]),dim=-1)
    meta=torch.stack((s[...,13],a[...,0],a[...,1]),dim=-1)
    return torch.cat((obj.unsqueeze(-2),src,goal.unsqueeze(-2),command.unsqueeze(-2),meta.unsqueeze(-2)),dim=-2)


def cophy_predict(model,s0,obs,ref,s):
    a=actions(s)
    refa=a[ref]
    ab=torch.stack((tokens(s0,refa),tokens(obs,refa)),dim=1)
    present=torch.ones_like(ab[:,0,:,0])
    ctx=model.rnn_on_AB_up(model.gcn_on_AB(ab,present))
    x=tokens(expand_state(s,a),a.expand(len(s),-1,-1))
    n,t,k,_=x.shape
    context=ctx[:,None].expand(-1,t,-1,-1).reshape(n*t,k,-1)
    out,stable=model.pred_D(context,x.reshape(n*t,k,3),present[:,None].expand(-1,t,-1).reshape(n*t,k),T=1)
    raw=out[:,0,0].reshape(n,t,3)
    delta=difference(torch.cat((raw[...,:2],raw[...,2:]*np.pi),dim=-1),s[:,None,:3])
    return delta,stable[:,0,0,0].reshape(n,t)


class NullPoint(nn.Module):
    """State adapter for NCII eqs. 2-4: masked PointNet likelihood and interaction classifier."""
    def __init__(self):
        super().__init__()
        self.encoder=mlp(10,64)
        self.forward_head=mlp(64+12,6)
        self.inference=mlp(18+3,1)
    def forward(self,s,a,valid):
        base,sites,_,_=geometry(s,a)
        # The relational input (including commanded motion) is erased for each removed source.
        event=torch.cat((sites,base[..., -5:].unsqueeze(-2).expand(*sites.shape[:-1],5),
                         base[...,12:14].unsqueeze(-2).expand(*sites.shape[:-1],2)),dim=-1)
        pooled=(self.encoder(event)*valid[...,None]).amax(dim=-2)
        own=torch.cat((s[...,:3].clone(),s[...,9:17],torch.cos(s[...,2:3])),dim=-1)
        out=self.forward_head(torch.cat((own,pooled),dim=-1))
        return out[...,:3],out[...,3:].clamp(-4,1.5)
    def edges(self,s,a):
        base,sites,_,_=geometry(s,a)
        return self.inference(torch.cat((base.unsqueeze(-2).expand(*sites.shape[:-1],18),sites),dim=-1)).squeeze(-1)


def nll(out,y):
    m,logstd=out
    return (.5*(difference(y,m)*(-logstd).exp()).square()+logstd).sum(-1)
