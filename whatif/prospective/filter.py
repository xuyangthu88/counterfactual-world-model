"""Filtering physics from a completed stroke, then transporting its response to the next contact."""
import numpy as np
import torch
from whatif.shared.model import body_delta,pose,grid,task_value,wrap
from .collect import DATA,OUT


def panel(task,sp,device):
    with np.load(DATA/f'{task}-{sp}.npz') as z:
        d={k:torch.as_tensor(z[k],device=device) for k in z.files}
    return {k:(v.float() if v.is_floating_point() else v) for k,v in d.items()}


def blank(s):
    return s.new_full((len(s),5),.2),s.new_zeros(len(s),5),s.new_zeros(len(s),5)


def response(s,a):
    # In scaled SE(2), the third component is a characteristic length times yaw.
    lever=a[...,5]*.47/(.5-.2791*s[...,13])
    return torch.cat((a[...,2:4],lever[...,None]),-1)*a[...,4:5]


def evidence(s,obs,a,z,edge,measure):
    n=len(s)
    center=body_delta(s[:,None],z).reshape(n,5,3,3).mean(2)
    actual=body_delta(s,obs)
    err=actual[:,None]-center
    err=torch.cat((err[...,:2],.47*wrap(err[...,2:3])),-1)
    gate=edge.sigmoid().amax(-1).reshape(n,5,3).mean(-1)
    h=response(s,a)[:,None]*gate[...,None]
    return err,h,measure


def update(prev,e,cfg):
    old,xx,xy=prev;err,h,measure=e
    rho=err.new_tensor([0.,.5*.08**.1,.08**.1,(.08**.1+.55**.1)/2,.55**.1])
    sigma=cfg['sigma'];strength=cfg['strength']
    # A heavy-tailed likelihood limits the influence of an imperfect contact prediction.
    motion=-3.5*torch.log1p(err.square().sum(-1)/(4*sigma*sigma))
    new_free=measure[:,0:1]/(measure[:,0:1]+3)
    decay=-.5*((measure[:,1:2]-rho)/.09).square().clamp(max=16)*new_free
    mass=(.80*old.clamp_min(1e-8).log()+motion+decay).softmax(-1)
    mass=torch.lerp(torch.full_like(mass,.2),mass,strength)
    xx=.75*xx+h.square().sum(-1)
    xy=.75*xy+(h*err).sum(-1)
    return mass,xx,xy


def gain(belief,cfg):
    return cfg['gain']*(belief[2]/(.12+belief[1])).clamp(-.8,.8)


def correct(s,a,z,edges,g):
    body=body_delta(s,z)
    h=response(s,a)*edges.sigmoid().amax(-1,keepdim=True)
    corr=g[...,None]*h
    new=body+torch.cat((corr[...,:2],corr[...,2:]/.47),-1)
    state=z.clone()
    state[...,:3]=pose(s,new)
    return state


def predict_grid(graph,s,a,belief,cfg):
    z,e,p,r=grid(graph,s,a)
    corrected=correct(s[:,None,None],a[:,:,None],z,e,gain(belief,cfg).repeat_interleave(3,-1)[:,None,:])
    weights=belief[0].repeat_interleave(3,-1)/3
    return corrected,e,p,r,weights


def fit_data(model,task,split):
    d=panel(task,split,model.device)
    i=torch.arange(len(d['b']),device=model.device);b=d['b'];s0=d['previous'];s=d['state'];a=d['cmd']
    prev,e,_,_=grid(model.graph,s0,d['previous_actions'][i,b][:,None])
    ev=evidence(s0,s,d['previous_actions'][i,b],prev[:,0],e[:,0],d['measurement'])
    z,edge,peak,rank=grid(model.graph,s,a)
    utility=task_value(s[:,None],d['end'],d['peak'])
    true=body_delta(s[:,None],d['end'])
    return d,ev,z,edge,peak,rank,utility,true


def calibration_score(batch,cfg):
    d,ev,z,edge,peak,rank,ut,true=batch
    s=d['state'];a=d['cmd'];i=torch.arange(len(s),device=s.device)
    bel=update(blank(s),ev,cfg)
    g=gain(bel,cfg).repeat_interleave(3,-1)
    zz=correct(s[:,None,None],a[:,:,None],z,edge,g[:,None])
    w=bel[0].repeat_interleave(3,-1)/3
    ur=.55*task_value(s[:,None,None],zz,peak)+.45*rank
    choice=(ur*w[:,None]).sum(-1).argmax(1)
    regret=(ut.amax(1)-ut[i,choice]).mean()
    pred=body_delta(s[:,None,None],zz)
    diff=(pred*w[:,None,:,None]).sum(2)-true
    diff=diff-diff.mean(1,keepdim=True)
    error=torch.cat((diff[...,:2],.47*wrap(diff[...,2:3])),-1).norm(dim=-1).mean()
    return [float(regret),float(error)]
