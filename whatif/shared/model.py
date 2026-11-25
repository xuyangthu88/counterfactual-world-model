"""Contact-conditioned transitions in a shared object frame."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from whatif.crossbench.models import mlp,wrap,cophy_class
from .actions import DATA,OUT,menu

VARY=[0,1,2,3,4,5,6,7,8,12,14,15,16,17,18]


def old_cw():
    with np.load('data/crossbench/causalworld-train.npz') as z:
        def extend(s):
            return np.concatenate([s,np.full_like(s[...,:1],.45),np.zeros_like(s[...,:1])],-1)
        n=len(z['seed'])
        states=extend(z['state']).reshape(-1,19)
        ends=extend(z['end']).reshape(-1,12,19)
        return dict(state=states,end=ends,cmd=np.array([menu(s) for s in states]),peak=ends[...,12],
                    edge=z['edge'].reshape(-1,12,3),ref=np.repeat(z['ref'],2),noise=z['execution'].reshape(-1),
                    physics=np.repeat(z['physics'],2),nullpose=z['null_pose'].reshape(-1,12,3),
                    source=z['null_source'].reshape(-1,12),seed=np.repeat(z['seed'],2),
                    steps=np.full((n*2,12),42),protocol=np.repeat(z['protocol'],2))


def panel(split,device,tasks=('causalworld','pusht')):
    rows=[]
    for task in tasks:
        with np.load(DATA/f'{task}-{split}.npz') as z:rows.append(dict(z))
        if task=='causalworld' and split=='train':rows.append(old_cw())
    d={k:torch.as_tensor(np.concatenate([v[k] for v in rows]),device=device) for k in rows[0]}
    return {k:(v.float() if v.is_floating_point() else v) for k,v in d.items()}


def rotate(x,angle):
    c,s=angle.cos(),angle.sin()
    return torch.stack((c*x[...,0]-s*x[...,1],s*x[...,0]+c*x[...,1]),-1)


def center(z,domain):
    offset=torch.stack((torch.zeros_like(domain),.45*domain),-1)
    return z[...,:2]+rotate(offset,z[...,2])


def body_delta(s,z):
    dom=s[...,13]
    return torch.cat((rotate(center(z,dom)-center(s,dom),-s[...,2]),wrap(z[...,2]-s[...,2])[...,None]),-1)


def pose(s,d):
    angle=s[...,2]+d[...,2]
    offset=torch.stack((torch.zeros_like(angle),.45*s[...,13].expand_as(angle)),-1)
    xy=center(s,s[...,13])+rotate(d[...,:2],s[...,2])-rotate(offset,angle)
    return torch.cat((xy,angle[...,None]),-1)


def features(s,a):
    dom=s[...,13]
    goal=s[...,9:12]
    d=body_delta(s,goal)
    v=rotate(s[...,14:16],-s[...,2])
    sites=rotate(s[...,3:9].reshape(*s.shape[:-1],3,2)-center(s,dom).unsqueeze(-2),-s[...,2:3])
    base=torch.cat((s[...,:2],s[...,2:3].cos(),s[...,2:3].sin(),d[...,:2],
                    d[...,2:3].cos(),d[...,2:3].sin(),s[...,12:14],v,s[...,16:19],a),-1)
    x=sites-a[...,:2].unsqueeze(-2)
    roles=dom[...,None]*s.new_tensor([0.,1.,1.])
    per=torch.cat((sites,x,x.norm(dim=-1,keepdim=True),roles[...,None]),-1)
    return base,per


def physical(s,a,k):
    domain=s[...,13]
    sign=2*domain-1
    gain=.88+.34*sign*k
    move=a[...,2:4]*a[...,4:5]*gain[...,None]
    lever=a[...,5]*a[...,4]/(.5-.2791*domain)
    drift=rotate(s[...,14:16],-s[...,2])*(.7+1.7*domain*(k+1))[...,None]
    return torch.cat((move+drift,(lever*gain+s[...,16]*(.7+1.7*domain*(k+1)))[...,None]),-1)


def task_value(s,z,peak):
    dom=s[...,13]
    error=center(z,dom)-center(s[...,9:12],dom)
    angle=wrap(z[...,2]-s[...,11])
    dist=(error.square().sum(-1)+(.47*dom*angle).square()+1e-6).sqrt()
    # Dense progress outside the goal and native overlap inside it.
    return .65*z[...,12]+.28*peak-.38*dist+.22*torch.sigmoid((peak-(.90+.05*dom))/.025)


class SharedGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.messages=mlp(22+2+6,13,128)
        self.read=mlp(22+2+12,17,128)
    def forward(self,s,a,k,u,keep=None):
        base,sites=features(s,a)
        shared=torch.cat((base,k[...,None],u[...,None]),-1)
        edges=self.messages(torch.cat((shared.unsqueeze(-2).expand(*sites.shape[:-1],24),sites),-1))
        logits=edges[...,0];gate=logits.sigmoid()
        if keep is not None:gate=gate*keep
        h=(edges[...,1:]*gate[...,None]).sum(-2)
        out=self.read(torch.cat((shared,h),-1))
        body=out[...,:3]+physical(s,a,k)*gate.amax(-1,keepdim=True)
        nextstate=s.clone()
        nextstate[...,:3]=pose(s,body)
        nextstate[...,3:9]=s[...,3:9]+out[...,3:9]
        nextstate[...,12]=out[...,9].sigmoid()
        nextstate[...,14:17]=s[...,14:17]+out[...,10:13]
        nextstate[...,17:19]=out[...,13:15].sigmoid()
        return nextstate,logits,out[...,15].sigmoid(),out[...,16]


def grid(model,s,a):
    pair=torch.cartesian_prod(s.new_tensor([-1.,-.5,0.,.5,1.]),s.new_tensor([-1.,0.,1.]))
    shape=(*a.shape[:-1],len(pair))
    z=s.unsqueeze(-2).unsqueeze(-2).expand(*shape,19)
    cmd=a.unsqueeze(-2).expand(*shape,7)
    return model(z,cmd,pair[:,0].expand(shape),pair[:,1].expand(shape))


def log_measured(s):
    anchors=s.new_tensor([0.,.5*.08**.1,.08**.1,(.08**.1+.55**.1)/2,.55**.1])
    return -.5*((s[:,17:18]-anchors)/.065).square()*s[:,18:19]*s[:,13:14]


def posterior(s,observed,pred,ref,sigma,history=None,passive=True):
    i=torch.arange(len(s),device=s.device)
    err=pred[i,ref,:,:3]-observed[:,None,:3]
    err=torch.cat((err[...,:2],wrap(err[...,2:3])),-1)
    loglik=-.5*(err/s.new_tensor([sigma,sigma,1.6*sigma])).square().sum(-1)
    prior=torch.zeros((len(s),5),device=s.device) if history is None else .7*history.clamp_min(1e-9).log()
    if passive:prior=prior+log_measured(observed)
    return (prior.repeat_interleave(3,-1)+loglik).softmax(-1)


def represent(s,a):
    base,src=features(s,a)
    return torch.cat((base,src.flatten(-2)),-1)


def pets_targets(s,z,peak):
    diff=z[...,VARY]-s[...,VARY]
    return torch.cat((diff[...,:2],wrap(diff[...,2:3]),diff[...,3:],peak[...,None]),-1)


def unpack(s,dy):
    z=s.clone()
    z[...,VARY]=s[...,VARY]+dy[...,:-1]
    z[...,12]=z[...,12].clamp(0,1)
    z[...,17:19]=z[...,17:19].clamp(0,1)
    return z,dy[...,-1].clamp(0,1)


class CoPhy(nn.Module):
    def __init__(self):
        super().__init__();self.core=cophy_class()(10)
    def tokens(self,s,a):
        tip=s[...,3:9].reshape(*s.shape[:-1],3,2)
        src=torch.cat((tip,torch.zeros_like(tip[...,:1])),-1)
        point=center(s,s[...,13])+rotate(a[...,:2],s[...,2])
        direction=rotate(a[...,2:4],s[...,2])
        obj=torch.cat((s[...,:2],wrap(s[...,2:3])/np.pi),-1)
        goal=torch.cat((s[...,9:11],wrap(s[...,11:12])/np.pi),-1)
        groups=[obj[...,None,:],src,goal[...,None,:],torch.cat((point,a[...,4:5]),-1)[...,None,:],
                torch.cat((direction,a[...,5:6]),-1)[...,None,:],s[...,14:17][...,None,:],
                torch.cat((s[...,17:19],s[...,13:14]),-1)[...,None,:],
                torch.stack((s[...,12],a[...,6],s[...,13]),-1)[...,None,:]]
        return torch.cat(groups,-2)
    def forward(self,previous,obs,refcmd,s,a):
        ab=torch.stack((self.tokens(previous,refcmd),self.tokens(obs,refcmd)),1)
        present=torch.ones_like(ab[:,0,:,0])
        ctx=self.core.rnn_on_AB_up(self.core.gcn_on_AB(ab,present))
        x=self.tokens(s[:,None].expand(-1,a.shape[1],-1),a)
        n,t,k,_=x.shape
        out,stable=self.core.pred_D(ctx[:,None].expand(-1,t,-1,-1).reshape(n*t,k,-1),x.reshape(n*t,k,3),
                                   present[:,None].expand(-1,t,-1).reshape(n*t,k),T=1)
        q=out[:,0,0].reshape(n,t,3)
        return torch.cat((q[...,:2],q[...,2:3]*np.pi),-1),stable[:,0,0,0].reshape(n,t)


class Null(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=mlp(6+7,64);self.read=mlp(64+15,6);self.classifier=mlp(22+6,1)
    def forward(self,s,a,keep):
        base,sites=features(s,a)
        enc=self.encoder(torch.cat((sites,a.unsqueeze(-2).expand(*sites.shape[:-1],7)),-1))
        pooled=(enc*keep[...,None]).amax(-2)
        raw=self.read(torch.cat((pooled,base[...,:15]),-1))
        return raw[...,:3],raw[...,3:].clamp(-4,1.5)
    def edges(self,s,a):
        base,sites=features(s,a)
        return self.classifier(torch.cat((base.unsqueeze(-2).expand(*sites.shape[:-1],22),sites),-1)).squeeze(-1)
