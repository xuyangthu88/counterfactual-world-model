"""Observable midpoint uncertainty, graph-encoded interventions, and counterfactual recourse."""
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import mlp,pool
from .core import Core
from .native import BANK
from .load import observation

MODES=8
ATOMS=17
DIM=56
BETA=.8
INFO_WEIGHT=.2

class World(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=Core()
        self.register_buffer('bank',torch.tensor(BANK).float())
        self.register_buffer('grid',torch.linspace(-.75,.75,ATOMS))
        self.register_buffer('center',torch.zeros(27))
        self.register_buffer('spread',torch.ones(27))
        self.register_buffer('prototypes',torch.zeros(MODES,27))
        self.lift=mlp(33+30,DIM,96)
        self.midpoint=mlp(DIM,MODES)
        self.mid_motion=mlp(DIM,9)
        self.mid_reward=mlp(DIM,1)
        self.mode=nn.Embedding(MODES,16)
        self.condition=mlp(3*DIM+16,DIM,96)
        self.after=mlp(DIM,DIM,80)
        self.result=mlp(DIM,4,72)
        self.motion=mlp(DIM,9,72)
        self.pair=mlp(3*DIM,3*ATOMS,96)
        self.residual=mlp(2*DIM,8,64)

    def tag(self,d):
        feat=(observation(d)-self.center)/self.spread
        return torch.cdist(feat,self.prototypes).argmin(-1)

    def encode(self,d,duration):
        o=self.core.continuation(d,self.bank,duration)
        role=d['role']
        x=pool(d['x'],role[...,0])
        last_move=pool(d['f1']-d['f0'],role[...,0])
        rot=torch.stack([x[:,3:6],x[:,6:],torch.cross(x[:,3:6],x[:,6:],dim=-1)],-1)
        gap=((d['goal']-x).reshape(-1,3,3)@rot).flatten(1)
        own=pool(d['x'][...,:3],role[...,1])-x[:,:3]
        ctx=torch.cat([x,gap,last_move,own],-1)
        z=self.lift(torch.cat([o['hr'],ctx[:,None].expand(-1,o['hr'].size(1),-1)],-1))
        return z,o

    def imagine(self,first,second,mode):
        shape=torch.broadcast_shapes(first.shape[:-1],second.shape[:-1],mode.shape)
        l,r=first.expand(*shape,DIM),second.expand(*shape,DIM)
        code=self.mode(mode).expand(*shape,-1)
        return r+self.condition(torch.cat([l,r,l-r,code],-1))

    def marginal(self,z):
        v=self.result(z)[...,:3].sigmoid()
        return v[...,0]+.5*v[...,1]+.25*v[...,2]

    def distribution(self,l,r):
        def branch(a,b):
            return self.pair(torch.cat([a,b,a-b],-1)).unflatten(-1,(3,ATOMS))
        logits=.5*(branch(l,r)+branch(r,l).flip((-1,-2)))
        return logits.flatten(-2).softmax(-1).reshape_as(logits)

    def effect(self,z,values,i,j):
        diff=z[:,i]-z[:,j]
        def head(d):
            return self.residual(torch.cat([d,d.abs()],-1))
        mean=values[:,i]-values[:,j]+.5*(head(diff)[...,:4]-head(-diff)[...,:4])
        return mean,.012+F.softplus(head(diff.abs())[...,4:])

    def against(self,z,ref):
        # z has arbitrary prefix dimensions followed by candidate and feature.
        r=z.gather(-2,ref[...,None,None].expand(*z.shape[:-2],1,z.size(-1)))
        p=self.distribution(z,r.expand_as(z))
        same=torch.arange(z.size(-2),device=z.device)==ref[...,None]
        zero=torch.zeros_like(p)
        zero[...,1,ATOMS//2]=1
        return torch.where(same[...,None,None],zero,p)

    def utilities(self,p):
        v=self.grid+torch.arange(-1,2,device=p.device)[:,None]
        mean=(p*v).sum((-1,-2))
        ce=-((p*(-BETA*v).exp()).sum((-1,-2))).clamp_min(1e-8).log()/BETA
        return mean,ce

    def first_scores(self,d):
        z,_=self.encode(d,8)
        b,k,_=z.shape
        ref=torch.arange(k,device=z.device)[None,:,None].expand(b,k,MODES)
        modes=torch.arange(MODES,device=z.device)[None,None,:,None]
        zz=self.imagine(z[:,:,None,None],z[:,None,None,:],modes.expand(b,k,MODES,k))
        q=self.midpoint(z).softmax(-1)
        m=self.marginal(zz)
        base=m.gather(-1,ref[...,None])
        p=self.against(zz,ref)
        mean,ce=self.utilities(p)
        mixture=(q[...,None,None,None]*p).sum(2,keepdim=True)
        kl=(p*(p.clamp_min(1e-8).log()-mixture.clamp_min(1e-8).log())).sum((-1,-2))
        def feedback(v):
            return (q*v.max(-1).values).sum(-1)
        success=self.result(zz)[...,0].sigmoid()
        success_base=success.gather(-1,ref[...,None])
        success_delta=p[...,2,:].sum(-1)-p[...,0,:].sum(-1)
        return dict(success_marginal=feedback(success),success_pair=feedback(success_base+success_delta),
                    marginal=feedback(m),pooled=((base+mean)*q[...,None]).sum(2).max(-1).values,
                    mean=feedback(base+mean),law=feedback(base+ce),kl=feedback(base+ce+INFO_WEIGHT*kl))

    def last_scores(self,d,ref):
        z,_=self.encode(d,27)
        z=z+self.after(z)
        p=self.against(z,ref)
        mean,ce=self.utilities(p)
        return dict(success_marginal=self.result(z)[...,0].sigmoid(),success_pair=p[...,2,:].sum(-1)-p[...,0,:].sum(-1),
                    marginal=self.marginal(z),pooled=mean,mean=mean,law=ce,kl=ce)
