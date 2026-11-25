"""Factual motion innovations revise the shared physical mechanism and directed graph."""
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import H,K,U,mlp,pairs,pairmask,pool
from whatif.recourse.core import Core as Previous
from whatif.recourse.fit import restore as previous

class Core(Previous):
    def __init__(self):
        super().__init__()
        self.load_state_dict(previous().core.state_dict())
        self.pred_node=mlp(9+7+3,32)
        self.pred_motion=mlp(64,9,48)
        self.correction=mlp(18,H,48)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.innovation_edge=mlp(19,1,32)
        nn.init.zeros_(self.innovation_edge[-1].weight)
        nn.init.zeros_(self.innovation_edge[-1].bias)

    def observed(self,d):
        x,a,y=(d['hx'],d['ha'],d['hy']) if 'hx' in d else (d['f0'][:,None],d['fa'][:,None],d['f1'][:,None])
        role=d['role'][:,None].expand(-1,x.size(1),-1,-1)
        local=torch.cat([x,a,role],-1)
        prior=self.prior(pairs(local))[...,0]
        mask=pairmask(d['role'])
        edge=prior.sigmoid()*mask[:,None]
        src=self.pred_node(local)
        msg=torch.einsum('btij,btil->btjl',edge,src)/edge.sum(-2).clamp_min(1)[...,None]
        estimate=self.pred_motion(torch.cat([src,msg],-1))
        residual=y-x-estimate
        return x,a,y,role,prior,estimate,residual

    def reconstruction(self,d):
        _,_,_,role,_,_,err=self.observed(d)
        weight=role.sum(-1)+2*role[...,0]
        v=F.smooth_l1_loss(err,torch.zeros_like(err),beta=.12,reduction='none').mean(-1)
        return (v*weight).sum()/weight.sum().clamp_min(1)

    def abduct(self,d):
        x,a,y,role,prior,estimate,res=self.observed(d)
        feature=self.fact(torch.cat([x,y-x,a,role],-1))+self.correction(torch.cat([estimate,res],-1))
        alpha=(self.attention(feature)[...,0]+self.time_bias[:x.size(1)][None,:,None]).softmax(1)
        base=(prior*(alpha[:,:,:,None]+alpha[:,:,None,:])*.5).sum(1)
        feature=(alpha[...,None]*feature).sum(1)
        resid=(alpha[...,None]*res).sum(1)
        mask=pairmask(d['role'])
        def degree(logit):
            return (logit.sigmoid()*mask).sum(-1,keepdim=True)/mask.sum(-1,keepdim=True).clamp_min(1)
        def unpack(v):
            logits,mu,s=v.split([K,U,U],-1)
            return logits.softmax(-1)@self.codes,mu,s.clamp(-4,2)
        first=self.latent(torch.cat([feature,degree(base)],-1))
        m,mu,_=unpack(first)
        fixed=base+self.revise(torch.cat([pairs(torch.cat([feature,m,mu],-1)),base[...,None]],-1))[...,0]
        fixed=fixed+self.innovation_edge(torch.cat([pairs(resid),base[...,None]],-1))[...,0]
        m,mu,s=unpack(first+self.refine(torch.cat([feature,degree(base),degree(fixed)],-1)))
        eps=d.get('eps')
        if eps is None:
            eps=torch.randn_like(mu) if self.training else torch.zeros_like(mu)
        kl=pool(.5*(mu.square()+(2*s).exp()-1-2*s),d['role'].sum(-1)).mean()
        return m,mu+s.exp()*eps,fixed,kl
