"""Pool observed transitions before inferring a shared mechanism, residual and factual graph."""
import torch
from torch import nn
from .model import Facts
from whatif.mechanism.model import K,U,H,mlp,pairs,pairmask,pool

class HistoryFacts(Facts):
    def __init__(self):
        super().__init__()
        self.attention=mlp(H,1,24)
        self.recency=nn.Parameter(torch.zeros(4))

    def forward(self,d):
        x,a,y,role=[d[k] for k in ('hx','ha','hy','role')]
        roles=role[:,None].expand(-1,x.size(1),-1,-1)
        ev=self.fact(torch.cat([x,y-x,a,roles],-1))
        alpha=(self.attention(ev)[...,0]+self.recency[None,:,None]).softmax(1)
        prior=self.prior(pairs(torch.cat([x,a,roles],-1)))[...,0]
        prior=(.5*(alpha[:,:,:,None]+alpha[:,:,None,:])*prior).sum(1)
        ev=(alpha[...,None]*ev).sum(1)
        valid=pairmask(role)
        def degree(logit):
            return (logit.sigmoid()*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
        init=self.latent(torch.cat([ev,degree(prior)],-1))
        def unpack(raw):
            logits,mu,s=raw.split([K,U,U],-1)
            return logits.softmax(-1)@self.codes,mu,s.clamp(-4,2)
        m,mu,_=unpack(init)
        factual=prior+self.revise(torch.cat([pairs(torch.cat([ev,m,mu],-1)),prior[...,None]],-1))[...,0]
        m,mu,logsd=unpack(init+self.refine(torch.cat([ev,degree(prior),degree(factual)],-1)))
        eps=d.get('eps')
        if eps is None:
            eps=torch.randn_like(mu) if self.training else torch.zeros_like(mu)
        kl=pool(.5*(mu.square()+(2*logsd).exp()-1-2*logsd),role.sum(-1)).mean()
        return m,mu+logsd.exp()*eps,factual,kl
