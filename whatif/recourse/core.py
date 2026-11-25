"""A common object–factor graph for atomic interventions and feedback continuations."""
import torch
from torch import nn
from whatif.mechanism.model import H,U,K,mlp,pairs,pool,pairmask
from whatif.paired.world import Transport
from whatif.paired.fit import world

class Core(Transport):
    def __init__(self):
        super().__init__()
        self.load_state_dict(world().state_dict())
        self.attention=mlp(H,1,24)
        self.time_bias=nn.Parameter(torch.zeros(3))
        self.control_token=mlp(7+11+9+1,7,48)

    def abduct(self,d):
        x,a,y=(d['hx'],d['ha'],d['hy']) if 'hx' in d else (
            d['f0'][:,None],d['fa'][:,None],d['f1'][:,None])
        role=d['role']
        roles=role[:,None].expand(-1,x.size(1),-1,-1)
        ev=self.fact(torch.cat([x,y-x,a,roles],-1))
        al=(self.attention(ev)[...,0]+self.time_bias[:x.size(1)][None,:,None]).softmax(1)
        prior=self.prior(pairs(torch.cat([x,a,roles],-1)))[...,0]
        prior=(prior*(al[:,:,:,None]+al[:,:,None,:])*.5).sum(1)
        ev=(al[...,None]*ev).sum(1)
        valid=pairmask(role)
        def deg(logit):
            return (logit.sigmoid()*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
        def unpack(raw):
            logits,mu,s=raw.split([K,U,U],-1)
            return logits.softmax(-1)@self.codes,mu,s.clamp(-4,2)
        first=self.latent(torch.cat([ev,deg(prior)],-1))
        m,mu,_=unpack(first)
        fixed=prior+self.revise(torch.cat([pairs(torch.cat([ev,m,mu],-1)),prior[...,None]],-1))[...,0]
        m,mu,s=unpack(first+self.refine(torch.cat([ev,deg(prior),deg(fixed)],-1)))
        eps=d.get('eps')
        if eps is None:
            eps=torch.randn_like(mu) if self.training else torch.zeros_like(mu)
        kl=pool(.5*(mu.square()+(2*s).exp()-1-2*s),role.sum(-1)).mean()
        return m,mu+s.exp()*eps,fixed,kl

    def continuation(self,d,bank,duration):
        a=d['a']
        b,k,n=a.shape[:3]
        actor=d['role'][:,None,:,1:2]
        w=bank[None,:,None].expand(b,-1,n,-1)
        goal=(d['goal'][:,None]-d['x'])[:,None].expand(-1,k,-1,-1)
        clock=torch.full_like(a[...,:1],duration/35.)
        token=self.control_token(torch.cat([a,w,goal,clock],-1))*actor
        return self(dict(d,a=a+token))
