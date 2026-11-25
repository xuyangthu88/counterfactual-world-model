"""Permutation-equivariant object graph with local, gated factor-to-factor messages."""
import torch
from torch import nn
from torch.nn import functional as F
from .native import A, D

H, K, U, FACTORS, WIDTH = 40, 16, 4, 3, 8


def mlp(i,o,h=H):
    return nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,o))


def pairs(h):
    n = h.shape[-2]
    return torch.cat([h.unsqueeze(-2).expand(*h.shape[:-2],n,n,h.size(-1)),
                      h.unsqueeze(-3).expand(*h.shape[:-2],n,n,h.size(-1))],-1)


def tile(v,k):
    return v[:,None].expand(-1,k,*v.shape[1:]).flatten(0,1)


def pool(v,mask):
    return (v*mask[...,None]).sum(-2)/mask.sum(-1,keepdim=True).clamp_min(1)


def pairmask(role):
    present = role.sum(-1)
    n = present.size(-1)
    return present[:,:,None]*present[:,None,:]*(1-torch.eye(n,device=role.device))


def labeled(role):
    return pairmask(role)*((role[:,:,None,0]+role[:,None,:,0])>0)


class Mechanism(nn.Module):
    def __init__(self,directed=False):
        super().__init__()
        self.transfer = mlp(2*(H+A)+1,2) if directed else None
        self.fact = mlp(D*2+A+3,H)
        self.prior = mlp(2*(D+A+3),1)
        self.latent = mlp(H+1,K+2*U)
        self.codes = nn.Parameter(torch.randn(K,12)*.2)
        self.revise = mlp(2*(H+12+U)+1,1)
        self.refine = mlp(H+2,K+2*U)
        self.node = mlp(D+3+12+U,H)
        self.edge = mlp(2*(H+A)+1,1+2*FACTORS**2)
        self.source = nn.Linear(A,FACTORS*WIDTH,bias=False)
        self.factor = nn.ModuleList([mlp(H+3,WIDTH) for _ in range(FACTORS)])
        self.update = nn.ModuleList([mlp(2*WIDTH,WIDTH) for _ in range(2)])
        self.delta = nn.ModuleList([mlp(WIDTH,3) for _ in range(FACTORS)])
        self.reward = mlp(2*D+FACTORS*WIDTH,3)
        self.state_pair = mlp(FACTORS*WIDTH,2*D)
        self.reward_pair = mlp(FACTORS*WIDTH+D,6)

    def abduct(self,d):
        f,fa,end,role = (d[k] for k in ('f0','fa','f1','role'))
        present = role.sum(-1)
        valid = pairmask(role)
        prior = self.prior(pairs(torch.cat([f,fa,role],-1))).squeeze(-1)
        evidence = self.fact(torch.cat([f,end-f,fa,role],-1))
        def degree(logits):
            return (logits.sigmoid()*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
        def unpack(v):
            coef,mu,logsd = v.split([K,U,U],-1)
            return coef.softmax(-1)@self.codes, mu, logsd.clamp(-4,2)
        first = self.latent(torch.cat([evidence,degree(prior)],-1))
        phys,mu,_ = unpack(first)
        local = torch.cat([evidence,phys,mu],-1)
        corrected = prior+self.revise(torch.cat([pairs(local),prior[...,None]],-1)).squeeze(-1)
        final = first+self.refine(torch.cat([evidence,degree(prior),degree(corrected)],-1))
        phys,mu,logsd = unpack(final)
        epsilon = d.get('eps')
        if epsilon is None:
            epsilon = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
        same = d['same'].view(-1,1,1)
        z = same*(mu+epsilon*logsd.exp())+(1-same)*epsilon
        kl = pool(.5*(mu.square()+(2*logsd).exp()-1-2*logsd),present).mean()
        return phys,z,corrected,kl

    def forward(self,d):
        x,role,act,goal = (d[k] for k in ('x','role','a','goal'))
        b,k,n,_ = act.shape
        phys,u,fact,kl = self.abduct(d)
        base = self.node(torch.cat([x,role,phys,u],-1))
        command = act.flatten(0,1)
        features = tile(base,k)
        mem = tile(fact,k)
        edge_features = torch.cat([pairs(torch.cat([features,command],-1)),mem[...,None]],-1)
        e = self.edge(edge_features)
        logits = mem+e[...,0]
        prob = logits.sigmoid()
        directed = {}
        if self.transfer is not None:
            evidence,magnitude = self.transfer(edge_features).unbind(-1)
            prob = prob*(evidence+1.5).sigmoid()
            directed = dict(causal=torch.logit(prob.clamp(1.e-6,1-1.e-6)).reshape(b,k,n,n),
                            impact=F.softplus(magnitude).reshape(b,k,n,n))
        hard = (prob >= .5).to(prob)
        gate = (hard+(prob-prob.detach()))*tile(pairmask(role),k)
        fg,fw = e[...,1:].chunk(2,-1)
        fp = fg.sigmoid()
        factor_gate = (fp>=.5).to(fp)+(fp-fp.detach())
        weights = (factor_gate*fw.tanh()).unflatten(-1,(FACTORS,FACTORS))*gate[...,None,None]
        s = torch.stack([net(torch.cat([features,tile(x[...,3*j:3*j+3],k)],-1))
                         for j,net in enumerate(self.factor)],-2)
        # The intervention token has one edge to the addressed node. No pooled action feature.
        s = s+self.source(command).unflatten(-1,(FACTORS,WIDTH))
        for net in self.update:
            msg = torch.einsum('bijfg,bifh->bjgh',weights,s)
            s = s+net(torch.cat([s,msg],-1))
        pred = tile(x,k)+torch.cat([net(s[:,:,j]) for j,net in enumerate(self.delta)],-1)
        hidden = s.flatten(-2)
        movable = tile(role[...,0],k)
        future = pool(pred,movable)
        g = tile(goal,k)
        reward_hidden = torch.cat([pool(hidden,movable),future-g],-1)
        r = self.reward(torch.cat([reward_hidden,g],-1))
        reshape = lambda v:v.reshape(b,k,*v.shape[1:])
        return dict(y=reshape(pred),r=reshape(r),edge=reshape(logits),fact=fact,kl=kl,
                    h=reshape(hidden),hr=reshape(reward_hidden),sparse=(fp*gate[...,None]).abs().mean(),**directed)

    def contrast(self,out,i,j):
        def contrast_head(mean,hidden,head):
            delta = hidden[:,i]-hidden[:,j]
            plus = head(delta).chunk(2,-1)[0]
            neg = head(-delta).chunk(2,-1)[0]
            # Scale uses only even features and is zero for identical branch representations.
            even = head(delta.abs()).chunk(2,-1)[1]
            sd = F.softplus(even)*delta.abs().mean(-1,keepdim=True)
            return mean[:,i]-mean[:,j]+.5*(plus-neg),sd
        return (contrast_head(out['y'],out['h'],self.state_pair),
                contrast_head(out['r'],out['hr'],self.reward_pair))
