"""Anchored graph dynamics. Endpoint evidence and edge strength enter propagation locally."""
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import Mechanism,H,U,FACTORS,WIDTH,mlp,pairs,tile,pairmask,pool
from whatif.mechanism.native import A,D


class Transport(Mechanism):
    def __init__(self,use_transfer=True):
        super().__init__(directed=True)
        self.use_transfer=use_transfer
        self.node=mlp(2*D+3+12+U,H)
        self.relation=mlp(2*(H+A)+1+6,H)
        self.edge=mlp(H,1+2*FACTORS**2)
        self.transfer=mlp(H,2)
        self.update=nn.ModuleList([nn.GRUCell(WIDTH,WIDTH) for _ in range(4)])

    def forward(self,d):
        x,role,commands,goal=(d[k] for k in ('x','role','a','goal'))
        b,k,n,_=commands.shape
        # Reference is evaluated under the identical abduction and disturbance draw.
        act=torch.cat([commands,d['fa'][:,None]],1)
        count=k+1
        phys,u,factual,kl=self.abduct(d)
        observed=d['f1']-d['f0']
        base=self.node(torch.cat([x,role,phys,u,observed],-1))
        base=tile(base,count)
        cmd=act.flatten(0,1)
        mem=tile(factual,count)
        coords=torch.cat([x[...,:3],observed[...,:3]],-1)
        rel=tile(coords[:,None]-coords[:,:,None],count)
        relation=self.relation(torch.cat([pairs(torch.cat([base,cmd],-1)),mem[...,None],rel],-1))
        edge=self.edge(relation)
        logit=mem+edge[...,0]
        pc=logit.sigmoid()
        hard=(pc>=.5).to(pc)+(pc-pc.detach())
        direction,magnitude=self.transfer(relation).unbind(-1)
        conductance=(direction+1.5).sigmoid()
        impact=F.softplus(magnitude)
        fg,fw=edge[...,1:].chunk(2,-1)
        pg=fg.sigmoid()
        factor=(pg>=.5).to(pg)+(pg-pg.detach())
        scale=hard*tile(pairmask(role),count)
        if self.use_transfer:
            scale=scale*conductance*(1+impact.tanh())
        weights=(factor*fw.tanh()).unflatten(-1,(FACTORS,FACTORS))*scale[...,None,None]
        state=torch.stack([net(torch.cat([base,tile(x[...,3*i:3*i+3],count)],-1))
                           for i,net in enumerate(self.factor)],-2)
        state=state+self.source(cmd).unflatten(-1,(FACTORS,WIDTH))
        for update in self.update:
            msg=torch.einsum('bijfg,bifh->bjgh',weights,state)
            state=state+update(msg.reshape(-1,WIDTH),state.reshape(-1,WIDTH)).reshape_as(state)
        delta=torch.cat([net(state[:,:,i]) for i,net in enumerate(self.delta)],-1).reshape(b,count,n,D)
        same=d['same'].view(b,1,1,1)
        future=x[:,None]+delta[:,:k]+same*(d['f1'][:,None]-x[:,None]-delta[:,k:])
        hidden=state.reshape(b,count,n,FACTORS*WIDTH)[:,:k]
        moving=role[:,None,:,0]
        g=goal[:,None].expand(-1,k,-1)
        hr=torch.cat([pool(hidden,moving),pool(future,moving)-g],-1)
        view=lambda v:v.reshape(b,count,n,n)[:,:k]
        return dict(y=future,r=self.reward(torch.cat([hr,g],-1)),h=hidden,hr=hr,
                    fact=factual,edge=view(logit),causal=view(torch.logit((pc*conductance).clamp(1e-6,1-1e-6))),
                    impact=view(impact),sparse=(scale[...,None]*pg).abs().mean(),kl=kl)
