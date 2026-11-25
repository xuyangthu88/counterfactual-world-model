"""Action paths over the eight macros before the second decision, learned from contact labels."""
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import mlp,pairs,pairmask,pool

class ProgramGraph(nn.Module):
    def __init__(self,bank):
        super().__init__()
        self.register_buffer('code',(bank-bank.mean(0))/bank.std(0).clamp_min(.02))
        self.node=mlp(9+3+12+4+9+9,32,64)
        self.source=mlp(11+7+1+32,24,64)
        self.relation=mlp(2*(32+24)+1,1+9,64)
        self.read=mlp(24+64,64,96)

    def forward(self,d,z,physics):
        m,u,mem=physics
        x=d['x']
        role=d['role']
        center=pool(x,role[...,0])
        rot=torch.stack([center[:,3:6],center[:,6:],torch.cross(center[:,3:6],center[:,6:],dim=-1)],-1)
        def frame(v):
            return (v.reshape(*v.shape[:-1],3,3)@rot[:,None]).flatten(-2)
        node=self.node(torch.cat([frame(x-center[:,None]),role,m,u,frame(d['f1']-d['f0']),frame(d['goal'][:,None]-x)],-1))
        a=d['a']
        b,k,n=a.shape[:3]
        clock=torch.full_like(a[...,:1],8/35)
        clock[:,0]=1
        code=torch.cat([self.code[:1],self.code])[None,:,None].expand(b,-1,n,-1)
        node=node[:,None].expand(-1,k,-1,-1)
        token=self.source(torch.cat([code,a,clock,node],-1))*role[:,None,:,1:2]
        pair=self.relation(torch.cat([pairs(torch.cat([node,token],-1)),mem[:,None,:,:,None].expand(-1,k,-1,-1,-1)],-1))
        logit=mem[:,None]+pair[...,0]
        p=logit.sigmoid()
        gate=((p>=.5).to(p)+(p-p.detach()))*pairmask(role)[:,None]
        weights=gate[...,None,None]*pair[...,1:].unflatten(-1,(3,3)).tanh()
        msg=torch.einsum('bkijfg,bkifw->bkjgw',weights,token.unflatten(-1,(3,8)))
        received=pool(msg.flatten(-2),role[:,None,:,0])
        return z+self.read(torch.cat([z,received],-1)),logit
