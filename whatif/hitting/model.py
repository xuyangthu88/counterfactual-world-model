"""A shared graph forecaster with an exactly marginal-preserving neural time copula."""
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import mlp,pool
from .core import Core
from .native import BANK

D=64
CATEGORIES=6
STRATA=8
FACTORS=3

def copula_basis():
    n=STRATA
    k=torch.arange(n)
    matrices=[torch.full((n,n),1/n**2)]
    for sign in (1,-1):
        for shift in range(n):
            matrices.append(F.one_hot((sign*k+shift)%n,n).float()/n)
    return torch.stack(matrices)

class Outcome(nn.Module):
    def __init__(self):
        super().__init__()
        self.time=mlp(D,CATEGORIES,72)
        self.scores=mlp(D,2,64)
        self.horizon=nn.Parameter(torch.randn(5,12)*.1)
        self.future=mlp(D+12,10*(2+FACTORS),96)
        self.depend=mlp(D*3,2*STRATA+1,80)
        self.register_buffer('basis',copula_basis().flatten(1))
        self.register_buffer('grid',torch.arange(STRATA+1).float()/STRATA)

    def forward(self,z):
        p=self.time(z).softmax(-1)
        clock=self.horizon.expand(*z.shape[:-1],-1,-1)
        v=self.future(torch.cat([z[:,:,None].expand(-1,-1,5,-1),clock],-1)).unflatten(-1,(10,2+FACTORS))
        return dict(z=z,p=p,score=self.scores(z).sigmoid(),mu=v[...,0],
                    tau=.018+F.softplus(v[...,1]),loading=v[...,2:])

    def cells(self,p):
        hi=p.cumsum(-1)[...,None]
        lo=hi-p[...,None]
        return STRATA*(torch.minimum(hi,self.grid[1:])-torch.maximum(lo,self.grid[:-1])).clamp_min(0)

    def coupling(self,o,i,j,detached_marginals=False):
        p=o['p'].detach() if detached_marginals else o['p']
        l,r=o['z'][:,i],o['z'][:,j]
        def kernel(a,b):
            prob=self.depend(torch.cat([a,b,a-b],-1)).softmax(-1)
            return (prob@self.basis).unflatten(-1,(STRATA,STRATA))
        w=.5*(kernel(l,r)+kernel(r,l).transpose(-1,-2))
        # Both margins of w are uniform. The independence floor bounds rare-cell log loss.
        w=.99*w+.01/(STRATA**2)
        cell=self.cells(p)
        return cell[:,i]@w@cell[:,j].transpose(-1,-2)

    def normal(self,o,i,j,independent=False):
        f=o['loading']
        common=(f[:,i].square()+f[:,j].square() if independent else (f[:,i]-f[:,j]).square()).sum(-1)
        variance=common+o['tau'][:,i].square()+o['tau'][:,j].square()
        return o['mu'][:,i]-o['mu'][:,j],variance.sqrt()

class World(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=Core()
        self.register_buffer('bank',torch.tensor(BANK).float())
        self.scene=mlp(3+9+9+3,32,48)
        self.lift=mlp(33+32+18,D,96)
        self.head=Outcome()

    def encode(self,d,early):
        a,role,x=d['a'],d['role'],d['x']
        bank=torch.cat([self.bank[:1],self.bank]) if early else self.bank
        role_actor=role[:,None,:,1:2]
        b,k,n=a.shape[:3]
        codes=bank[None,:,None].expand(b,-1,n,-1)
        goals=(d['goal'][:,None]-x)[:,None].expand(-1,k,-1,-1)
        clock=torch.full_like(a[...,:1],8/35. if early else 27/35.)
        if early:
            clock[:,0]=1.
        token=self.core.control_token(torch.cat([a,codes,goals,clock],-1))*role_actor
        o=self.core(dict(d,a=a+token))
        mover=pool(x,role[...,0])
        rot=torch.stack([mover[:,3:6],mover[:,6:9],torch.cross(mover[:,3:6],mover[:,6:9],dim=-1)],-1)
        local=((x-mover[:,None]).reshape(b,n,3,3)@rot[:,None]).flatten(-2)
        vel=((d['f1']-d['f0']).reshape(b,n,3,3)@rot[:,None]).flatten(-2)
        node=self.scene(torch.cat([role,local,vel,x[...,:3]],-1))
        gap=((d['goal']-mover).reshape(b,3,3)@rot).flatten(1)
        ctx=torch.cat([pool(node,role.sum(-1)),gap,mover],-1)
        return self.lift(torch.cat([o['hr'],ctx[:,None].expand(-1,k,-1)],-1)),o

    def forward(self,d,early=False):
        z,o=self.encode(d,early)
        if early and hasattr(self,'program_graph'):
            z,o['program_edge']=self.program_graph(d,z,self.core.abduct(d)[:3])
        return self.head(z),o
