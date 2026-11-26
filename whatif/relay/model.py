"""A shared-regime law of first hit times, future poses, rewards and paired action effects."""
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import mlp,pool
from .core import Relay
from .native import OUT

H=4
class World(nn.Module):
    def __init__(self,kind='dynamic'):
        super().__init__()
        self.core=Relay(fixed=kind=='fixed')
        self.scene=mlp(3+9+9+3,32,48)
        self.lift=mlp(33+32+18,64,96)
        self.regime=mlp(16+50,H,48)
        self.hit=mlp(64,H,64)
        self.point=mlp(64,10,80)
        self.emission=mlp(64,H*10*3,96)
        self.score=mlp(64,2,56)

    def forward(self,d):
        g=self.core(d)
        role,x=d['role'],d['x']
        mover=pool(x,role[...,0])
        rot=torch.stack([mover[:,3:6],mover[:,6:],torch.cross(mover[:,3:6],mover[:,6:],dim=-1)],-1)
        def frame(v):
            return (v.unflatten(-1,(3,3))@rot[:,None]).flatten(-2)
        local=frame(x-mover[:,None])
        motion=frame(d['f1']-d['f0'])
        scene=self.scene(torch.cat([role,local,motion,x[...,:3]],-1))
        gap=((d['goal']-mover).unflatten(-1,(3,3))@rot).flatten(1)
        ctx=torch.cat([pool(scene,role.sum(-1)),gap,mover],-1)
        tr=g['track']
        heads=self.backup_heads if 'adaptive' in d else self
        z=heads.lift(torch.cat([tr,ctx[:,None,None].expand(-1,tr.size(1),5,-1)],-1))
        w=heads.regime(torch.cat([g['context'],ctx],-1)).softmax(-1)
        hazard=heads.hit(z).sigmoid()
        surv=(1-hazard).cumprod(2)
        timing=torch.cat([hazard*torch.cat([torch.ones_like(surv[:,:,:1]),surv[:,:,:-1]],2),surv[:,:,-1:]],2).transpose(2,3)
        p=(timing*w[:,None,:,None]).sum(2)
        motion=((pool(g['trajectory'],role[:,None,None,:,0])-mover[:,None,None]).unflatten(-1,(3,3))@rot[:,None,None]).flatten(-2)
        point=heads.point(z)
        mu=torch.cat([motion+point[...,:9],point[...,9:]],-1)
        raw=heads.emission(z).unflatten(-1,(H,10,3))
        offset,tau,load=raw.unbind(-1)
        offset=offset-(offset*w[:,None,None,:,None]).sum(3,keepdim=True)
        return dict(w=w,timing=timing,p=p,mu=mu,offset=offset,tau=.025+F.softplus(tau),
                    load=load,score=heads.score(z[:,:,-1]).sigmoid(),z=z),g

def coupled(o,i,j,independent=False):
    p=o['p']
    if independent:
        return p[:,i,:,None]*p[:,j,None,:]
    t=o['timing']
    return torch.einsum('bh,biht,bihs->bits',o['w'],t[:,i],t[:,j])

def difference(o,i,j,independent=False,detach_mean=False):
    # Output is [layout, pair, horizon, mixture component, coordinate].
    mu=o['mu'].detach() if detach_mean else o['mu']
    v=mu.unsqueeze(-2)+o['offset']
    t,l,w=o['tau'],o['load'],o['w']
    if independent:
        m=(v[:,i].unsqueeze(4)-v[:,j].unsqueeze(3)).flatten(3,4)
        marginal_var=t.square()+l.square()
        sd=(marginal_var[:,i].unsqueeze(4)+marginal_var[:,j].unsqueeze(3)).sqrt().flatten(3,4)
        return m,sd,(w[:,:,None]*w[:,None,:]).flatten(1)
    sd=(t[:,i].square()+t[:,j].square()+(l[:,i]-l[:,j]).square()).sqrt()
    return v[:,i]-v[:,j],sd,w

def restore(kind='dynamic',dev='cpu'):
    net=World(kind.removesuffix('-full')).to(dev)
    if kind.endswith('-full'):
        from .readout import Endpoint
        net.core.atomic_readout=Endpoint().to(dev)
    state=torch.load(OUT/f'{kind}.pt',map_location=dev)
    net.load_state_dict(state['model'])
    return net.eval()
