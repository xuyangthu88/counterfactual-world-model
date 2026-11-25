"""A measured no-change policy and an explicit point mass for identical paired returns."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from whatif.mechanism.model import mlp
from .backup import Coupled,loss,rows
from .fit import restore as world
from .load import read,take,early
from .native import OUT,DATA,DOMAINS,BANK
from .continuations import CACHE
from .model import BETA,INFO_WEIGHT

class Anchored(Coupled):
    def __init__(self):
        super().__init__()
        self.change=mlp(3*64,52,96)

    def paired(self,x,o):
        h=self.encode((x-self.center)/self.scale)
        ref=h[:,:1].expand_as(h)
        v=self.change(torch.cat([h,ref,h-ref],-1))
        i=torch.arange(x.size(1),device=x.device)
        prior=self.diff(o,i,torch.zeros_like(i))
        atom=v[...,0].sigmoid()
        conditional=(prior.clamp_min(1e-7).log()+v[...,1:]).softmax(-1)
        zero=torch.zeros_like(conditional)
        zero[...,25]=1
        p=atom[...,None]*zero+(1-atom[...,None])*conditional
        return torch.cat([zero[:,:1],p[:,1:]],1),atom,conditional

    def decisions(self,x):
        o=self(x)
        pi,ps,mu,sd=o
        p,_,_=self.paired(x,o)
        marginal=((ps@self.support)*pi[:,None]).sum(-1)
        mean=p@self.delta
        ce=-((p*(-BETA*self.delta).exp()).sum(-1)).clamp_min(1e-8).log()/BETA
        sym=.5*(p+p.flip((-1,)))
        signed=(p*(p.clamp_min(1e-8).log()-sym.clamp_min(1e-8).log())*self.delta.sign()).sum(-1)
        return dict(anchor_marginal=marginal,anchor_mean=mean,anchor_law=ce,anchor_kl=ce+INFO_WEIGHT*signed)

def objective(net,x,y,v):
    coupled=loss(net,x,y,v)
    o=net(x)
    _,atom,cond=net.paired(x,o)
    s,peak,last=y.unbind(-1)
    aux=.5*peak+.25*last
    diff=aux[:,1:]-aux[:,:1]
    disc=s[:,1:]-s[:,:1]
    unchanged=(diff.abs()<1e-6)&(disc==0)
    loc=((diff+.75)*16/1.5).clamp(0,16)
    lo=loc.floor().long()
    group=(1+disc.long())*17
    def nll(bin):
        return -cond[:,1:].gather(-1,(group+bin)[...,None]).squeeze(-1).clamp_min(1e-8).log()
    conditional=(1-loc.frac())*nll(lo)+loc.frac()*nll((lo+1).clamp_max(16))
    return (coupled+F.binary_cross_entropy(atom[:,1:],unchanged.float())
            +(conditional*(~unchanged)).sum()/(~unchanged).sum().clamp_min(1))

@torch.no_grad()
def full_feature(net,d):
    return net.encode(early(d),35)[0][:,:1]

def data(dom,split,device):
    x,y,v=rows(dom,split,device)
    full=torch.as_tensor(np.load(CACHE/f'anchor-{dom}-{split}.npy'),device=device)
    with np.load(DATA/f'anchor-{dom}-{split}.npz') as z,np.load(DATA/f'backup-{dom}-{split}.npz') as f:
        starts=f['x'][::len(BANK)]
        role=f['role'][::len(BANK),:,0:1]
        if not np.array_equal(z['seed'],f['seed'][::len(BANK)]):
            raise ValueError('unmatched full factual policy')
        state=((z['final']-starts)*role).sum(1)/role.sum(1).clip(min=1)
        value=np.concatenate([state,z['native'][:,None]],-1)
        first_y=torch.tensor(z['outcome'],device=device).float()
        first_v=torch.tensor(value,device=device).float()
    return torch.cat([full,x],1),torch.cat([first_y[:,None],y],1),torch.cat([first_v[:,None],v],1)

def restore(device='cpu'):
    net=Anchored().to(device)
    net.load_state_dict(torch.load(OUT/'anchor.pt',map_location=device)['state'])
    return net.eval()

def train(features_only=False):
    torch.manual_seed(271)
    torch.set_num_threads(2)
    physical=world('cuda')
    with torch.no_grad():
        for split in ('train','val','test'):
            for dom in DOMAINS:
                # Only one common initial observation per layout is needed.
                with np.load(DATA/f'backup-{dom}-{split}.npz') as z:
                    ix=np.arange(0,len(z['seed']),len(BANK))
                    d={k:torch.as_tensor(z[k][ix],device='cuda').float() for k in ('x','role','f0','f1','fa','goal','seed')}
                from whatif.feedback.native import command
                a=np.stack([command(x,r,g,None) for x,r,g in zip(*[d[k].cpu().numpy() for k in ('x','role','goal')])])
                # The graph has independent branch calculations; the other six
                # columns can duplicate the measured command since only column zero is read.
                d['a']=torch.as_tensor(a,device='cuda')[:,None].expand(-1,len(BANK),-1,-1)
                parts=[full_feature(physical,take(d,j)).cpu() for j in torch.arange(len(a),device='cuda').split(48)]
                np.save(CACHE/f'anchor-{dom}-{split}.npy',torch.cat(parts).numpy())
    if features_only:
        return
    def load(split):
        return tuple(torch.cat(v) for v in zip(*[data(dom,split,'cuda') for dom in DOMAINS]))
    x,y,v=load('train')
    vx,vy,vv=load('val')
    net=Anchored().cuda()
    net.load_state_dict(torch.load(OUT/'backup.pt',map_location='cuda')['state'],strict=False)
    opt=torch.optim.AdamW(net.parameters(),lr=.00065)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,48,eta_min=.00007)
    hist=[]
    best=float('inf')
    for epoch in range(1,49):
        for ix in torch.randperm(len(x),device='cuda').split(80):
            obj=objective(net,x[ix],y[ix],v[ix])
            opt.zero_grad()
            obj.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),4)
            opt.step()
        sched.step()
        with torch.no_grad():
            value=sum(len(ix)*objective(net,vx[ix],vy[ix],vv[ix]).item()
                      for ix in torch.arange(len(vx),device='cuda').split(64))/len(vx)
        hist.append(value)
        if value<best:
            best=value
            torch.save(dict(epoch=epoch,state=net.state_dict()),OUT/'anchor.pt')
        if epoch%8==0:
            print(epoch,value,flush=True)
    (OUT/'anchor-fit.json').write_text(json.dumps(hist))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--features-only',action='store_true')
    train(parser.parse_args().features_only)
