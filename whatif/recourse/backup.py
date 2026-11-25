"""Joint potential outcomes for initial actions followed by the same learned recourse policy."""
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import mlp
from .model import DIM,BETA,INFO_WEIGHT
from .continuations import CACHE
from .native import DATA,OUT,BANK,DOMAINS

HYP=6
LEVELS=9
def rows(dom,split,device):
    path=DATA/f'target-{dom}-{split}.npz'
    with np.load(DATA/f'backup-{dom}-{split}.npz') as fact,np.load(path) as target:
        if not np.array_equal(target['seed'],fact['seed']) or not np.array_equal(target['first'],fact['first']):
            raise ValueError('unmatched training return')
        k=len(BANK)
        if not np.array_equal(target['first'].reshape(-1,k),np.tile(np.arange(k),(len(target['first'])//k,1))):
            raise ValueError('candidate order')
        mover=fact['role'][...,0:1]
        future=((target['final']-fact['x'])*mover).sum(1)/mover.sum(1).clip(min=1)
        values=np.concatenate([future,target['native'][:,None]],-1).reshape(-1,k,10)
        out=target['outcome'].reshape(-1,k,3)
    features=np.load(CACHE/f'features-{dom}-{split}.npy')
    return tuple(torch.as_tensor(v,device=device).float() for v in (features,out,values))

class Coupled(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('center',torch.zeros(DIM))
        self.register_buffer('scale',torch.ones(DIM))
        self.encode=mlp(DIM,64,96)
        self.prior=mlp(64,HYP,56)
        self.atom=mlp(64,HYP*2*LEVELS,96)
        self.state=mlp(64,HYP*20,96)
        atoms=torch.arange(2*LEVELS)
        index=((atoms[:,None]//LEVELS-atoms//LEVELS+1)*(2*LEVELS-1)
               +(atoms[:,None]%LEVELS-atoms%LEVELS+LEVELS-1))
        self.register_buffer('map',F.one_hot(index.flatten(),3*(2*LEVELS-1)).float())
        self.register_buffer('support',atoms//LEVELS+.75*(atoms%LEVELS)/(LEVELS-1))
        self.register_buffer('delta',(torch.arange(-1,2)[:,None]+torch.linspace(-.75,.75,2*LEVELS-1)).flatten())

    def forward(self,features):
        h=self.encode((features-self.center)/self.scale)
        pi=self.prior(h[:,0]).softmax(-1)
        p=self.atom(h).unflatten(-1,(HYP,2*LEVELS)).softmax(-1)
        mu,raw=self.state(h).unflatten(-1,(HYP,20)).chunk(2,-1)
        return pi,p,mu,.018+F.softplus(raw)

    def diff(self,o,i,j,independent=False):
        pi,p,mu,sd=o
        if independent:
            m=(p*pi[:,None,:,None]).sum(2)
            joint=m[:,i,:,None]*m[:,j,None,:]
        else:
            joint=torch.einsum('bm,bpml,bpmr->bplr',pi,p[:,i],p[:,j])
        return joint.flatten(-2)@self.map

    def scores(self,o):
        pi,p,mu,sd=o
        k=p.size(1)
        i=torch.arange(k,device=p.device)
        j=torch.ones_like(i) # fixed Stage 23 population policy, same for both domains
        m=((p@self.support)*pi[:,None]).sum(-1)
        res=dict(backup_marginal=m)
        for independent,name in [(True,'backup_independent'),(False,'backup_law')]:
            law=self.diff(o,i,j,independent)
            ce=-((law*(-BETA*self.delta).exp()).sum(-1)).clamp_min(1e-8).log()/BETA
            ce[:,1]=0
            res[name]=ce
        # Exact algebraic expectation; identical branches use the marginal tie order.
        res['backup_mean']=m.double()-m[:,1:2].double()
        # Signed divergence from a sign-symmetric distribution of G differences.
        symmetric=.5*(law+law.flip((-1,)))
        term=law*(law.clamp_min(1e-8).log()-symmetric.clamp_min(1e-8).log())
        signed=(term*self.delta.sign()).sum(-1)
        signed[:,1]=0
        res['backup_kl']=res['backup_law']+INFO_WEIGHT*signed
        return res

def labels(y):
    s,peak,last=y.unbind(-1)
    aux=.5*peak+.25*last
    v=(aux*(LEVELS-1)/.75).clamp(0,LEVELS-1)
    lo=v.floor().long()
    def one(t):
        return F.one_hot(s.long()*LEVELS+t,2*LEVELS)
    return (1-v.frac()[...,None])*one(lo)+v.frac()[...,None]*one((lo+1).clamp_max(LEVELS-1))

def loss(net,x,y,values):
    o=net(x)
    pi,p,mu,sd=o
    lp=pi.clamp_min(1e-8).log()
    t=labels(y)
    within=(t[:,:,None]*p.clamp_min(1e-8).log()).sum(-1)
    joint=-(lp+within.sum(1)).logsumexp(-1).mean()/y.size(1)
    marginal=-(lp[:,None]+within).logsumexp(-1).mean()
    i,j=torch.triu_indices(y.size(1),y.size(1),1,device=y.device)
    law=net.diff(o,i,j)
    observed=(t[:,i,:,None]*t[:,j,None,:]).flatten(-2)@net.map
    pair=-(observed*law.clamp_min(1e-8).log()).sum(-1).mean()
    def nll(target,loc,sigma):
        logp=-sigma.log()-.5*((target[:,:,None]-loc)/sigma).square()
        return -(lp[:,None,:,None]+logp).logsumexp(2).mean()
    pos=nll(values,mu,sd)
    paired=nll(values[:,i]-values[:,j],mu[:,i]-mu[:,j],(sd[:,i].square()+sd[:,j].square()).sqrt())
    point=(pi[:,None,:,None]*mu).sum(2)
    return joint+.45*marginal+.5*pair+.18*pos+.10*paired+.6*F.smooth_l1_loss(point,values,beta=.15)

def restore(device='cpu'):
    net=Coupled().to(device)
    net.load_state_dict(torch.load(OUT/'backup.pt',map_location=device)['state'])
    return net.eval()

def train():
    torch.manual_seed(271)
    torch.set_num_threads(2)
    def load(split):
        ds=[rows(dom,split,'cuda') for dom in DOMAINS]
        return tuple(torch.cat(v) for v in zip(*ds))
    x,y,v=load('train')
    vx,vy,vv=load('val')
    net=Coupled().cuda()
    net.center.copy_(x.mean((0,1)))
    net.scale.copy_(x.flatten(0,1).std(0).clamp_min(.05))
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,56,eta_min=.0001)
    best=float('inf')
    history=[]
    for epoch in range(1,57):
        net.train()
        for ix in torch.randperm(len(x),device=x.device).split(80):
            obj=loss(net,x[ix],y[ix],v[ix])
            opt.zero_grad()
            obj.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),4)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            score=sum(len(idx)*loss(net,vx[idx],vy[idx],vv[idx]).item()
                      for idx in torch.arange(len(vx),device=x.device).split(64))/len(vx)
        history.append(score)
        if score<best:
            best=score
            torch.save(dict(epoch=epoch,state=net.state_dict()),OUT/'backup.pt')
        if epoch%8==0:
            print(epoch,score,flush=True)
    (OUT/'backup-fit.json').write_text(json.dumps(history))

if __name__=='__main__':
    train()
