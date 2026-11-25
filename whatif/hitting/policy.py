"""A shared policy fitted to measured sibling outcomes under nested distribution inputs."""
import argparse
import json
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import mlp,pool
from .data import data,take
from .fit import restore,pairs
from .native import OUT,DOMAINS

MODES=('means','independent','shape','kl')
MEAN=24
SHAPE=36+20
def ordered(k,device):
    i=torch.arange(k,device=device)[:,None].expand(k,k)
    j=i.T
    mask=i!=j
    return i[mask],j[mask]

@torch.no_grad()
def features(net,d,early):
    o,_=net(d,early)
    p=o['p']
    b,k,t=p.shape
    i,j=ordered(k,p.device)
    independent=p[:,i,:,None]*p[:,j,None,:]
    joint=net.head.coupling(o,i,j)
    means=torch.cat([1-p[...,-1:],(p*torch.arange(t,device=p.device)/(t-1)).sum(-1,keepdim=True),
                     o['score'],o['mu'][...,[0,1,2,9]].flatten(-2)],-1)
    def vector(law,ind):
        _,sd=net.head.normal(o,i,j,independent=ind)
        return torch.cat([law.flatten(-2),sd[...,[0,1,2,9]].clamp_min(1e-4).log().flatten(-2)],-1).reshape(b,k,k-1,SHAPE)
    # KL between the predicted lead/lag law and its independent-marginals counterpart.
    axis=torch.arange(t,device=p.device)
    proj=F.one_hot((axis[None,:]-axis[:,None]+t-1).flatten(),2*t-1).to(p)
    pd=joint.flatten(-2)@proj
    pi=independent.flatten(-2)@proj
    kl=(pd*(pd.clamp_min(1e-9).log()-pi.clamp_min(1e-9).log())).sum(-1).clamp_min(0)
    role=d['role']
    obj=pool(d['x'],role[...,0])
    obs=pool(d['f1']-d['f0'],role[...,0])
    own=pool(d['x'][...,:3],role[...,1])-obj[:,:3]
    ctx=torch.cat([obj,d['goal']-obj,obs,own],-1)
    return dict(m=means,c=ctx,joint=vector(joint,False),ind=vector(independent,True),kl=kl.reshape(b,k,k-1,1))

class Selector(nn.Module):
    def __init__(self):
        super().__init__()
        self.msg=mlp(2*MEAN+SHAPE+1,32,64)
        self.score=mlp(MEAN+30+64,1,64)
        self.register_buffer('center',torch.zeros(MEAN))
        self.register_buffer('scale',torch.ones(MEAN))
        self.register_buffer('kl_scale',torch.ones(()))

    def forward(self,d,mode):
        m=(d['m']-self.center)/self.scale
        b,k,_=m.shape
        i,j=ordered(k,m.device)
        l=m[:,i].reshape(b,k,k-1,-1)
        diff=(m[:,i]-m[:,j]).reshape_as(l)
        if mode=='means':
            shape=torch.zeros_like(d['joint'])
        else:
            shape=d['ind'] if mode=='independent' else d['joint']
        kl=d['kl']/self.kl_scale if mode=='kl' else torch.zeros_like(d['kl'])
        msg=self.msg(torch.cat([l,diff,shape,kl],-1))
        z=torch.cat([m,d['c'][:,None].expand(-1,k,-1),msg.mean(2),msg.amax(2)],-1)
        return torch.logit(d['m'][...,0].clamp(.0001,.9999))+self.score(z)[...,0]

def loss(logits,y):
    s=y[...,0]
    i,j=pairs(s.size(1),s.device)
    delta=s[:,i]-s[:,j]
    rank=F.softplus(-(logits[:,i]-logits[:,j])*delta)
    directional=(rank*delta.abs()).sum()/delta.abs().sum().clamp_min(1)
    payoff=s+.1*(y[...,1]+y[...,2])/2
    value=(logits.softmax(1)*(payoff-payoff.mean(1,keepdim=True))).sum(1).mean()
    return F.binary_cross_entropy_with_logits(logits,s)+.4*directional-.35*value

@torch.no_grad()
def panel(net,dom,early,split,device):
    d=data(dom,'early' if early else 'late',split,device)
    parts=[features(net,take(d,ix),early) for ix in torch.arange(len(d['x']),device=device).split(48)]
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]},d['outcome']

def selector(early,device='cpu'):
    s=Selector().to(device)
    s.load_state_dict(torch.load(OUT/('early-selector.pt' if early else 'late-selector.pt'),map_location=device)['state'])
    return s.eval()

def train(early):
    torch.manual_seed(271)
    torch.set_num_threads(2)
    dev='cuda'
    phase='early' if early else 'late'
    world=restore('joint',early,dev)
    train=[panel(world,d,early,'policy',dev) for d in DOMAINS]
    val=[panel(world,d,early,'val',dev) for d in DOMAINS]
    net=Selector().to(dev)
    m=torch.cat([x['m'].flatten(0,1) for x,y in train])
    net.center.copy_(m.mean(0))
    net.scale.copy_(m.std(0).clamp_min(.05))
    net.kl_scale.copy_(torch.cat([x['kl'].flatten() for x,y in train]).mean().clamp_min(.015))
    opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.015)
    steps=960
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00008)
    best=float('inf')
    log=[]
    for step in range(1,steps+1):
        samples=[torch.randint(len(y),(48,),device=dev) for x,y in train]
        x={k:torch.cat([v[k][idx] for (v,_),idx in zip(train,samples)]) for k in train[0][0]}
        y=torch.cat([v[idx] for (_,v),idx in zip(train,samples)])
        # Every mask sees the very same observed outcomes at every update.
        obj=sum(loss(net(x,mode),y) for mode in MODES)/len(MODES)
        opt.zero_grad()
        obj.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),3)
        opt.step()
        sched.step()
        if step%60==0:
            with torch.no_grad():
                v=sum(loss(net(x,mode),y).item() for x,y in val for mode in MODES)/(2*len(MODES))
            log.append([step,v])
            if v<best:
                best=v
                torch.save(dict(step=step,state=net.state_dict()),OUT/f'{phase}-selector.pt')
    net=selector(early,dev)
    ranks={}
    for mode in ('direct',*MODES):
        score=[]
        for x,y in val:
            logits=x['m'][...,0] if mode=='direct' else net(x,mode)
            choice=logits.argmax(1)
            chosen=y[torch.arange(len(y),device=dev),choice]
            score.append(dict(n=len(y),success=int(chosen[:,0].sum()),value=float((chosen[:,0]+.05*(chosen[:,1]+chosen[:,2])).mean())))
        ranks[mode]=score
    bestmode=max(ranks,key=lambda k:(sum(v['success']/v['n'] for v in ranks[k]),sum(v['value'] for v in ranks[k])))
    (OUT/f'{phase}-selection.json').write_text(json.dumps(dict(mode=bestmode,validation=ranks,fit=log),indent=1))
    print(phase,bestmode,ranks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--early',action='store_true')
    train(p.parse_args().early)
