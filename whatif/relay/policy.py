"""Learn the decision from observed sibling success differences, with nested distribution inputs."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import mlp,pool
from whatif.hitting.policy import ordered
from whatif.hitting.data import take
from .model import restore,coupled,difference
from .native import OUT,DOMAINS
from .data import dataset

MODES=('means','independent','law','kl')
POINT=24
SHAPE=11+3+40

def reverse(v):
    b,k,j,c=v.shape
    i,t=ordered(k,v.device)
    reverse_index=(t*(k-1)+i-(i>t).long())
    return v.flatten(1,2)[:,reverse_index].reshape(b,k,j,c)

@torch.no_grad()
def features(world,d):
    o,_=world(d)
    p=o['p']
    b,k,c=p.shape
    i,j=ordered(k,p.device)
    mean=torch.cat([1-p[...,-1:],(p*torch.arange(c,device=p.device)/5).sum(-1,keepdim=True),
                    o['score'],o['mu'][...,[0,1,2,9]].flatten(-2)],-1)
    axis=torch.arange(c,device=p.device)
    proj=F.one_hot((axis[None,:]-axis[:,None]+5).flatten(),11).to(p)
    def encode(ind):
        law=coupled(o,i,j,ind)
        pm=law.flatten(-2)@proj
        minus=law[:,:,-1,:-1].sum(-1)
        plus=law[:,:,:-1,-1].sum(-1)
        tied=(1-minus)-plus
        flip=torch.stack([minus,tied,plus],-1).clamp_min(1e-7)
        v,sd,w=difference(o,i,j,independent=ind)
        v,sd=v[...,[0,1,2,9]],sd[...,[0,1,2,9]]
        w=w[:,None,None,:,None]
        mu=(w*v).sum(3,keepdim=True)
        sigma=(w*(sd.square()+(v-mu).square())).sum(3).clamp_min(1e-7).sqrt()
        sign=(w*torch.special.ndtr(v/sd)).sum(3)
        vec=torch.cat([pm,flip,sigma.log().flatten(-2),sign.flatten(-2)],-1)
        return pm,flip.reshape(b,k,k-1,3),vec.reshape(b,k,k-1,SHAPE)
    pm,base,ind=encode(True)
    pj,joint,dep=encode(False)
    kl=(pj*(pj.clamp_min(1e-8).log()-pm.clamp_min(1e-8).log())).sum(-1).clamp_min(0)
    role=d['role']
    mover=pool(d['x'],role[...,0])
    context=torch.cat([mover,d['goal']-mover,pool(d['f1']-d['f0'],role[...,0]),
                       pool(d['x'][...,:3],role[...,1])-mover[:,:3],d['width'].sum(-1,keepdim=True)/288],-1)
    return dict(m=mean,ctx=context,ind=ind,dep=dep,base=base,joint=joint,kl=kl.reshape(b,k,k-1,1))

class Decision(nn.Module):
    def __init__(self):
        super().__init__()
        self.pair=mlp(2*POINT+31+SHAPE+1,3,80)
        self.register_buffer('center',torch.zeros(POINT))
        self.register_buffer('scale',torch.ones(POINT))
        self.register_buffer('kl_scale',torch.ones(()))

    def forward(self,d,mode):
        m=(d['m']-self.center)/self.scale
        b,k,_=m.shape
        i,j=ordered(k,m.device)
        shape=(torch.zeros_like(d['ind']) if mode=='means' else d['ind'] if mode=='independent' else d['dep'])
        prior=d['base'] if mode in ('means','independent') else d['joint']
        info=d['kl']/self.kl_scale if mode=='kl' else torch.zeros_like(d['kl'])
        z=torch.cat([m[:,i],m[:,i]-m[:,j]],-1).reshape(b,k,k-1,-1)
        context=d['ctx'][:,None,None].expand(-1,k,k-1,-1)
        v=self.pair(torch.cat([z,context,shape,info],-1))
        logp=(prior.log()+.5*(v+reverse(v).flip((-1,)))).log_softmax(-1)
        q=logp.exp()
        score=(q[...,2]-q[...,0]).mean(-1)
        return score,logp

def loss(net,d,mode):
    score,logp=net(d,mode)
    s=d['s']
    i,j=ordered(s.size(1),s.device)
    label=(s[:,i]-s[:,j]+1).long().reshape(*logp.shape[:-1],1)
    nll=-logp.gather(-1,label).mean()
    # Conditional win on discordant outcomes has a direct interpretation as pair preference.
    gap=s[:,i]-s[:,j]
    logodds=(logp[...,2]-logp[...,0]).flatten(1)
    rank=(F.softplus(-logodds*gap)*gap.abs()).sum()/gap.abs().sum().clamp_min(1)
    gain=((5*score).softmax(1)*(s-s.mean(1,keepdim=True))).sum(1).mean()
    return nll+.3*rank-.3*gain

def panel(net,dom,phase,split,dev):
    d=dataset(dom,phase,split,dev)
    parts=[features(net,take(d,ix)) for ix in torch.arange(len(d['x']),device=dev).split(24)]
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}|dict(s=d['outcome'][...,0])

def load(dev):
    model=Decision().to(dev)
    model.load_state_dict(torch.load(OUT/'selector.pt',map_location=dev)['model'])
    return model.eval()

def train():
    dev='cuda'
    torch.manual_seed(271)
    torch.set_num_threads(2)
    net=restore('dynamic',dev)
    tr,val=[[panel(net,dom,p,s,dev) for dom in DOMAINS for p in ('early','late')] for s in ('policy','val')]
    rank=Decision().to(dev)
    x=torch.cat([d['m'].flatten(0,1) for d in tr])
    rank.center.copy_(x.mean(0))
    rank.scale.copy_(x.std(0).clamp_min(.05))
    rank.kl_scale.copy_(torch.cat([d['kl'].flatten() for d in tr]).mean().clamp_min(.005))
    steps=880
    opt=torch.optim.AdamW(rank.parameters(),lr=.0007,weight_decay=.012)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00007)
    best=float('inf')
    hist=[]
    for step in range(1,steps+1):
        rows=[take(d,torch.randint(len(d['s']),(32,),device=dev)) for d in tr]
        d={k:torch.cat([s[k] for s in rows]) for k in rows[0]}
        obj=sum(loss(rank,d,mode) for mode in MODES)/len(MODES)
        opt.zero_grad()
        obj.backward()
        torch.nn.utils.clip_grad_norm_(rank.parameters(),3)
        opt.step()
        sched.step()
        if step%40==0:
            with torch.no_grad():
                v=sum(loss(rank,d,m).item() for d in val for m in MODES)/(len(val)*len(MODES))
            hist.append([step,v])
            if v<best:
                best=v
                torch.save(dict(step=step,model=rank.state_dict()),OUT/'selector.pt')
    (OUT/'selector-fit.json').write_text(json.dumps(hist))
    print('selector',min(hist,key=lambda v:v[1]),flush=True)
if __name__=='__main__':
    train()
