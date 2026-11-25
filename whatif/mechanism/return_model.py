"""Learn the coupled distribution of six-macro native return differences."""
import argparse
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import mlp,pool
from .train import OUT,DATA,build,SEED
from .planner import KL_WEIGHT
from .continuations import HORIZON

DEST=OUT/'continuations'
BINS=31
RANGE=1.75
MIN_GAIN,HARM_LIMIT,HARM_DROP=.04,.20,.10


def backbone(name):
    net=build('pets' if name=='pets' else 'mechanism','cpu')
    net.load_state_dict(torch.load(OUT/'demonstrations'/('pets.pt' if name=='pets' else 'mechanism.pt'),map_location='cpu'))
    return net.eval()


def features(net,d):
    o=net(d)
    moving=d['role'][:,None,:,0]
    now=pool(d['x'],moving[:,0])[:,None]
    future=pool(o['y'],moving)
    g=d['goal'][:,None].expand_as(future)
    context=[future-g,future-now,now.expand_as(future)-g,g,o['r']]
    if 'h' in o:
        context.append(pool(o['h'],moving))
    else:
        members=pool(o['ensemble'][...,:9],moving)
        context.append(members.permute(1,2,0,3).flatten(-2))
    return torch.cat(context,-1)


class ReturnHead(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.register_buffer('center',torch.zeros(dim))
        self.register_buffer('spread',torch.ones(dim))
        self.register_buffer('bins',torch.linspace(-RANGE,RANGE,BINS))
        self.encode=mlp(dim,64,80)
        self.value=mlp(64,1,64)
        self.compare=mlp(192,BINS,80)

    def forward(self,x,i,j):
        h=self.encode((x-self.center)/self.spread)
        a,b=h[:,i],h[:,j]
        def logits(left,right):
            return self.compare(torch.cat([left,right,left-right],-1))
        lp=.5*(logits(a,b)+logits(b,a).flip(-1))
        return self.value(h).squeeze(-1),lp.log_softmax(-1)


def outcomes(data):
    s,peak,last=data.unbind(-1)
    return s+.5*peak+.25*last


def loss(head,x,g):
    i,j=torch.triu_indices(x.size(1),x.size(1),1)
    value,logp=head(x,i,j)
    truth=g[:,i]-g[:,j]
    distance=(truth[...,None]-head.bins).abs()/(2*RANGE/(BINS-1))
    label=(1-distance).clamp_min(0)
    ce=-(label*logp).sum(-1).mean()
    mu=(logp.exp()*head.bins).sum(-1)
    return F.smooth_l1_loss(value,g,beta=.12)+.8*ce+.35*F.smooth_l1_loss(mu,truth,beta=.12)


@torch.no_grad()
def panel(net,split):
    chunks=[dict(np.load(DATA/f'continuations/{d}-{split}.npz')) for d in ('causalworld','pusht')]
    d={k:torch.from_numpy(np.concatenate([c[k] for c in chunks])).float() for k in chunks[0]}
    xs=[]
    for ix in torch.arange(len(d['x'])).split(96):
        xs.append(features(net,{k:v[ix] for k,v in d.items()}))
    return torch.cat(xs),outcomes(d['outcome']),d['prior']


def load_head(name):
    ckpt=torch.load(DEST/f'{name}.pt',map_location='cpu')
    net=ReturnHead(len(ckpt['center']))
    net.load_state_dict(ckpt)
    return net.eval()


def score(head,x,prior,policy):
    k=x.size(1)
    indices=torch.arange(k)
    values,logp=head(x,indices,torch.zeros_like(indices))
    p=logp.exp()
    preference=.075*prior.clamp(min=-6.)
    if policy=='value':
        scores=values-values[:,:1]+preference
    else:
        mu=(p*head.bins).sum(-1)
        harm=p[:,:,head.bins < -HARM_DROP].sum(-1)
        middle=BINS//2
        benefit=(p[:,:,middle+1:].sum(-1)+.5*p[:,:,middle]).clamp(1e-6,1-1e-6)
        scores=mu+preference
        if policy in ('delta_kl','pets'):
            divergence=np.log(2.)+benefit*benefit.log()+(1-benefit)*(1-benefit).log()
            scores=scores+KL_WEIGHT*torch.sign(benefit-.5)*divergence
        scores=scores.masked_fill(harm>HARM_LIMIT,-torch.inf)
    scores=scores-MIN_GAIN
    scores[:,0]=0
    return scores.argmax(-1)


def fit(name):
    torch.manual_seed(SEED)
    torch.set_num_threads(3)
    net=backbone(name)
    x,g,_=panel(net,'train')
    vx,vg,_=panel(net,'val')
    model=ReturnHead(x.size(-1))
    model.center.copy_(x.mean((0,1)))
    model.spread.copy_(x.flatten(0,1).std(0).clamp_min(.05))
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,90,eta_min=.0001)
    best=float('inf')
    history=[]
    DEST.mkdir(parents=True,exist_ok=True)
    for epoch in range(90):
        for ix in torch.randperm(len(x)).split(64):
            obj=loss(model,x[ix],g[ix])
            opt.zero_grad()
            obj.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
        schedule.step()
        if epoch%3==2:
            with torch.no_grad():
                l=loss(model,vx,vg).item()
            history.append([epoch+1,l])
            if l<best:
                best=l
                torch.save(model.state_dict(),DEST/f'{name}.pt')
    (DEST/f'{name}-fit.json').write_text(json.dumps(history))
    print(name,min(history,key=lambda p:p[1]),flush=True)


@torch.no_grad()
def evaluate():
    table={}
    for name in ['graph','pets']:
        x,g,prior=panel(backbone(name),'test')
        head=load_head(name)
        for dom,idx in zip(['causalworld','pusht'],torch.arange(len(x)).chunk(2)):
            xx,gg,pp=x[idx],g[idx],prior[idx]
            i,j=torch.triu_indices(4,4,1)
            _,lp=head(xx,i,j)
            prob=lp.exp()
            mu=(prob*head.bins).sum(-1)
            truth=gg[:,i]-gg[:,j]
            expected_abs=(prob*(head.bins-truth[...,None]).abs()).sum(-1)
            # CRPS of a finite discrete distribution via its sorted support.
            cumulative=prob.cumsum(-1)
            dispersion=(prob*head.bins*(2*cumulative-prob-1)).sum(-1)
            r=dict(effect_rmse=float((mu-truth).square().mean().sqrt()),crps=float((expected_abs-dispersion).mean()))
            for p in ['value','delta','delta_kl']:
                choice=score(head,xx,pp,p)
                selected=gg.gather(1,choice[:,None]).squeeze(1)
                r[p]=dict(mean_gain=float((selected-gg[:,0]).mean()),switch_rate=float((choice!=0).float().mean()),
                          mean_regret=float((gg.max(-1).values-selected).mean()))
            table.setdefault(dom,{})[name]=r
    (DEST/'offline.json').write_text(json.dumps(table,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['graph','pets','evaluate'])
    args=p.parse_args()
    torch.set_num_threads(3)
    evaluate() if args.mode=='evaluate' else fit(args.mode)
