"""PointNet reproduction of NCII §4.1 and App. D on native object states."""
import argparse
import json
import time

import torch
from torch import nn
from torch.nn import functional as F

from .collect import DOMAINS, OUT, PROTOCOL
from .data import clock, load
from .model import average

HIDDEN = 512


def point_mlp(in_size, out_size):
    return nn.Sequential(nn.Linear(in_size,HIDDEN), nn.LeakyReLU(),
                         nn.Linear(HIDDEN,HIDDEN), nn.LeakyReLU(),
                         nn.Linear(HIDDEN,HIDDEN), nn.LeakyReLU(),
                         nn.Linear(HIDDEN,out_size))


class PointEmbedding(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.object_embedding=nn.Linear(9+9+3,HIDDEN)
        self.action_embedding=nn.Linear(n*7+2,HIDDEN)
        self.source=nn.Linear(HIDDEN,HIDDEN)
        self.destination=nn.Linear(HIDDEN,HIDDEN)

    def forward(self,x,evidence,role,action,time):
        own=F.leaky_relu(self.object_embedding(torch.cat([x,evidence,role],-1)))
        act=F.leaky_relu(self.action_embedding(torch.cat([action.flatten(1),time],-1)))[:,None]
        src=self.source(torch.cat([own,act],1))
        dst=self.destination(own)
        return own,F.leaky_relu(src[:,:,None]+dst[:,None])


class NCII(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.embed=PointEmbedding(n)
        self.forward_model=point_mlp(2*HIDDEN,18)
        self.interaction_embed=PointEmbedding(n)
        self.interaction=point_mlp(HIDDEN,1)
        self.register_buffer('scale',torch.ones(9))

    def distribution(self,own,peers):
        mu,v=self.forward_model(torch.cat([own,peers],-1)).chunk(2,-1)
        return mu,F.softplus(v)+.001

    def logits(self,*args):
        _,pair=self.interaction_embed(*args)
        return self.interaction(pair)[...,0]


def valid_sources(role):
    return F.pad(role.sum(-1).bool(),(0,1),value=True)


def command_inputs(d):
    k=d['a'].size(1)
    evidence=(d['fact'][:,-1]-d['fact'][:,-2])[:,None].expand(-1,k,-1,-1)
    role=d['role'][:,None].expand(-1,k,-1,-1)
    times=clock(d['widths'])[:,None].expand(-1,k,-1,-1)
    return evidence, role, d['a'], times


def training_examples(d):
    evidence,role,act,t=command_inputs(d)
    old=torch.cat([(d['x'][:,None]+d['reset'])[:,:,None],d['pose'][:,:,:-1]],2)
    def expand(v):
        return v[:,:,None].expand(-1,-1,old.size(2),-1,-1).flatten(0,2)
    return (old.flatten(0,2),expand(evidence),expand(role),expand(act),t.flatten(0,2),
            (d['pose']-old).flatten(0,2))


def nll(mu,variance,target):
    return .5*((target-mu).square()/variance+variance.log()).sum(-1)


def max_peers(pair,mask):
    return pair.masked_fill(~mask[...,None],-1.e4).amax(-3)


def train(domain):
    dev='cuda'
    d=load(domain,'train',dev)
    data=training_examples(d)
    n=d['x'].size(1)
    model=NCII(n).to(dev)
    model.scale.copy_(data[-1][data[2].sum(-1).bool()].std(0).clamp_min(.08))
    of=torch.optim.Adam([*model.embed.parameters(),*model.forward_model.parameters()],lr=1.e-4,fused=True)
    oh=torch.optim.Adam([*model.interaction_embed.parameters(),*model.interaction.parameters()],lr=1.e-4,fused=True)
    idx=torch.arange(n,device=dev)
    self_edges=F.pad(torch.eye(n,device=dev,dtype=torch.bool),(0,0,0,1))
    histories=[]
    start=time.monotonic()
    for step in range(1,PROTOCOL['budgets']['ncii_updates']+1):
        batch=torch.randint(len(data[0]),(64,),device=dev)
        *args,target=(v[batch] for v in data)
        present=args[2].sum(-1)
        valid=valid_sources(args[2])[:,:,None]
        target=target/model.scale
        own,pair=model.embed(*args)
        full=model.distribution(own,max_peers(pair,valid))
        passive=model.distribution(own,pair[:,idx,idx])
        passive_nll=nll(*passive,target)
        objective=average(nll(*full,target)+.5*passive_nll,present)
        with torch.no_grad():
            # Simulated nulls sample absent variables in high passive-likelihood states.
            null_ok=(passive_nll<=passive_nll[present.bool()].median())[:,None]
            drop=torch.rand_like(pair[...,0])<.25
            keep=valid&(~(drop&null_ok)|self_edges)
            if step>=400:
                learned=valid&((model.logits(*args)>=0)|self_edges)
                keep=torch.where(null_ok,keep,learned)
        masked=model.distribution(own,max_peers(pair,keep))
        objective=objective+.5*average(nll(*masked,target),present)
        of.zero_grad(set_to_none=True)
        objective.backward()
        of.step()
        if step>=400:
            with torch.no_grad():
                own,pair=model.embed(*args)
                value,source=pair.masked_fill(~valid[...,None],-1.e4).topk(2,dim=1)
                inds=torch.arange(n+1,device=dev)[None,:,None,None]
                omitted=torch.where(inds==source[:,:1],value[:,1:2],value[:,:1])
                omitted=torch.where(self_edges[...,None],value[:,:1],omitted)
                full=model.distribution(own,value[:,0])
                null=model.distribution(own[:,None].expand(-1,n+1,-1,-1),omitted)
                label=(nll(*null,target[:,None])-nll(*full,target)[:,None]>1).float()
            logits=model.logits(*args)
            bce=F.binary_cross_entropy_with_logits(logits,label,reduction='none')
            h_loss=average(bce,valid*present[:,None]*(~self_edges))
            oh.zero_grad(set_to_none=True)
            h_loss.backward()
            oh.step()
        if step%400==0:
            rec=dict(step=step, loss=float(objective.detach()), interaction_loss=float(h_loss.detach()),
                     seconds=round(time.monotonic()-start,1))
            histories.append(rec)
            (OUT/f'ncii-{domain}-fit.json').write_text(json.dumps(histories,indent=2)+'\n')
            print(domain,'ncii',rec,flush=True)
    torch.save(dict(model=model.state_dict(),n=n),OUT/f'ncii-{domain}.pt')


def restore(domain,device):
    state=torch.load(OUT/f'ncii-{domain}.pt',map_location=device)
    net=NCII(state['n']).to(device)
    net.load_state_dict(state['model'])
    return net.eval()


def predict(model,d):
    b,k,n=d['a'].shape[:3]
    evidence,role,act,t=command_inputs(d)
    evidence,role,act=[v.flatten(0,1) for v in (evidence,role,act)]
    valid=valid_sources(role)[...,None]
    x=(d['x'][:,None]+d['reset']).flatten(0,1)
    logits=[]
    for time in t.flatten(0,1).unbind(1):
        args=(x,evidence,role,act,time)
        logits.append(model.logits(*args)[:,:n].unflatten(0,(b,k)))
        own,pair=model.embed(*args)
        mu,_=model.distribution(own,max_peers(pair,valid))
        x=x+mu*model.scale
    edges=torch.stack(logits,2)
    return dict(edge=edges,causal=edges.cummax(2).values)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('domain',choices=DOMAINS)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(PROTOCOL['seed'])
    train(args.domain)
