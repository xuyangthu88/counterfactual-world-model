"""Calibrate the endpoint of one physical intervention using all five propagated factor states."""
import argparse
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from whatif.hitting.data import take
from whatif.hitting.physical import sources
from whatif.mechanism.model import mlp,pool
from whatif.mechanism.train import avg,goal_training
from .data import dataset
from .model import restore
from .native import OUT,DOMAINS

class Endpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.node=mlp(120+9+9+3+9+16,72,128)
        self.delta=mlp(72,9,72)
        self.noise=mlp(72,18,64)
        self.reward=mlp(72+18,9,80)
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self,z,base,d):
        h=self.node(z)
        pred=base+self.delta(h)
        y=pred[:,:-1]+d['same'][:,None,None,None]*(d['f1'][:,None]-pred[:,-1:])
        move=d['role'][:,None,:,0]
        goal=d['goal'][:,None].expand(-1,y.size(1),-1)
        ctx=torch.cat([pool(h[:,:-1],move),pool(y,move)-goal,goal],-1)
        v=self.reward(ctx).unflatten(-1,(3,3))
        dual=self.noise(h[:,:-1]).unflatten(-1,(9,2))
        def noise(s,l):
            return torch.stack([.01+F.softplus(s),l],-1)
        return dict(y=y,r=v[...,0],aleatoric=noise(dual[...,0],dual[...,1]),reward_noise=noise(v[...,1],v[...,2]))

@torch.no_grad()
def pack(core,d):
    d=goal_training(d,False)
    parts=[]
    for ix in torch.arange(len(d['x']),device=d['x'].device).split(32):
        row=take(d,ix)
        for key in ('hx','ha','hy'):
            row.pop(key,None)
        out=core(dict(row,capture=True))
        parts.append((out['packed'],out['basepose']))
    z,base=map(torch.cat,zip(*parts))
    return {k:d[k] for k in ('role','goal','f1','same','y','r','reward_mask')}|dict(z=z,base=base)

def sample(ds,n):
    chunks=[]
    for group in ds:
        src=group[int(torch.randint(len(group),()).item())]
        idx=torch.randint(len(src['z']),(n,),device=src['z'].device)
        row=take(src,idx)
        k=row['y'].size(1)
        ids=torch.randperm(k,device=idx.device)[:4]
        for key in ('y','r'):
            row[key]=row[key][:,ids]
        labels=torch.cat([ids,ids.new_tensor([k])])
        for key in ('z','base'):
            row[key]=row[key][:,labels]
        chunks.append(row)
    return {k:torch.cat([v[k] for v in chunks]) for k in chunks[0]}

def loss(net,d):
    o=net(d['z'],d['base'],d)
    i,j=torch.triu_indices(o['y'].size(1),o['y'].size(1),1,device=o['y'].device)
    def term(mean,truth,params,mask,floor):
        err=mean-truth
        de=err[:,i]-err[:,j]
        tau,load=params.unbind(-1)
        sd=(tau[:,i].square()+tau[:,j].square()+(load[:,i]-load[:,j]).square()).sqrt()+floor
        nll=.5*(de.detach()/sd).square()+sd.log()
        return avg(err.square()+.2*err.abs(),mask)+.8*avg(de.square(),mask)+.085*avg(nll,mask)
    movable=d['role'][:,None,:,0:1]
    obj=movable+.12*d['role'][:,None].sum(-1,keepdim=True)
    return term(o['y'],d['y'],o['aleatoric'],obj,.12)+2*term(o['r'],d['r'],o['reward_noise'],d['reward_mask'][:,None,None],.055)

def train(kind):
    torch.manual_seed(271)
    torch.set_num_threads(2)
    dev='cuda'
    model=restore(kind,dev)
    old=sources(dev) # new Stage 25 towers, Stage 24 towers, original towers, then three Stage 22 domains
    current=[dataset(d,'short','train',dev) for d in ('cophy',*DOMAINS)]
    groups=[[current[0],*old[:4]],[current[1],old[5]],[current[2],old[4]]]
    tr=[[pack(model.core,d) for d in g] for g in groups]
    val=[pack(model.core,dataset(dom,'short','val',dev)) for dom in ('cophy',*DOMAINS)]
    head=Endpoint().to(dev)
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    steps=1800
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00007)
    hist=[]
    best=float('inf')
    for step in range(1,steps+1):
        obj=loss(head,sample(tr,32))
        opt.zero_grad()
        obj.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),4)
        opt.step()
        sch.step()
        if step%150==0:
            with torch.no_grad():
                v=[sum(len(i)*loss(head,take(d,i)).item() for i in torch.arange(len(d['z']),device=dev).split(48))/len(d['z']) for d in val]
            record=dict(step=step,validation=v)
            hist.append(record)
            if np.mean(v)<best:
                best=np.mean(v)
                model.core.atomic_readout=head
                torch.save(dict(step=step,model=model.state_dict()),OUT/f'{kind}-full.pt')
            print(kind,step,round(float(np.mean(v)),5),flush=True)
    (OUT/f'{kind}-readout.json').write_text(json.dumps(hist))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['dynamic','fixed'])
    train(p.parse_args().kind)
