"""Fit the eight-program head on the measured downstream policy and observed early contact paths."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from whatif.mechanism.model import labeled
from whatif.mechanism.train import avg
from .fit import restore,response
from .native import OUT,DATA,DOMAINS,BANK
from .data import data,take,load_np
from .path import ProgramGraph

def train(kind,dev):
    torch.manual_seed(271)
    torch.set_num_threads(2)
    net=restore(kind,False,dev)
    def cache(dom,split):
        d=data(dom,'early',split,dev)
        raw=load_np(DATA/f'contact-{dom}-{split}.npz')
        if not np.array_equal(raw['seed'].reshape(-1,len(BANK))[:,0],d['seed'].cpu().numpy()):
            raise ValueError('unmatched contact layouts')
        e=torch.tensor(raw['contact'],device=dev).unflatten(0,(-1,len(BANK)))
        d['path_edge']=torch.cat([e[:,:1],e],1)
        z,p=[],[]
        with torch.no_grad():
            for ix in torch.arange(len(d['x']),device=dev).split(64):
                row=take(d,ix)
                z.append(net.encode(row,True)[0])
                p.append(net.core.abduct(row)[:3])
        d['z']=torch.cat(z)
        d.update({f'factual{k}':torch.cat([v[k] for v in p]) for k in range(3)})
        return d
    train_rows=[cache(dom,'world') for dom in DOMAINS]
    val=[cache(dom,'val') for dom in DOMAINS]
    path=ProgramGraph(net.bank).to(dev)
    opt=torch.optim.AdamW(list(net.head.parameters())+list(path.parameters()),lr=.0009)
    steps=1920
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00006)
    def objective(d,paired):
        z,edge=path(d,d['z'],[d[f'factual{k}'] for k in range(3)])
        mask=labeled(d['role'])[:,None]
        contact=avg(F.binary_cross_entropy_with_logits(edge,d['path_edge'],reduction='none'),mask)
        return response(net.head,net.head(z),d,paired)+.35*contact
    log=[]
    best=float('inf')
    for step in range(1,steps+1):
        ds=[take(d,torch.randint(len(d['x']),(40,),device=dev)) for d in train_rows]
        row={k:torch.cat([d[k] for d in ds]) for k in ds[0]}
        loss=objective(row,kind=='joint')
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.head.parameters())+list(path.parameters()),4.)
        opt.step()
        sched.step()
        if step%160==0:
            with torch.no_grad():
                v=[sum(len(ix)*objective(take(d,ix),False).item() for ix in torch.arange(len(d['x']),device=dev).split(64))/len(d['x']) for d in val]
            criterion=float(np.mean(v))
            log.append(dict(step=step,validation=v,criterion=criterion))
            if criterion<best:
                best=criterion
                torch.save(dict(step=step,state=net.head.state_dict(),path=path.state_dict()),OUT/f'{kind}-early.pt')
            print(kind,'early',log[-1],flush=True)
            (OUT/f'{kind}-early-fit.json').write_text(json.dumps(log,indent=1))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['joint','marginal'])
    train(p.parse_args().kind,'cuda')
