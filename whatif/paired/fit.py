"""Balanced three-benchmark training with direct supervision on untouched objects."""
import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
from whatif.mechanism.train import DATA as OLD,SEED,hindsight,sub,cost,avg,goal_training
from whatif.mechanism.model import labeled
from .data import DATA,OUT
from .world import Transport

DOMAINS=['cophy','causalworld','pusht']
KEYS=['f0','fa','f1','fe','x','role','goal','a','y','e','r','seed']


def load_rows(path,domain,device,excision=None):
    with np.load(path) as z:
        rows={k:torch.as_tensor(z[k],device=device).float() for k in KEYS}
        for k in ['outcome','prior','remaining']:
            if k in z:
                rows[k]=torch.as_tensor(z[k],device=device).float()
    rows['same']=torch.full_like(rows['seed'],float(domain=='cophy'))
    rows['reward_mask']=1-rows['same']
    rows['impact']=torch.zeros_like(rows['e'])
    rows['measured']=torch.zeros_like(rows['same'])
    if excision:
        with np.load(excision) as f:
            if not np.array_equal(rows['seed'].cpu().numpy(),f['seed']):
                raise ValueError('excision seed order')
            rows['impact']=torch.tensor(f['impact'],device=device)
        rows['measured']+=1
    return rows


def current(domain,split,device):
    ex=DATA/f'excision-{split}.npz' if domain=='cophy' and split!='train' else None
    return load_rows(DATA/f'{domain}-{split}.npz',domain,device,ex)


def old_training(domain,device):
    if domain=='cophy':
        return load_rows(OLD/'cophy-train.npz',domain,device,OLD/'directed/excision-train.npz')
    return load_rows(OLD/f'demonstrations/{domain}-train.npz',domain,device)


def four(d,idx):
    d=sub(d,idx)
    cols=torch.randperm(d['a'].size(1),device=idx.device)[:4]
    for k in ['a','y','e','r','impact']:
        d[k]=d[k][:,cols]
    return {k:v for k,v in d.items() if k in KEYS+['same','reward_mask','impact','measured']}


def loss(net,d):
    o=net(d)
    base=cost('mechanism',net,d,net.training,o)
    mask=labeled(d['role'])[:,None]*d['measured'][:,None,None,None]
    causal=avg(F.binary_cross_entropy_with_logits(o['causal'],(d['impact']>.10).float(),reduction='none'),mask)
    strength=avg(F.smooth_l1_loss(o['impact'],d['impact'].log1p(),reduction='none',beta=.2),mask)
    i,j=torch.triu_indices(d['a'].size(1),d['a'].size(1),1,device=d['a'].device)
    (mu,_),_=net.contrast(o,i,j)
    err=mu[...,:3]-(d['y'][:,i]-d['y'][:,j])[...,:3]
    untouched=(d['a'][:,i,:,0]+d['a'][:,j,:,0]==0)*d['role'][:,None,:,0]
    # Explicit conditional-mean target for the position difference, including remote receivers.
    remote=avg(err.square(),untouched[...,None])
    return base+.75*causal+.2*strength+.85*remote


def train(device,contact_only=False):
    tag="contact" if contact_only else "world"
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    data=[(old_training(dom,device),current(dom,'train',device)) for dom in DOMAINS]
    # Validation for checkpointing uses only the one-step and effect labels.
    val=[goal_training(current(dom,'val',device),False) for dom in DOMAINS]
    model=Transport(use_transfer=not contact_only).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0015)
    updates=3200
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,updates,eta_min=.00015)
    OUT.mkdir(parents=True,exist_ok=True)
    best=float('inf')
    history=[]
    for step in range(1,updates+1):
        model.train()
        parts=[]
        for old,new in data:
            for d,n in [(old,8),(new,16)]:
                parts.append(four(d,torch.randint(len(d['x']),(n,),device=device)))
        row={k:torch.cat([p[k] for p in parts]) for k in parts[0]}
        row=hindsight(row,torch.rand(len(row['x']),device=device)<.25,
                      torch.randint(4,(len(row['x']),),device=device))
        obj=loss(model,goal_training(row,True))
        opt.zero_grad()
        obj.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        sched.step()
        if step%160==0:
            model.eval()
            with torch.no_grad():
                scores=[]
                for d in val:
                    values=[loss(model,sub(d,ix)).item()*len(ix)
                            for ix in torch.arange(len(d['x']),device=device).split(48)]
                    scores.append(sum(values)/len(d['x']))
            history.append([step,*scores])
            if np.mean(scores)<best:
                best=np.mean(scores)
                torch.save({k:v.cpu() for k,v in model.state_dict().items()},OUT/f'{tag}.pt')
            print(step,scores,flush=True)
    (OUT/f'{tag}-fit.json').write_text(json.dumps(history,indent=1))


def world(contact_only=False):
    tag='contact' if contact_only else 'world'
    model=Transport(use_transfer=not contact_only)
    model.load_state_dict(torch.load(OUT/f'{tag}.pt',map_location='cpu'))
    return model.eval()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu')
    p.add_argument('--contact-only',action='store_true')
    args=p.parse_args()
    train(args.device,args.contact_only)
