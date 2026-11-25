import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from .model import Mechanism,labeled
from .baselines import CoPhy,PETS,NCII

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/mechanism'
OUT = ROOT/'results/mechanism'
DOMAINS = ['cophy','causalworld','pusht']
NAMES = ['mechanism','cophynet','pets','ncii','ncii_contact']
SEED, EPOCHS, BATCH, LR = 271,60,64,.0015


def read(split,device,domains=DOMAINS,fine=False,proposal=False):
    rows=[]
    for name in domains:
        archive=DATA/'fine' if fine and name!='cophy' else DATA
        if proposal and name!='cophy':
            archive=DATA/'demonstrations'
        with np.load(archive/f'{name}-{split}.npz') as f:
            d={k:f[k] for k in f}
        d['reward_mask']=np.full(len(d['x']),name!='cophy')
        d['same']=np.full(len(d['x']),name=='cophy')
        rows.append(d)
    return {k:torch.as_tensor(np.concatenate([r[k] for r in rows]),device=device).float() for k in rows[0]}


def build(name,device):
    if name=='pets':
        return PETS(device).to(device)
    return {'mechanism':Mechanism,'cophynet':CoPhy,'ncii':NCII,'ncii_contact':NCII}[name]().to(device)


def sub(d,idx):
    return {k:v[idx] for k,v in d.items()}


def hindsight(d,mask,ref):
    d=dict(d)
    row=torch.arange(len(ref),device=ref.device)
    for key,new in [('f0',d['x']),('fa',d['a'][row,ref]),('f1',d['y'][row,ref]),('fe',d['e'][row,ref])]:
        d[key]=torch.where(mask.reshape(-1,*([1]*(new.ndim-1))),new,d[key])
    d['same']=torch.maximum(d['same'],mask.float())
    return d


def avg(z,weight):
    w=torch.broadcast_to(weight,z.shape)
    return (z*w).sum()/w.sum().clamp_min(1)


def goal_training(d,augment):
    # Reuse achieved object poses as additional goals; no simulator reward is relabeled.
    d=dict(d)
    movable=d['role'][:,None,:,0:1]
    achieved=(d['y']*movable).sum(-2)/movable.sum(-2).clamp_min(1)
    b,k,_=achieved.shape
    synth=torch.zeros(b,device=achieved.device,dtype=torch.bool)
    if augment:
        idx=torch.arange(b,device=achieved.device)
        synth=(torch.rand(b,device=achieved.device)<.4)&d['reward_mask'].bool()
        own=achieved[idx,torch.randint(k,(b,),device=idx.device)]
        replacement=torch.where((torch.rand(b,1,device=idx.device)<.5),own,own.roll(1,0))
        d['goal']=torch.where(synth[:,None],replacement,d['goal'])
    gap=achieved-d['goal'][:,None]
    distance=(gap[...,:3].square().sum(-1)+.25*gap[...,3:].square().sum(-1)).sqrt()
    d['r']=d['r'].clone()
    d['r'][:,:,2]=-torch.log1p(distance)
    d['reward_valid']=torch.stack([~synth,~synth,torch.ones_like(synth)],-1)[:,None,:]
    return d


def cost(name,m,d,training,output=None):
    out=m(d) if output is None else output
    target=d['y']
    movable=d['role'][:,None,:,0:1]
    observed=d['role'].sum(-1)[:,None,:,None]
    which=d['reward_mask'][:,None,None]
    rw=which*d.get('reward_valid',1)
    edge_mask=labeled(d['role'])[:,None]
    def bce(o,t,w):
        return avg(F.binary_cross_entropy_with_logits(o,t,reduction='none'),w)
    if name.startswith('ncii'):
        contact=bce(out['edge'],d['e'],edge_mask)
        if not training:
            return contact
        return out['loss']+(contact if name=='ncii_contact' else 0.)
    if name=='pets' and training:
        obs=out['ensemble']
        outcome=torch.cat([target-d['x'][:,None],
                           d['r'][:,:,None].expand(-1,-1,target.size(2),-1)],-1)
        loss=.5*((outcome-obs).square()*(-out['lv']).exp()+out['lv'])
        # Independent online Poisson bootstrap per ensemble member and layout.
        boot=torch.poisson(torch.ones_like(obs[:,:,:1,:1,:1]))
        return (avg(loss[...,:9],boot*(observed+movable))+
                3*avg(loss[...,9:],boot*movable*rw.unsqueeze(-2)))
    state=avg(F.smooth_l1_loss(out['y'],target,reduction='none',beta=.18),observed+3*movable)
    rerr=out['r']-d['r']
    importance_index=1 if 'reward_valid' in d else 2
    importance=1+3*d['r'][...,importance_index:importance_index+1]
    reward=avg(F.smooth_l1_loss(rerr,torch.zeros_like(rerr),reduction='none',beta=.09),
               rw*importance)
    if name=='cophynet':
        stable=((target-d['x'][:,None]).norm(dim=-1)<.05).float()
        return state+2*reward+.12*bce(out['stable'],stable,observed[...,0])
    if name=='pets':
        return state+2*reward
    i,j=torch.triu_indices(target.size(1),target.size(1),1,device=target.device)
    (dm,ds),(rm,rs)=m.contrast(out,i,j)
    def effect(mean,sd,truth,mask,floor):
        residual=mean-truth
        sigma=sd+floor
        # Laplace-like robust fitting term and Gaussian proper score with a resolution floor.
        return avg(residual.abs(),mask)+.035*avg(sigma.log()+.5*(residual/sigma).square(),mask)
    state_effect=effect(dm,ds,target[:,i]-target[:,j],movable,.12)
    reward_effect=effect(rm,rs,d['r'][:,i]-d['r'][:,j],rw,.055)
    edge=bce(out['edge'],d['e'],edge_mask)
    factual=bce(out['fact'],d['fe'],edge_mask[:,0])
    return state+2*reward+.65*state_effect+1.5*reward_effect+.32*edge+.16*factual+.002*out['kl']+.002*out['sparse']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('name',choices=NAMES)
    p.add_argument('--device',default='cpu')
    p.add_argument('--fine',action='store_true')
    p.add_argument('--proposal',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(SEED)
    device=args.device
    train,val=read('train',device,fine=args.fine,proposal=args.proposal),read('val',device,fine=args.fine,proposal=args.proposal)
    outdir=OUT/'fine' if args.fine else OUT
    if args.proposal:
        outdir=OUT/'demonstrations'
    model=build(args.name,device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.00015)
    outdir.mkdir(parents=True,exist_ok=True)
    history=[]
    best=float('inf')
    for epoch in range(EPOCHS):
        model.train()
        for ix in torch.randperm(len(train['x']),device=device).split(BATCH):
            d=sub(train,ix)
            ref=torch.randint(4,(len(ix),),device=device)
            d=hindsight(d,torch.rand(len(ix),device=device)<.35,ref)
            if args.fine or args.proposal:
                d=goal_training(d,True)
            loss=cost(args.name,model,d,True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
        sched.step()
        if epoch%3==2 or epoch==0:
            model.eval()
            with torch.no_grad():
                vals=[]
                for ix in torch.arange(len(val['x']),device=device).split(BATCH):
                    d=sub(val,ix)
                    if args.fine or args.proposal:
                        d=goal_training(d,False)
                    v=cost(args.name,model,d,False).item()
                    vals.append(v*len(ix))
                score=sum(vals)/len(val['x'])
            history.append([epoch+1,score])
            if score<best:
                best=score
                torch.save(model.state_dict(),outdir/f'{args.name}.pt')
            print(args.name,epoch+1,round(score,4),flush=True)
    (outdir/f'{args.name}-fit.json').write_text(json.dumps(dict(seed=SEED,epochs=EPOCHS,
               batch=BATCH,lr=LR,validation=history,best=min(history,key=lambda r:r[1]),
               parameters=sum(p.numel() for p in model.parameters())),indent=2))


if __name__=='__main__':
    main()
