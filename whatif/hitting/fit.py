"""Matched-budget marginal supervision versus explicit same-unit distribution supervision."""
import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
from whatif.mechanism.model import labeled
from whatif.mechanism.train import avg,goal_training,sub
from whatif.paired.fit import current
from whatif.recourse.physical import draw
from .native import OUT,DOMAINS
from .data import data,take
from .model import World
from .physical import sources,objective as physics

def pairs(k,device):
    return tuple(torch.triu_indices(k,k,1,device=device))

def response(head,o,d,paired):
    i,j=pairs(d['a'].size(1),o['p'].device)
    y=d['value']
    target=d['bin']
    time=F.nll_loss(o['p'].clamp_min(1e-8).log().flatten(0,1),target.flatten())
    scalar=F.mse_loss(o['score'],d['outcome'][...,1:])
    mu=o['mu']
    sd=(o['loading'].square().sum(-1)+o['tau'].square()).sqrt()
    norm=lambda err,width:(.5*(err/width).square()+width.log()).mean()
    err=mu-y
    diff_err=err[:,i]-err[:,j]
    base=(time+2*scalar+F.smooth_l1_loss(mu,y,beta=.15)+.55*diff_err.abs().mean()+.16*norm(err,sd))
    if not paired:
        return base
    law=head.coupling(o,i,j,detached_marginals=True)
    def pick(prob):
        linear=target[:,i]*prob.size(-1)+target[:,j]
        return prob.flatten(-2).gather(-1,linear[...,None]).squeeze(-1).clamp_min(1e-9)
    independent=o['p'][:,i,:,None]*o['p'][:,j,None,:]
    dependence=(pick(independent).detach().log()-pick(law).log()).mean()
    _,delta_sd=head.normal(o,i,j)
    return base+.8*dependence+.13*norm(diff_err,delta_sd)

def restore(kind,early=False,device='cpu'):
    net=World().to(device)
    state=torch.load(OUT/f'{kind}-world.pt',map_location=device)
    net.load_state_dict(state['state'])
    if early:
        from .path import ProgramGraph
        extra=torch.load(OUT/f'{kind}-early.pt',map_location=device)
        net.head.load_state_dict(extra['state'])
        net.program_graph=ProgramGraph(net.bank).to(device)
        net.program_graph.load_state_dict(extra['path'])
    return net.eval()

def validation(net,ds,early):
    result=[]
    with torch.no_grad():
        for d in ds:
            values=[]
            for ix in torch.arange(len(d['x']),device=d['x'].device).split(64):
                row=take(d,ix)
                o,_=net(row,early)
                values.append(len(ix)*response(net.head,o,row,False).item())
            result.append(sum(values)/len(d['x']))
    return result

def train(kind,device):
    torch.manual_seed(271)
    torch.set_num_threads(2)
    train=[data(dom,'late','world',device) for dom in DOMAINS]
    val=[data(dom,'late','val',device) for dom in DOMAINS]
    net=World().to(device)
    net.core.load_state_dict(torch.load(OUT/'initial.pt',map_location=device))
    phys=sources(device)
    gv=goal_training(current('cophy','val',device),False)
    groups=[[],[]]
    for name,p in net.named_parameters():
        core=name.startswith('core.') and not name.startswith('core.control_token')
        groups[int(core)].append(p)
    opt=torch.optim.AdamW([dict(params=groups[0],lr=.0011),dict(params=groups[1],lr=.00015)])
    steps=2240
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.000055)
    log=[]
    best=float('inf')
    for step in range(1,steps+1):
        net.train()
        ds=[take(d,torch.randint(len(d['x']),(32,),device=device)) for d in train]
        row={k:torch.cat([d[k] for d in ds]) for k in ds[0]}
        o,graph=net(row)
        mask=labeled(row['role'])[:,None]
        ce=avg(F.binary_cross_entropy_with_logits(graph['edge'],row['edge'],reduction='none'),mask)
        obj=(response(net.head,o,row,kind=='joint')+.25*ce+.002*graph['kl']
             +.75*physics(net.core,draw(phys,4))+.18*net.core.reconstruction(row))
        opt.zero_grad()
        obj.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),4.)
        opt.step()
        sched.step()
        if step%160==0:
            net.eval()
            scores=validation(net,val,False)
            with torch.no_grad():
                tower=sum(len(ix)*physics(net.core,sub(gv,ix)).item()
                          for ix in torch.arange(len(gv['x']),device=device).split(40))/len(gv['x'])
            value=np.mean(scores)+.45*tower
            scores.append(tower)
            log.append(dict(step=step,validation=scores,criterion=value))
            if value<best:
                best=value
                torch.save(dict(step=step,state=net.state_dict()),OUT/f'{kind}-world.pt')
            print(kind,'late',log[-1],flush=True)
            (OUT/f'{kind}-world-fit.json').write_text(json.dumps(log,indent=1))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['marginal','joint'])
    p.add_argument('--early',action='store_true')
    args=p.parse_args()
    if args.early:
        from .early import train as early_train
        early_train(args.kind,'cuda')
    else:
        train(args.kind,'cuda')
