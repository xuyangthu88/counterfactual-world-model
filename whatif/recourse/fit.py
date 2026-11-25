"""Train one transport network with physical excisions and measured contingent futures."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans
from whatif.mechanism.model import pool,labeled
from whatif.mechanism.train import avg,goal_training,sub
from whatif.paired.fit import loss as physical_loss,current
from .native import DOMAINS,OUT
from .model import World,MODES,ATOMS
from .load import read,take,early,late,observation
from .physical import sources,draw

def indices(k,dev):
    return tuple(torch.triu_indices(k,k,1,device=dev))

def terminal(net,z,start,d):
    scalar=net.result(z)
    estimate=start[:,None]+net.motion(z)
    tar=pool(d['final'],d['role'][:,None,:,0])
    value=torch.cat([estimate[...,:3],scalar[...,3:]],-1)
    truth=torch.cat([tar[...,:3],d['native'][...,None]],-1)
    i,j=indices(z.size(1),z.device)
    mu,sd=net.effect(z,value,i,j)
    diff=truth[:,i]-truth[:,j]
    y=d['outcome']
    s=y[...,0]
    aux=.5*y[...,1]+.25*y[...,2]
    p=net.distribution(z[:,i],z[:,j])
    event=(1+s[:,i]-s[:,j]).long()
    loc=((aux[:,i]-aux[:,j]+.75)*(ATOMS-1)/1.5).clamp(0,ATOMS-1)
    lo=loc.floor().long()
    subp=p.gather(-2,event[...,None,None].expand(-1,-1,1,ATOMS)).squeeze(-2)
    def nll(ix):
        return -subp.gather(-1,ix[...,None]).squeeze(-1).clamp_min(1e-8).log()
    joint=((1-loc.frac())*nll(lo)+loc.frac()*nll((lo+1).clamp_max(ATOMS-1))).mean()
    success=F.binary_cross_entropy_with_logits(scalar[...,0],s)
    marginal=F.mse_loss(scalar[...,1:3].sigmoid(),y[...,1:])+F.mse_loss(scalar[...,3],d['native'])
    direction=p[:,:,2].sum(-1)/(p[:,:,2].sum(-1)+p[:,:,0].sum(-1)).clamp_min(1e-8)
    sign=avg(F.binary_cross_entropy(direction.clamp(1e-6,1-1e-6),(event==2).float(),reduction='none'),event!=1)
    g=s+aux
    paired_rmse=(net.utilities(p)[0]-(g[:,i]-g[:,j])).square().mean()
    err=mu-diff
    return (joint+.45*sign+.5*paired_rmse+success+2*marginal
            +F.smooth_l1_loss(estimate,tar,beta=.2)+.45*err.abs().mean()
            +.07*(sd.log()+.5*(err/sd).square()).mean())

def edge_loss(logit,target,mask):
    return avg(F.binary_cross_entropy_with_logits(logit,target,reduction='none'),mask)

def objective(net,d):
    de,dl=early(d),late(d)
    first=d['first']
    row=torch.arange(len(first),device=first.device)
    mode=net.tag(d)
    ze,eo=net.encode(de,8)
    za=ze[row,first]
    enc=net.imagine(za[:,None],ze,mode[:,None])
    zl,lo=net.encode(dl,27)
    zl=zl+net.after(zl)
    start=pool(d['x'],d['role'][...,0])
    middle=pool(d['mid'],d['role'][...,0])
    initial=terminal(net,enc,start,d)
    seen=terminal(net,zl,middle,d)
    m=labeled(d['role'])
    return (.7*initial+seen+.65*F.cross_entropy(net.midpoint(za),mode)
            +F.smooth_l1_loss(start+net.mid_motion(za),middle,beta=.18)
            +F.mse_loss(net.mid_reward(za)[:,0],d['midscore'][:,3])
            +.32*edge_loss(eo['edge'][row,first],d['mide'],m)
            +.32*edge_loss(lo['edge'],d['final_e'],m[:,None])+.002*(eo['kl']+lo['kl']))

def restore(device='cpu'):
    net=World().to(device)
    net.load_state_dict(torch.load(OUT/'world.pt',map_location=device)['state'])
    return net.eval()

@torch.no_grad()
def validation(net,vals,towers):
    score=[]
    for d in vals:
        parts=[(len(ix),float(objective(net,take(d,ix)))) for ix in torch.arange(len(d['x']),device=d['x'].device).split(48)]
        score.append(sum(n*v for n,v in parts)/len(d['x']))
    tower=sum(len(ix)*float(physical_loss(net.core,sub(towers,ix)))
              for ix in torch.arange(len(towers['x']),device=d['x'].device).split(40))/len(towers['x'])
    return score+[tower]

def train():
    torch.manual_seed(271)
    torch.set_num_threads(2)
    device='cuda'
    ds=[read(dom,'train',device) for dom in DOMAINS]
    val=[read(dom,'val',device) for dom in DOMAINS]
    train={k:torch.cat([d[k] for d in ds]) for k in ds[0]}
    net=World().to(device)
    net.core.load_state_dict(torch.load(OUT/'physical-init.pt',map_location=device))
    feat=observation(train)
    net.center.copy_(feat.mean(0))
    net.spread.copy_(feat.std(0).clamp_min(.12))
    norm=((feat-net.center)/net.spread).cpu().numpy()
    km=MiniBatchKMeans(n_clusters=MODES,random_state=271,n_init=8,batch_size=768).fit(norm)
    net.prototypes.copy_(torch.tensor(km.cluster_centers_,device=device))
    physical=sources(device)
    towers=goal_training(current('cophy','val',device),False)
    groups=[[],[]]
    for name,p in net.named_parameters():
        groups[int(name.startswith('core.') and not name.startswith('core.control_token'))].append(p)
    opt=torch.optim.AdamW([dict(params=groups[0],lr=.00125),dict(params=groups[1],lr=.00012)])
    steps=2640
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.000035)
    log=[]
    best=float('inf')
    for step in range(1,steps+1):
        net.train()
        ix=torch.randint(len(train['x']),(56,),device=device)
        o=objective(net,take(train,ix))+.9*physical_loss(net.core,draw(physical,5))
        opt.zero_grad()
        o.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),4.)
        opt.step()
        sched.step()
        if step%120==0:
            net.eval()
            scores=validation(net,val,towers)
            metric=np.mean(scores[:2])+.6*scores[2]
            log.append(dict(step=step,val=scores,criterion=metric))
            if metric<best:
                best=metric
                torch.save(dict(step=step,state=net.state_dict()),OUT/'world.pt')
            print(log[-1],flush=True)
            (OUT/'fit.json').write_text(json.dumps(log,indent=1))

if __name__=='__main__':
    train()
