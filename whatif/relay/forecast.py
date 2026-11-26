"""Matched full trajectories for all seven programs and their pairwise effect laws."""
import json
import math
import torch
from whatif.hitting.data import take
from whatif.hitting.fit import restore as previous,pairs
from whatif.hitting.forecast import law_metrics
from whatif.mechanism.model import pool
from whatif.recourse.fit import restore as stage24
from whatif.recourse.accuracy import forecasts,stats
from .native import DOMAINS,OUT
from .model import restore,coupled,difference
from .data import dataset

def absnormal(m,sd):
    z=m/sd
    return m*(2*torch.special.ndtr(z)-1)+sd*math.sqrt(2/math.pi)*(-z.square()/2).exp()

def crps(y,mu,sd,w):
    q=w[:,None,None,:,None]
    first=(q*absnormal(mu-y.unsqueeze(3),sd)).sum(3)
    diff=mu.unsqueeze(4)-mu.unsqueeze(3)
    sig=(sd.square().unsqueeze(4)+sd.square().unsqueeze(3)).sqrt()
    second=(q.unsqueeze(4)*q.unsqueeze(3)*absnormal(diff,sig)).sum((3,4))
    return first-.5*second

@torch.no_grad()
def older(d,unit):
    net=stage24('cuda')
    parts=[]
    for ix in torch.arange(len(d['x']),device='cuda').split(32):
        v=take(d,ix)
        z=net.encode(v,27)[0]
        known=(z+net.after(z))[:,None]
        start=pool(v['x'],v['role'][...,0])
        parts.append([q.cpu() for q in forecasts(net,known,torch.ones_like(start[:,:1]),start)])
    target=torch.cat([pool(d['traj'][:,:,-1],d['role'][:,None,:,0])[...,:3],d['value'][:,:,-1,9:]],-1)
    out=[torch.cat(v) for v in zip(*parts)]+[(torch.arange(-1,2)[:,None]+net.grid.cpu()).flatten()]
    return stats(out,d['outcome'].cpu(),target.cpu(),unit)

@torch.no_grad()
def evaluate(net,d,kind,unit):
    i,j=pairs(d['a'].size(1),'cuda')
    outputs=[]
    selected=[0,1,2,9]
    for ix in torch.arange(len(d['x']),device='cuda').split(12):
        v=take(d,ix)
        old=kind=='stage25'
        o,_=net(v,False) if old else net(v)
        y=v['value']; diff=(y[:,i]-y[:,j])[...,selected]
        p=o['p']; mu=o['mu']; err=mu-y
        part=dict(err=err,de=err[:,i]-err[:,j],p=p,score=o['score'])
        for name,ind in [('joint',False),('independent',True)]:
            if old:
                dm,sd=net.head.normal(o,i,j,independent=ind)
                dm,sd=dm[...,selected].unsqueeze(3),sd[...,selected].unsqueeze(3)
                w=p.new_ones(len(p),1)
                law=p[:,i,:,None]*p[:,j,None,:] if ind else net.head.coupling(o,i,j)
            else:
                dm,sd,w=difference(o,i,j,independent=ind)
                dm,sd=dm[...,selected],sd[...,selected]
                law=coupled(o,i,j,ind)
            part['law_'+name]=law
            part['crps_'+name]=crps(diff,dm,sd,w)
        outputs.append({k:v.cpu() for k,v in part.items()})
    r={k:torch.cat([o[k] for o in outputs]) for k in outputs[0]}
    target=d['bin'].cpu()
    i,j=i.cpu(),j.cpu()
    s=d['outcome'][...,0].cpu()
    prob=1-r['p'][...,-1]
    result=dict(layouts=len(target),success_brier=(prob-s).square().mean().item(),
                success_delta_rmse=((prob-s)[:,i]-(prob-s)[:,j]).square().mean().sqrt().item(),
                terminal_score_rmse=(r['score']-d['outcome'][...,1:].cpu()).square().mean((0,1)).sqrt().tolist())
    for key,v in [('future',r['err']),('effect',r['de'])]:
        result[key+'_xyz']=(unit*v[...,:3].square().sum(-1).mean((0,1)).sqrt()).tolist()
        result[key+'_reward']=v[...,9].square().mean((0,1)).sqrt().tolist()
    for k in ('joint','independent'):
        c=r['crps_'+k]
        result[k]=law_metrics(r['law_'+k],target,i,j)
        result[k].update(xyz_crps=(unit*c[...,:3].mean((0,1,3))).tolist(),reward_crps=c[...,3].mean((0,1)).tolist())
    return result

if __name__=='__main__':
    torch.set_num_threads(2)
    records={}
    for dom in DOMAINS:
        for phase in ('early','late'):
            d=dataset(dom,phase,'test','cuda')
            unit=100 if dom=='pusht' else 65
            r={k:evaluate(restore(k,'cuda'),d,k,unit) for k in ('dynamic','fixed','unpaired')}
            if phase=='late':
                r['stage25']=evaluate(previous('joint',False,'cuda'),d,'stage25',unit)
                r['stage24']=older(d,unit)
            records[dom+'-'+phase]=r
            print(dom,phase,flush=True)
    (OUT/'forecast.json').write_text(json.dumps(records,indent=1,allow_nan=False))
