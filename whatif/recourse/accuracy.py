"""Blind forecast, forecast after the genuine midpoint, and fitted-policy outcome laws."""
import argparse
import json
import math
import torch
from sklearn.metrics import average_precision_score
from whatif.mechanism.model import pool
from .model import MODES,DIM
from .load import read,take,early,late
from .fit import restore,indices
from .native import OUT,DOMAINS
from .backup import restore as backed, rows

def normabs(delta,sd):
    sd=sd.clamp_min(1e-7)
    t=delta/sd
    return delta*(2*torch.special.ndtr(t)-1)+sd*math.sqrt(2/math.pi)*(-.5*t.square()).exp()

def normal_score(m,w,sd,truth):
    # [layout, pair, component, coordinate]
    a=(w[...,None]*normabs(m-truth[:,:,None],sd)).sum(2)
    gap=m.unsqueeze(3)-m.unsqueeze(2)
    mix=(sd.square().unsqueeze(3)+sd.square().unsqueeze(2)).sqrt()
    return a-.5*(w[:,:,None,:,None]*w[:,:,:,None,None]*normabs(gap,mix)).sum((2,3))

def stats(out,y,target,unit):
    val,dm,ds,w,law,support=out
    i,j=indices(y.size(1),'cpu')
    position=val[...,:3]-target[...,:3]
    dr=(target[:,i]-target[:,j])
    predicted=(dm*w[...,None]).sum(2)
    crps=normal_score(dm,w,ds,dr)
    G=y[...,0]+.5*y[...,1]+.25*y[...,2]
    gain=G[:,i]-G[:,j]
    mean=law@support
    first=(law*(support-gain[...,None]).abs()).sum(-1)
    diff=(support[:,None]-support).abs()
    dcrps=first-.5*((law@diff)*law).sum(-1)
    s=y[...,0]
    # Compare all unordered pairs, oriented by the common fixed bank ordering.
    event=s[:,i]-s[:,j]
    p=law.unflatten(-1,(3,17))
    def rmse(x):
        return x.square().mean().sqrt().item()
    return dict(position=unit*rmse(position.norm(dim=-1)),position_delta=unit*rmse((predicted-dr)[...,:3].norm(dim=-1)),
                position_crps=unit*crps[...,:3].mean().item(),native=rmse(val[...,3]-target[...,3]),
                native_delta=rmse((predicted-dr)[...,3]),native_crps=crps[...,3].mean().item(),
                return_delta=rmse(mean-gain),return_crps=dcrps.mean().item(),
                positive=float((event>0).float().mean()),rescue_ap=float(average_precision_score((event>0).flatten(),p[:,:,2].sum(-1).flatten())),
                negative=float((event<0).float().mean()),break_ap=float(average_precision_score((event<0).flatten(),p[:,:,0].sum(-1).flatten())))

def forecasts(net,z,q,start):
    b,m,k,_=z.shape
    z=z.reshape(-1,k,DIM)
    scalar=net.result(z)
    pos=start[:,None,None,:3]+net.motion(z)[...,:3].reshape(b,m,k,3)
    val=torch.cat([pos,scalar[...,3:].reshape(b,m,k,1)],-1)
    i,j=indices(k,z.device)
    mu,sd=net.effect(z,val.reshape(-1,k,4),i,j)
    p=net.distribution(z[:,i],z[:,j]).reshape(b,m,len(i),-1)
    def transpose(v):
        return v.reshape(b,m,len(i),4).transpose(1,2)
    return [(q[:,:,None,None]*val).sum(1),transpose(mu),transpose(sd),
            q[:,None].expand(-1,len(i),-1),(q[:,:,None,None]*p).sum(1)]

@torch.no_grad()
def long():
    net=restore('cuda')
    r={}
    for dom in DOMAINS:
        d=read(dom,'test','cuda')
        # Both forecasts for exactly the same still-actionable second decisions.
        d=take(d,d['midscore'][:,0]==0)
        collected={'before':[],'observed':[]}
        mids=[]
        for ix in torch.arange(len(d['seed']),device='cuda').split(32):
            v=take(d,ix)
            row=torch.arange(len(ix),device='cuda')
            e,oe=net.encode(early(v),8)
            a=e[row,v['first']]
            q=net.midpoint(a).softmax(-1)
            zz=net.imagine(a[:,None,None],e[:,None],torch.arange(MODES,device='cuda')[None,:,None])
            start=pool(v['x'],v['role'][...,0])
            mid=pool(v['mid'],v['role'][...,0])
            pred=start+net.mid_motion(a)
            mids.append(torch.stack([(pred[:,:3]-mid[:,:3]).norm(dim=-1),
                                     net.mid_reward(a)[:,0]-v['midscore'][:,3],
                                     (q.argmax(-1)==net.tag(v)).float()],-1).cpu())
            late_z,ol=net.encode(late(v),27)
            known=(late_z+net.after(late_z))[:,None]
            for key,f in [('before',forecasts(net,zz,q,start)),('observed',forecasts(net,known,torch.ones_like(q[:,:1]),mid))]:
                collected[key].append([v.cpu() for v in f])
        target=torch.cat([pool(d['final'][...,:3],d['role'][:,None,:,0]),d['native'][...,None]],-1).cpu()
        y=d['outcome'].cpu()
        support=(torch.arange(-1,2)[:,None]+net.grid.cpu()).flatten()
        unit=100 if dom=='pusht' else 65
        m=torch.cat(mids)
        result=dict(actionable=len(y),layouts=int(d['seed'].unique().numel()),mid_position=unit*m[:,0].square().mean().sqrt().item(),
                    mid_reward=m[:,1].square().mean().sqrt().item(),mode_accuracy=m[:,2].mean().item())
        for k,records in collected.items():
            out=[torch.cat(v) for v in zip(*records)]+[support]
            result[k]=stats(out,y,target,unit)
        r[dom]=result
    (OUT/'long.json').write_text(json.dumps(r,indent=2,allow_nan=False))

@torch.no_grad()
def backup():
    net=backed()
    result={}
    for dom in DOMAINS:
        x,y,v=rows(dom,'test','cpu')
        o=net(x)
        pi,p,mu,sd=o
        i,j=indices(y.size(1),'cpu')
        w=pi[:,None].expand(-1,len(i),-1)
        pick=[0,1,2,9]
        mu,sd=mu[...,pick],sd[...,pick]
        val=(pi[:,None,:,None]*mu).sum(2)
        law=net.diff(o,i,j)
        metrics=stats([val,mu[:,i]-mu[:,j],(sd[:,i].square()+sd[:,j].square()).sqrt(),w,law,net.delta],
                      y,v[...,pick],100 if dom=='pusht' else 65)
        decision={}
        s=y[...,0]
        g=s+.5*y[...,1]+.25*y[...,2]
        for arm,score in net.scores(o).items():
            j=score.argmax(1)
            success=s.gather(1,j[:,None])[:,0]
            decision[arm]=dict(success=int(success.sum()),rescued=int((success>s[:,1]).sum()),
                               lost=int((success<s[:,1]).sum()),G=g.gather(1,j[:,None]).mean().item())
        metrics.update(fixed_first_with_recourse=int(s[:,1].sum()),original_first_with_recourse=int(s[:,0].sum()),
                       dec=decision,oracle=int(s.max(1).values.sum()))
        result[dom]=metrics
    (OUT/'backup-test.json').write_text(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=['long','backup'])
    torch.set_num_threads(3)
    (long if p.parse_args().job=='long' else backup)()
