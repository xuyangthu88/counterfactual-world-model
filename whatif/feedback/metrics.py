"""Paired policy predictions over time and retained one-macro benchmark comparisons."""
import argparse
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from whatif.mechanism.model import labeled,pool
from whatif.mechanism.train import OUT as OLD,build,goal_training
from whatif.mechanism.evaluate import graph,metrics,mixture_crps
from whatif.mechanism.directed import DirectedNCII
from whatif.paired.data import OUT as PREVIOUS
from whatif.paired.fit import load_rows,world
from whatif.paired.evaluate import graph_metrics,remote,frozen
from .fit import data,restore,pairs,subset
from .model import preference
from .native import OUT,DATA,DOMAINS

@torch.no_grad()
def short():
    old=json.loads((PREVIOUS/'offline.json').read_text())
    records={}
    for dom in ('cophy',*DOMAINS):
        path=DATA/('cophy-test.npz' if dom=='cophy' else f'expanded-{dom}-test.npz')
        raw=load_rows(path,dom,'cpu',DATA/'excision-test.npz' if dom=='cophy' else None)
        d=goal_training(raw,False)
        models=[('transport','mechanism',world())]
        if dom=='cophy':
            models.append(('ncii_excision','ncii',frozen(DirectedNCII(),OLD/'directed/ncii.pt')))
        else:
            for name in ('ncii','ncii_contact','pets','cophynet'):
                path=(OLD/f'{name}.pt' if name.startswith('ncii') else OLD/'demonstrations'/f'{name}.pt')
                models.append((name,'ncii' if name.startswith('ncii') else name,frozen(build(name,'cpu'),path)))
        records[dom]={}
        for name,mode,net in models:
            r={}
            if mode in ('ncii','mechanism'):
                for key,label in [('edge','contact'),('causal','excision')][:2 if dom=='cophy' else 1]:
                    r[label]=graph_metrics(net,None,d,key,old[dom][name][label]['threshold'])
            if mode!='ncii':
                r.update(metrics(mode,net,d,dom))
                if dom=='cophy':
                    dist,sq,cnt=remote(net,d)
                    r.update(distant_effect_rmse=dist,distant_receiver_pairs=int(cnt.sum()))
            records[dom][name]=r
    (OUT/'short.json').write_text(json.dumps(records,indent=2,allow_nan=False))

@torch.no_grad()
def predict(net,d):
    parts=[]
    for ix in torch.arange(len(d['x'])).split(32):
        row=subset(d,ix)
        i,j=pairs(row['a'].size(1),'cpu')
        o=net(row)
        mu,sd,p=net.contrasts(o,i,j,row['role'][...,0])
        parts.append(dict(y=o['y'],edge=o['edge'].sigmoid(),scalar=o['scalar'],mu=mu,sd=sd,p=p))
    return {k:torch.cat([v[k] for v in parts]) for k in parts[0]}

def rms(x):
    return x.square().mean().sqrt().item()

def effects(net,raw,o,i,j):
    last=raw['outcome'][:,:,-1]
    g=last[...,0]+.5*last[...,1]+.25*last[...,2]
    true=g[:,i]-g[:,j]
    p=o['p'].flatten(-2)
    values=(torch.arange(-1,2)[:,None]+net.grid).flatten()
    first=(p*(values-true[...,None]).abs()).sum(-1)
    kernel=(values[:,None]-values[None,:]).abs()
    crps=(first-.5*((p@kernel)*p).sum(-1)).mean()
    a,b=last[:,i,0],last[:,j,0]
    r=dict(return_difference_rmse=rms((p*values).sum(-1)-true),crps=crps.item())
    for name,indicator,index in [('win',a>b,2),('loss',a<b,0)]:
        pred=o['p'][:,:,index].sum(-1)
        r[name]=dict(rate=indicator.float().mean().item(),
                     ap=float(average_precision_score(indicator.flatten(),pred.flatten())),
                     brier=(indicator.float()-pred).square().mean().item())
    # In the upper triangle policy zero is the left endpoint. Reverse orientation
    # to report rescuing (or breaking) the generator by using the alternative.
    original=(i==0)
    alt=last[:,j[original],0]
    base=last[:,:1,0]
    r['available_rescues']=int(((alt>base).any(-1)).sum())
    r['available_regressions']=int(((alt<base).any(-1)).sum())
    for name,indicator,index in [('rescue_generator',alt>base,0),('break_generator',alt<base,2)]:
        pred=o['p'][:,original,index].sum(-1)
        r[name]=dict(ap=float(average_precision_score(indicator.flatten(),pred.flatten())),
                     rate=indicator.float().mean().item())
    return r

@torch.no_grad()
def long(history):
    tag='history-' if history else ''
    net=restore(history=history)
    records={}
    for dom in DOMAINS:
        val,test=[data(dom,s,'cpu',history=history) for s in ('val','test')]
        if history:
            val,test=[subset(d,d['alive'].bool()) for d in (val,test)]
        def mask(d):
            return (labeled(d['role'])*d['role'][:,None,:,0]).bool()[:,None].expand_as(d['e'])
        v,o=predict(net,val),predict(net,test)
        i,j=pairs(test['a'].size(1),'cpu')
        mover=test['role'][...,0]
        pos=pool(test['future'][...,:3],mover[:,None,None,:])
        y=pool(o['y'][...,:3],mover[:,None,None,:])
        scale=100 if dom=='pusht' else 65
        r=dict(n_layouts=len(y),pairs=len(i)*len(y),outcomes=effects(net,test,o,i,j),horizons={})
        # Describe how many of the available rescues each selector actually identifies.
        # This upper bound uses held-out branch outcomes only after predictions are made.
        scores=preference(net,o,i,j,o['p'])
        s=test['outcome'][:,:,-1,0].bool()
        counts={}
        for name,value in scores.items():
            pick=value.argmax(1)
            observed=s.gather(1,pick[:,None])[:,0]
            counts[name]=dict(rescues=int((observed&~s[:,0]).sum()),breaks=int((~observed&s[:,0]).sum()))
        r['rescue_counts']=counts
        for t,time in enumerate([1,4,12,18 if history else 35]):
            thr=graph(val['contact'][:,:,t][mask(val)],v['edge'][:,:,t][mask(val)])['threshold']
            edges=graph(test['contact'][:,:,t][mask(test)],o['edge'][:,:,t][mask(test)],thr)
            dn=test['native'][:,i,t]-test['native'][:,j,t]
            dp=pos[:,i,t]-pos[:,j,t]
            mu=o['mu'][:,:,t]
            sd=o['sd'][:,:,t]
            r['horizons'][time]=dict(contact=edges,position_rmse=scale*rms((y[:,:,t]-pos[:,:,t]).norm(dim=-1)),
                                     difference_rmse=scale*rms((mu[...,:3]-dp).norm(dim=-1)),
                                     difference_crps=scale*mixture_crps(mu[None,...,:3],sd[None,...,:3],dp).mean().item(),
                                     reward_rmse=rms(o['scalar'][:,:,t,3]-test['native'][:,:,t]),
                                     reward_effect_rmse=rms(mu[...,3]-dn))
        records[dom]=r
    (OUT/(tag+'long.json')).write_text(json.dumps(records,indent=2,allow_nan=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('job',choices=['short','long','history'])
    args=parser.parse_args()
    torch.set_num_threads(3)
    if args.job=='short':
        short()
    else:
        long(args.job=='history')
