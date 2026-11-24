"""Evaluate frozen shared mechanisms, abduction, and two second-action objectives."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from .chain_model import Branches,ContactTransfer,LENGTH,wrap


def bank(data):
    return torch.tensor(list(itertools.product([.025,.075,.14],[.18,.5,.85],[-.0025,0.,.0025])),
                        device=data.state.device,dtype=torch.float32)


@torch.no_grad()
def predict(model,data,time):
    a=torch.arange(len(data.actions),device=data.state.device)
    q=bank(data)
    result=[]
    for chunk in torch.arange(len(data.ids),device=q.device).split(12):
        i,action,k=torch.meshgrid(chunk,a,torch.arange(len(q),device=q.device),indexing="ij")
        t=torch.full_like(i,time)
        args=data.inputs(i,t,data.actions[action],q[k,:2],q[k,2])
        pred=model(*args)[0]*data.scale
        result.append(pred)
    return torch.cat(result)


def diff(a,b):
    d=a-b
    return torch.cat((d[...,:2],wrap(d[...,2:3])),dim=-1)


def weights(pred,obs,sigma,mask):
    err=diff(pred,obs[:,None])/sigma
    cost=(err.square()*mask[:,None,:,None]).sum(dim=(-1,-2))/mask.sum(dim=-1)[:,None]
    return (-.5*cost).softmax(dim=-1)


def calibrate(model,val):
    pred=predict(model,val,0)
    true=torch.cat((val.physics,val.noise[:,0,None]),dim=-1)
    match=(true[:,None]==bank(val)[None]).all(dim=-1)
    assert match.sum(dim=1).eq(1).all()
    ids=torch.arange(len(val.ids),device=pred.device)
    pred=pred[ids,:,match.long().argmax(dim=1)]
    err=diff(pred,val.delta[:,0]*val.scale)
    sigma=(err.square()*val.mask[:,None,:,None]).sum(dim=(0,2))/val.mask.sum()
    return torch.maximum(sigma.sqrt(),torch.tensor([.002,.002,.045],device=pred.device))


def score(model,data,scale,split,seed,gated):
    ids=torch.arange(len(data.ids),device=data.state.device)
    target=data.target
    ref=1
    alternatives=torch.arange(len(data.actions),device=ids.device)!=ref
    actual=data.delta*data.scale
    pred=predict(model,data,0)
    prior=torch.full((len(ids),27),1/27,device=ids.device)
    post=weights(pred[:,ref],actual[:,0,ref],scale[ref],data.mask)
    post_probe=weights(pred[:,data.probe],actual[:,0,data.probe],scale[data.probe],data.mask)
    fresh=post_probe.reshape(-1,9,3).sum(dim=-1).repeat_interleave(3,dim=-1)/3
    diff_pred=diff(pred[ids,:,:,target,:],pred[ids,ref,:,target,:][:,None])
    diff_true=diff(data.end[ids,0,:,target,:],data.end[ids,0,ref,target,:][:,None])
    forecast=predict(model,data,1)[ids,:,:,target,:2]+data.state[ids,1,target,:2][:,None,None]
    d=torch.linalg.vector_norm(forecast-data.goal[:,None,None],dim=-1)
    real=torch.linalg.vector_norm(data.end[ids,1,:,target,:2]-data.goal[:,None],dim=-1)
    zero_error=diff_true[:,alternatives,:2].norm(dim=-1).mean(dim=-1)*1000
    rows=[]
    for mode,w0,w1 in [('prior',prior,prior),('reflect',post,fresh)]:
        w=w0[:,None,:,None]
        pos=(w*diff_pred[...,:2]).sum(dim=2)
        err=(pos-diff_true[...,:2]).norm(dim=-1)[:,alternatives].mean(dim=-1)*1000
        theta=diff_pred[...,2]
        yaw=torch.atan2((w0[:,None]*theta.sin()).sum(dim=-1),(w0[:,None]*theta.cos()).sum(dim=-1))
        yaw_err=wrap(yaw-diff_true[...,2]).abs()[:,alternatives].mean(dim=-1).rad2deg()
        for policy,utility in [('mean',-d),('paired',torch.tanh((d[:,ref:ref+1]-d)/.012))]:
            selected=(utility*w1[:,None]).sum(dim=-1).argmax(dim=1)
            end=real[ids,selected]
            columns=torch.stack((err,yaw_err,zero_error,end*1000,(end<=.012).float(),
                                 real.min(dim=1).values*1000,selected.float()),dim=1).cpu().numpy()
            for scene,row in zip(data.ids,columns):
                rows.append(dict(split=split,scene=int(scene),seed=seed,
                                 model="graph" if gated else "dense",mode=mode,policy=policy,
                                 **dict(zip(("effect_mm","yaw_deg","zero_mm","end_mm","success","oracle_mm","action"),
                                            row.tolist()))))
    return rows


def summarize(rows):
    groups={}
    keys=('effect_mm','yaw_deg','zero_mm','end_mm','success','oracle_mm')
    for row in rows:
        label="/".join(row[k] for k in ("split","model","mode","policy"))
        groups.setdefault(label,[]).append(row)
    matrices,means={},{}
    for label,group in groups.items():
        group.sort(key=lambda r:(r["seed"],r["scene"]))
        n=len({r["scene"] for r in group})
        mat=np.array([[r[k] for k in keys] for r in group]).reshape(-1,n,len(keys))
        matrices[label]=mat.mean(axis=0)
        means[label]={k:dict(mean=float(mat[:,:,i].mean()),per_seed=mat[:,:,i].mean(axis=1).tolist())
                      for i,k in enumerate(keys)}
    rng=np.random.default_rng(20261124)
    contrasts={}
    for split in ("two","three","physical"):
        base=split+"/graph/reflect/mean"
        ix=rng.integers(len(matrices[base]),size=(5000,len(matrices[base])))
        for ref in ("dense/reflect/mean","graph/prior/mean","graph/reflect/paired"):
            diffmat=matrices[base]-matrices[split+"/"+ref]
            ci=np.quantile(diffmat[ix].mean(axis=1),[.025,.975],axis=0)
            contrasts[base+" minus "+ref]={k:[float(diffmat[:,i].mean()),*ci[:,i].tolist()] for i,k in enumerate(keys)}
    return dict(means=means,contrasts=contrasts)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path('data/chains'))
    ap.add_argument('--out',type=Path,default=Path('results/chains'))
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,29,47])
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    torch.set_num_threads(2)
    val=Branches(args.data/'val.npz',args.device)
    splits={k:Branches(args.data/(k+'.npz'),args.device) for k in ('two','three','physical')}
    rows=[]
    with torch.no_grad():
        for seed in args.seeds:
            for gated in (True,False):
                name=('graph' if gated else 'dense')+f'-{seed}'
                model=ContactTransfer(gated).to(args.device)
                model.load_state_dict(torch.load(args.out/(name+'.pt'),map_location=args.device,weights_only=True))
                model.eval()
                scale=calibrate(model,val)
                for k,data in splits.items():
                    rows.extend(score(model,data,scale,k,seed,gated))
    with (args.out/'per_scene.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    (args.out/'summary.json').write_text(json.dumps(summarize(rows),indent=2)+'\n')


if __name__=='__main__':
    main()
