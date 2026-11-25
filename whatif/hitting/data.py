"""Shared factual tensors and action menus at either prediction horizon."""
import numpy as np
import torch
from whatif.feedback.native import command
from .native import DATA,BANK

def menus(x,role,goal):
    # Reproduce the frozen output layer for all seven policies with one actor call.
    a=command(x,role,goal,None)
    node=a[:,0].argmax()
    obj=np.average(x,axis=0,weights=role[:,0])
    u,v=obj[3:6],obj[6:9]
    rot=np.stack([u,v,np.cross(u,v)])
    rel=(goal-obj).reshape(3,3)
    d=a[node,1:].reshape(2,3)
    gains=BANK[:,:2]+BANK[:,6:8]*np.tanh(np.linalg.norm(rel))
    shift=BANK[:,2:5]@rot+(BANK[:,8:11]@np.tanh(rel@rot.T))@rot
    out=np.tile(a,(len(BANK),1,1))
    cmd=out[:,node,1:].reshape(-1,2,3)
    cmd+=gains[...,None]*d+shift[:,None]
    cmd[:,1]+=BANK[:,5,None]*(d[1]-d[0])
    return out

def load_np(path):
    with np.load(path) as f:
        return {k:f[k] for k in f.files}

def tensor(d,device):
    return {k:torch.as_tensor(v,device=device).to(torch.long if k in ('seed','first','hit','bin') else torch.float32)
            for k,v in d.items()}

def data(dom,phase,split,device,live=False,path=None):
    if phase=='late':
        raw=load_np(path or DATA/(f'{dom}-live.npz' if live else f'late-{dom}-{split}.npz'))
        if not live:
            ix=raw['mid_alive'].astype(bool)
            raw={k:v[ix] for k,v in raw.items()}
        d={k:raw[k] for k in ('x','role','goal','hx','ha','hy','seed','first')}
        d.update(f0=raw['hx'][:,-1],fa=raw['ha'][:,-1],f1=raw['hy'][:,-1])
        candidates=np.stack([menus(x,r,g) for x,r,g in zip(d['x'],d['role'],d['goal'])])
        cap=(10,13,17,25,36)
    else:
        raw=load_np(path or DATA/(f'{dom}-initial.npz' if live else f'mid-{dom}-{split}.npz'))
        if not live:
            raw={k:v[::len(BANK)] for k,v in raw.items()}
            targets=load_np(DATA/f'early-{dom}-{split}.npz')
            if not np.array_equal(raw['seed'],targets['seed']):
                raise ValueError('early layout mismatch')
            raw.update(targets)
        d={k:raw[k] for k in ('f0','fa','f1','role','goal','seed','alive')}
        d['x']=d['f1']
        menu=np.stack([menus(x,r,g) for x,r,g in zip(d['x'],d['role'],d['goal'])])
        candidates=np.concatenate([menu[:,:1],menu],axis=1)
        cap=(2,5,9,17,36)
    d['a']=candidates
    d['same']=np.zeros(len(candidates))
    if not live:
        weight=d['role'][:,None,None,:,0:1]
        body=(raw['traj']*weight).sum(-2)/weight.sum(-2).clip(min=1)
        x=(d['x']*d['role'][...,0:1]).sum(1)/d['role'][...,0:1].sum(1).clip(min=1)
        # Continuous future in the initial mover's local frame, in simulator units.
        rot=np.stack([x[:,3:6],x[:,6:],np.cross(x[:,3:6],x[:,6:])],axis=-1)
        change=(body-x[:,None,None]).reshape(*body.shape[:-1],3,3)@rot[:,None,None]
        d.update(value=np.concatenate([change.reshape(body.shape),raw['rewards'][...,None]],-1),
                 outcome=raw['outcome'],bin=np.searchsorted(cap,raw['hit']))
        if phase=='late':
            d.update(edge=raw['edges'][:,:,-1],y=raw['y'],e=raw['e'],r=raw['r'])
    return tensor(d,device)

def take(d,ix):
    return {k:v[ix] for k,v in d.items()}
