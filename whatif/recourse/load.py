"""Tensor views of observed transitions, candidate commands, and matched training labels."""
import numpy as np
import torch
from whatif.feedback.native import command
from .native import DATA,BANK

def read(dom,split,device='cpu',backup=False):
    tag='backup-' if backup else ''
    with np.load(DATA/f'{tag}{dom}-{split}.npz') as z:
        raw={k:z[k] for k in z.files}
    if 'first' in raw:
        raw['first']=raw['first'].astype('int64')
    unique,ix,inv=np.unique(raw['seed'],return_index=True,return_inverse=True)
    def menu(x,r,g):
        return np.stack([command(x,r,g,w) for w in BANK])
    raw['a']=np.stack([menu(raw['x'][i],raw['role'][i],raw['goal'][i]) for i in ix])[inv]
    if split.endswith('live') or backup:
        raw['late_a']=np.stack([menu(x,r,g) for x,r,g in zip(raw['mid'],raw['role'],raw['goal'])])
    return {k:torch.as_tensor(v,device=device).to(torch.long if k in ('seed','first') else torch.float32)
            for k,v in raw.items()}

def early(d):
    return {k:d[k] for k in ('x','role','goal','a','f0','fa','f1')} | {'same':torch.zeros_like(d['seed']).float()}

def late(d):
    return dict(x=d['mid'],role=d['role'],goal=d['goal'],a=d['late_a'],hx=d['hx'],ha=d['ha'],hy=d['hy'],
                f0=d['hx'][:,-1],fa=d['ha'][:,-1],f1=d['hy'][:,-1],same=torch.zeros_like(d['seed']).float())

def take(d,ix):
    return {k:v[ix] for k,v in d.items()}

def observation(d):
    # Motion and goal error in the initial object's frame, from public poses alone.
    m=d['role'][...,0:1]
    def movable(x):
        return (x*m).sum(1)/m.sum(1).clamp_min(1)
    start=movable(d['x']).reshape(-1,3,3)
    end=movable(d['mid']).reshape(-1,3,3)
    rot=torch.stack([start[:,1],start[:,2],torch.cross(start[:,1],start[:,2],dim=-1)],-1)
    change=(end-start)@rot
    gap=(d['goal'].reshape(-1,3,3)-end)@rot
    velocity=movable(d['hy'][:,-1]-d['hx'][:,-1]).reshape(-1,3,3)@rot
    return torch.cat([change,gap,velocity],1).flatten(1)
