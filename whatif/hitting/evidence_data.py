"""Graph representations at macro 1, observed displacement to macro 9 and frozen late scores."""
import numpy as np
import torch
from whatif.mechanism.model import pool
from .native import DATA
from .data import data,menus,take
from .fit import restore
from .policy import features,selector

@torch.no_grad()
def panel(dom,split,initial,late,readout,dev='cuda'):
    live=split=='live'
    d=data(dom,'late',split,dev,live=live,path=(DATA/f'evidence-mid-{dom}.npz' if live else None))
    first=dict(x=d['hy'][:,0],f0=d['hx'][:,0],fa=d['ha'][:,0],f1=d['hy'][:,0],role=d['role'],goal=d['goal'],same=d['same'])
    x,r,g=[first[k].cpu().numpy() for k in ('x','role','goal')]
    a=torch.tensor(np.stack([menus(v,w,u) for v,w,u in zip(x,r,g)]),device=dev)
    first['a']=torch.cat([a[:,:1],a],1)
    origin=pool(first['x'],d['role'][...,0])
    rot=torch.stack([origin[:,3:6],origin[:,6:9],torch.cross(origin[:,3:6],origin[:,6:9],dim=-1)],-1)
    difference=d['x']-first['x']
    mover=pool(difference,d['role'][...,0]).reshape(-1,3,3)@rot
    robot=pool(difference[...,:3],d['role'][...,1])[:,None]@rot
    obs=torch.cat([mover.flatten(1),robot[:,0]],-1)
    x,y=[],[]
    for ix in torch.arange(len(obs),device=dev).split(64):
        e=take(first,ix)
        z,_=initial.encode(e,True)
        z,_=initial.program_graph(e,z,initial.core.abduct(e)[:3])
        x.append(z[:,1:])
        f=features(late,take(d,ix),False)
        y.append(readout(f,'means'))
    codes=torch.cat(x)
    out=dict(z=codes,first=codes[torch.arange(len(codes),device=dev),d['first']],w=obs,base=torch.cat(y),seed=d['seed'])
    if not live:
        out['target']=d['outcome']
    return out

def worlds(dev='cuda'):
    return restore('joint',True,dev),restore('joint',False,dev),selector(False,dev)
