"""The same object tensors serve a moved block, one physical command, and a feedback program."""
import numpy as np
import torch
from whatif.hitting.data import load_np,menus
from .native import BANK,DATA,ENDS,TICKS

def tensor(d,dev):
    return {k:torch.as_tensor(v,device=dev,dtype=torch.long if k in ('seed','bin','first') else torch.float32)
            for k,v in d.items()}

def common(raw,dom,phase,dev,live=False):
    tower=dom=='cophy'
    x=raw['x']
    b,n=x.shape[:2]
    if tower:
        hx,ha,hy=[raw[k][:,None] for k in ('f0','fa','f1')]
        fe=raw['fe']
    else:
        hx,ha,hy=[raw[k] for k in ('hx','ha','hy')]
        fe=raw.get('he',None)
        fe=fe[:,-1] if fe is not None else np.zeros((b,n,n))
    if hx.shape[1]==1:
        hx,ha,hy=[np.repeat(q,3,axis=1) for q in (hx,ha,hy)]
    if live:
        a=np.stack([menus(v,r,g) for v,r,g in zip(x,raw['role'],raw['goal'])])
    else:
        a=raw['a']
    k=a.shape[1]
    atomic=tower or phase=='short'
    program=np.zeros((b,k,11)) if atomic else np.broadcast_to(BANK,(b,k,11)).copy()
    if tower:
        widths=[6]*5
    elif phase=='short':
        widths=np.diff([0,*TICKS])
    else:
        widths=np.diff([1 if phase=='early' else 9,*ENDS[phase]])*8
    row=dict(x=x,hx=hx,ha=ha,hy=hy,f0=hx[:,-1],fa=ha[:,-1],f1=hy[:,-1],fe=fe,
             role=raw['role'],goal=raw['goal'],a=a,program=program,same=np.full(b,float(tower)),
             width=np.tile(widths,(b,1)),feedback=np.full(b,float(not atomic)),seed=raw['seed'],
             first=raw.get('first',np.zeros(b)))
    if live:
        return tensor(row,dev)
    micro=phase=='short' and not tower
    t=raw['short_state'] if micro else raw['traj']
    edge=raw['short_edge'] if micro else raw['contact']
    valid=raw['short_valid'] if micro else raw.get('valid',np.ones(t.shape[:3]))
    weights=raw['role'][:,None,None,:,0:1]
    body=(t*weights).sum(-2)/weights.sum(-2).clip(min=1)
    center=(x*weights[:,0,0]).sum(-2)/weights[:,0,0].sum(-2).clip(min=1)
    rot=np.stack([center[:,3:6],center[:,6:9],np.cross(center[:,3:6],center[:,6:9])],-1)
    value=((body-center[:,None,None]).reshape(b,k,5,3,3)@rot[:,None,None]).reshape(b,k,5,9)
    rew=raw['rewards'] if not atomic else np.zeros((b,k,5))
    reward=raw['r1'].copy() if not tower else np.zeros((b,k,3))
    outcome=raw.get('outcome',np.zeros((b,k,3)))
    row.update(traj=t,contact=edge,valid=valid,y=t[:,:,-1],e=edge.max(2),r=reward,
               reward_mask=np.full(b,float(not tower)),measured=np.full(b,float(tower)),
               effects=raw.get('impact',np.zeros_like(edge)),value=np.concatenate([value,rew[...,None]],-1),
               outcome=outcome,bin=np.zeros((b,k)) if atomic else np.searchsorted(ENDS[phase],raw['hit']))
    row['impact']=row['effects'][:,:,-1]
    return tensor(row,dev)

def dataset(dom,phase,split,dev):
    if dom=='cophy':
        raw=load_np(DATA/f'cophy-{split}.npz')
    else:
        raw=load_np(DATA/f'{dom}-{"late" if phase=="short" else phase}-{split}.npz')
        mask=raw['alive'].astype(bool)
        raw={k:v[mask] for k,v in raw.items()}
    return common(raw,dom,phase,dev)

def live(dom,phase,path,dev):
    raw=load_np(path)
    if phase=='early':
        raw.update(x=raw['f1'],hx=raw['f0'][:,None],ha=raw['fa'][:,None],hy=raw['f1'][:,None])
    return common(raw,dom,phase,dev,True),raw

BRANCH=('a','program','traj','contact','valid','y','e','r','effects','value','outcome','bin','impact')
def sample(d,n,k):
    ix=torch.randint(len(d['x']),(n,),device=d['x'].device)
    cols=torch.randperm(d['a'].size(1),device=ix.device)[:k]
    return {key:v[ix][:,cols] if key in BRANCH else v[ix] for key,v in d.items()}
