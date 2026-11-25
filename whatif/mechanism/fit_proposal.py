"""Fit a single mixture over addressed objects and trajectory landmarks on two tasks."""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import sys
from pathlib import Path
import zipfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .proposals import ROOT,WEIGHTS,MODES,local_view
from .native import planar,roles,N,D


def read_demo(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!='63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9':
        raise ValueError('unexpected official Push-T archive')
    lib=ctypes.CDLL(ctypes.util.find_library('blosc') or str(Path(sys.base_prefix)/'lib/libblosc.so.1'))
    def array(z,key):
        base='pusht/pusht_cchi_v7_replay.zarr/'+key+'/'
        meta=json.loads(z.read(base+'.zarray'))
        dt=np.dtype(meta['dtype'])
        chunk=meta['chunks']
        total=meta['shape'][0]
        rows=[]
        for j in range((total+chunk[0]-1)//chunk[0]):
            key='.'.join([str(j)]+['0']*(len(chunk)-1))
            src=z.read(base+key)
            buf=ctypes.create_string_buffer(int(np.prod(chunk))*dt.itemsize)
            count=lib.blosc_decompress(src,buf,len(buf))
            if count!=len(buf):
                raise RuntimeError('invalid compressed demo chunk')
            rows.append(np.frombuffer(buf,dtype=dt).reshape(chunk))
        return np.concatenate(rows)[:total]
    with zipfile.ZipFile(archive) as z:
        return [array(z,k) for k in ['data/state','data/action','meta/episode_ends']]


def examples(archive):
    state,actions,ends=read_demo(archive)
    splits=[[],[]]
    def append(split,x,role,goal,cmd):
        ids,rotation,features=local_view(x,role,goal)
        packed=np.zeros((3,features.shape[1]),np.float32)
        packed[:len(ids)]=features
        mask=np.arange(3)<len(ids)
        address=np.argmax(cmd[:,0])
        actor=np.where(ids==address)[0].item()
        target=cmd[address,1:].reshape(2,3)@rotation
        splits[split].append((packed,mask,actor,target.ravel()))
    role=roles([0],[1],[2,3,4,5])
    goal=planar([0.,0.],np.pi/4)
    for ep,(start,end) in enumerate(zip(np.r_[0,ends[:-1]],ends)):
        for t in range(start,end):
            x=np.zeros((N,D),np.float32)
            x[0]=planar((state[t,2:4]-256)/100,state[t,4])
            x[1]=planar((state[t,:2]-256)/100)
            cmd=np.zeros((N,7),np.float32)
            points=(actions[np.minimum(t+np.array([3,7]),end-1)]-state[t,:2])/100
            cmd[1]=np.r_[1,points[0],0.,points[1],0.]
            append(int(ep%10==9),x,role,goal,cmd)
    for split,part in enumerate(['train','val']):
        data=dict(np.load(ROOT/f'data/mechanism/fine/causalworld-{part}.npz'))
        for x,role,goal,a,y,r in zip(*(data[key] for key in ['x','role','goal','a','y','r'])):
            gap=y[:,0]-goal
            distance=(gap[:,:3]**2).sum(1)+.25*(gap[:,3:]**2).sum(1)
            pick=np.argmax(r[:,1]-.35*np.sqrt(distance))
            append(split,x,role,goal,a[pick])
            for j in range(len(a)):
                append(split,x,role,y[j,0],a[j])
    return splits


def main():
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    torch.manual_seed(271)
    torch.set_num_threads(3)
    device=args.device
    train,val=[[torch.as_tensor(np.stack(c),device=device) for c in zip(*rows)]
               for rows in examples(args.archive)]
    dim=train[0].shape[-1]
    center=train[0][train[1]].mean(0)
    spread=train[0][train[1]].std(0).clamp_min(.12)
    net=nn.Sequential(nn.Linear(dim,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),
                      nn.Linear(192,192),nn.SiLU(),nn.Linear(192,MODES*7)).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    epochs=110
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.0001)
    def objective(batch):
        feat,valid,actor,target=batch
        z=net((feat-center)/spread).unflatten(-1,(MODES,7))
        logits=z[...,0].masked_fill(~valid[...,None],-1e5).flatten(1).log_softmax(1)
        row=torch.arange(len(z),device=device)
        diff=z[row,actor,:,1:]-target[:,None]
        lp=logits.unflatten(-1,(3,MODES))[row,actor]
        # Fixed Laplace component scale in normalized object units.
        return -torch.logsumexp(lp-diff.abs().sum(-1)/.28,dim=-1).mean()
    WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
    best=np.inf
    for epoch in range(epochs):
        for ix in torch.randperm(len(train[0]),device=device).split(256):
            loss=objective([x[ix] for x in train])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if epoch%5==4:
            with torch.no_grad():
                score=float(objective(val).item())
            if score<best:
                best=score
                saved=dict(center=center.cpu().numpy(),spread=spread.cpu().numpy())
                for j,layer in enumerate(net[::2]):
                    saved['w'+str(j)]=layer.weight.detach().cpu().numpy()
                    saved['b'+str(j)]=layer.bias.detach().cpu().numpy()
                np.savez(WEIGHTS,**saved)
                meta=dict(train=len(train[0]),val=len(val[0]),epoch=epoch+1,val_loss=score,
                          seed=271,epochs=epochs,batch=256,components=MODES)
                WEIGHTS.with_suffix('.json').write_text(json.dumps(meta,indent=2))
            print(epoch+1,score,flush=True)


if __name__=='__main__':
    main()
