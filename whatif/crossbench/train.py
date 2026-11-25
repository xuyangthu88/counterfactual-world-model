"""Train once per method. Select only on held-out layouts, before decision evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .models import (OUT, TASKS, ID, ContactModel, NullPoint, actions, cophy_class,
                     cophy_predict, difference, expand_state, graph_loss, nll, panel, wrap)


def pets_features(s,a):
    # Periodic coordinates plus task geometry. Each ensemble predicts the same observable state.
    return torch.cat((s[...,:2],s[...,2:3].cos(),s[...,2:3].sin(),s[...,3:11],
                      s[...,11:12].cos(),s[...,11:12].sin(),s[...,12:],a),dim=-1)


def flat(d):
    z=d['state'][:,:,None].expand_as(d['end'])
    a=actions(z).expand(*z.shape[:-1],-1)
    x=pets_features(z,a).reshape(-1,22)
    y=difference(d['end'],z)[...,ID].reshape(-1,len(ID))
    return x,y


def pets_train(d,v,device):
    from mbrl.models import GaussianMLP
    m=GaussianMLP(22,len(ID),device,num_layers=3,ensemble_size=5,hid_size=128)
    x,y=flat(d)
    vx,vy=flat(v)
    xm,ym=x.mean(0),y.mean(0)
    xs,ys=x.std(0).clamp_min(.05),y.std(0).clamp_min(.04)
    x,y=(x-xm)/xs,(y-ym)/ys
    vx,vy=(vx-xm)/xs,(vy-ym)/ys
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    best=np.inf
    bootstrap=torch.randint(len(x),(5,len(x)),device=device)
    for epoch in range(65):
        for ids in torch.randperm(len(x),device=device).split(384):
            ix=bootstrap[:,ids]
            loss=m.loss(x[ix],y[ix])[0]
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch%4==0 or epoch==64:
            with torch.no_grad():
                mean, _=m(vx,use_propagation=False)
                score=(mean-vy).square().mean().item()
            if score<best:
                best=score
                torch.save(dict(net=m.state_dict(),xm=xm,xs=xs,ym=ym,ys=ys),OUT/'pets.pt')
    return best


def null_fit(m,d,ids,t,epoch):
    s=d['state'][ids,t]
    y=difference(d['end'][ids,t,:,:3],s[:,None,:3])
    yn=difference(d['null_pose'][ids,t],s[:,None,:3])
    a=actions(s).expand(len(s),-1,-1); s=expand_state(s,a)
    keep=1-F.one_hot(d['null_source'][ids,t],3).float()
    ones=torch.ones_like(keep)
    logits=m.edges(s,a)
    mask=ones if epoch<16 else torch.bernoulli(.30+.70*logits.detach().sigmoid())
    loss=(nll(m(s,a,mask),y).mean()+nll(m(s,a,keep),yn).mean())*.5
    if epoch>=12:
        with torch.no_grad():
            full=nll(m(s,a,ones),y)
            nulled=torch.stack([nll(m(s,a,ones*(1-torch.eye(3,device=s.device)[j])),y)-full
                                for j in range(3)],dim=-1)
            pseudo=(nulled>.15).float()
        loss=loss+F.binary_cross_entropy_with_logits(logits,pseudo)
    return loss


def cophy_loss(m,d,ids,t):
    # The AB movie and the reference command are observed. D remains hidden.
    r=d['ref'][ids]
    initial=d['state'][ids,0]
    obs=d['end'][ids,0,r]
    current=d['state'][ids,t]
    pred,stable=cophy_predict(m,initial,obs,r,current)
    y=difference(d['end'][ids,t,:,:3],current[:,None,:3])
    static=y.norm(dim=-1)<.02
    delta=difference(pred,y)
    return delta.square().mean()+.25*F.binary_cross_entropy_with_logits(stable,static.float())


def train(name,device):
    torch.manual_seed(11)
    subset={'cw_only':('causalworld',),'tee_only':('pusht',)}.get(name,TASKS)
    d,v=[panel(sp,device,subset) for sp in ('train','val')]
    if name=='pets':
        return pets_train(d,v,device)
    m=(NullPoint() if name=='ncii' else cophy_class()(7) if name=='cophy' else ContactModel()).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=.0014)
    train_ids=torch.arange(len(d['ref'])*2,device=device)
    best=np.inf
    epochs=85 if name=='cophy' else 72
    for epoch in range(epochs):
        m.train()
        for sample in train_ids[torch.randperm(len(train_ids),device=device)].split(80):
            ids,t=sample//2,sample%2
            if name=='ncii':
                loss=null_fit(m,d,ids,t,epoch)
            elif name=='cophy':
                loss=cophy_loss(m,d,ids,t)
            else:
                loss=graph_loss(m,d,ids,t,name!='null_only')
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),8.)
            opt.step()
        if epoch in (48,64):
            for g in opt.param_groups:g['lr']*=.45
        if epoch%4==3 or epoch==epochs-1:
            m.eval()
            with torch.no_grad():
                scores=[]
                for sample in torch.arange(len(v['ref'])*2,device=device).split(96):
                    ids,t=sample//2,sample%2
                    if name=='ncii':
                        st=v['state'][ids,t]; a=actions(st).expand(len(st),-1,-1)
                        y=difference(v['end'][ids,t,:,:3],st[:,None,:3]);st=expand_state(st,a)
                        # Unsupervised fit checkpoint; calibration thresholds are selected separately.
                        value=nll(m(st,a,torch.ones_like(y)),y).mean()
                    elif name=='cophy':
                        value=cophy_loss(m,v,ids,t)
                    else:
                        value=graph_loss(m,v,ids,t,supervised=name!='null_only')
                    scores.append(value.item())
                score=float(np.mean(scores))
            if score<best and (name!='ncii' or epoch>=20):
                best=score
                torch.save(m.state_dict(),OUT/f'{name}.pt')
    return best


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',nargs='+',default=['ours','ncii','cophy','pets','null_only','cw_only','tee_only'])
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    for name in args.models:
        best=train(name,args.device)
        print(name,best,flush=True)

if __name__=='__main__':
    main()
