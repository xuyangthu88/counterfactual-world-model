import argparse
import numpy as np
import torch
from torch.nn import functional as F
from .model import (OUT,CoPhy,Null,SharedGraph,body_delta,panel,pets_targets,represent,task_value,wrap)


def huber(x,y,beta=.08):
    return F.smooth_l1_loss(x,y,beta=beta)


def losses(name,m,d,ix,epoch):
    s=d['state'][ix]; a=d['cmd'][ix]; y=d['end'][ix]; z=s[:,None].expand_as(y)
    ref=d['ref'][ix]; rows=torch.arange(len(s),device=s.device)
    body=body_delta(z,y)
    if name=='cophy':
        pi,st=m(s,y[rows,ref],a[rows,ref],s,a)
        err=body_delta(z,pi)-body
        err=torch.cat((err[...,:2],wrap(err[...,2:3])),-1)
        return err.square().mean()+.15*F.binary_cross_entropy_with_logits(st,(body.norm(dim=-1)<.025).float())
    if name=='ncii':
        ones=torch.ones_like(body)
        def nll(mask,target):
            mu,l=m(z,a,mask)
            err=mu-target
            return (.5*(err*(-l).exp()).square()+l).sum(-1)
        if not d['nullpose'].numel():return nll(ones,body).mean()
        keep=1-F.one_hot(d['source'][ix],3).float()
        logits=m.edges(z,a)
        valid=ones if epoch<16 else torch.bernoulli(.3+.7*logits.detach().sigmoid())
        loss=.5*(nll(valid,body).mean()+nll(keep,body_delta(z,d['nullpose'][ix])).mean())
        if epoch>=12:
            with torch.no_grad():
                full=nll(ones,body)
                gap=torch.stack([nll(ones*(1-torch.eye(3,device=s.device)[j]),body)-full for j in range(3)],-1)
            loss=loss+F.binary_cross_entropy_with_logits(logits,(gap>.15).float())
        return loss
    k=(d['physics'][ix].float()-1)[:,None].expand_as(a[...,0])
    u=(d['noise'][ix]/.04)[:,None].expand_as(k)
    pred,edges,peak,rank=m(z,a,k,u)
    residual=body_delta(z,pred)-body
    residual=torch.cat((residual[...,:2],wrap(residual[...,2:3])),-1)
    pair=residual-residual[rows,ref][:,None]
    ut=task_value(z,y,d['peak'][ix])
    loss=(huber(residual,torch.zeros_like(residual))+.65*huber(pair,torch.zeros_like(pair))+
          .2*F.binary_cross_entropy_with_logits(edges,d['edge'][ix].float())+
          .25*huber(pred[...,3:9],y[...,3:9])+.16*huber(pred[...,14:19],y[...,14:19])+
          .35*F.mse_loss(pred[...,12],y[...,12])+.25*F.mse_loss(peak,d['peak'][ix])+
          .2*huber(rank,ut)+.6*huber(rank-rank.mean(1,keepdim=True),ut-ut.mean(1,keepdim=True)))
    if d['nullpose'].numel():
        keep=1-F.one_hot(d['source'][ix],3).float()
        null=m(z,a,k,u,keep)[0]
        loss=loss+.25*huber(body_delta(z,null),body_delta(z,d['nullpose'][ix]))
    return loss


def pets_fit(d,v,device):
    from mbrl.models import GaussianMLP
    def flatten(ds):
        z=ds['state'][:,None].expand_as(ds['end'])
        return represent(z,ds['cmd']).flatten(0,1),pets_targets(z,ds['end'],ds['peak']).flatten(0,1)
    x,y=flatten(d);vx,vy=flatten(v)
    xm,xs=x.mean(0),x.std(0).clamp_min(.025)
    ym,ys=y.mean(0),y.std(0).clamp_min(.025)
    x=(x-xm)/xs;y=(y-ym)/ys
    vx=(vx-xm)/xs;vy=(vy-ym)/ys
    net=GaussianMLP(x.shape[1],y.shape[1],device,num_layers=3,ensemble_size=5,hid_size=144)
    opt=torch.optim.AdamW(net.parameters(),lr=.0012)
    bag=torch.randint(len(x),(5,len(x)),device=device);best=np.inf
    for epoch in range(95):
        for idx in torch.randperm(len(x),device=device).split(512):
            j=bag[:,idx];loss=net.loss(x[j],y[j])[0]
            opt.zero_grad();loss.backward();opt.step()
        if epoch%4==2:
            with torch.no_grad():
                mu,_=net(vx,use_propagation=False);score=(mu-vy).square().mean().item()
            if score<best:
                best=score;torch.save(dict(net=net.state_dict(),xm=xm,xs=xs,ym=ym,ys=ys),OUT/'pets.pt')
    return best


def main(args):
    torch.manual_seed(11);torch.set_num_threads(2)
    d,v=[panel(sp,args.device) for sp in ('train','val')]
    name=args.name
    if name=='pets':
        print(name,pets_fit(d,v,args.device),flush=True);return
    net={'graph':SharedGraph,'cophy':CoPhy,'ncii':Null}[name]().to(args.device)
    epochs={'graph':145,'ncii':90,'cophy':100}[name]
    opt=torch.optim.AdamW(net.parameters(),lr=.0014)
    best=np.inf
    for epoch in range(epochs):
        net.train()
        for ix in torch.randperm(len(d['ref']),device=args.device).split(64 if name=='cophy' else 96):
            loss=losses(name,net,d,ix,epoch)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),6.);opt.step()
        if epoch in (epochs*2//3,epochs*5//6):
            for g in opt.param_groups:g['lr']*=.45
        if epoch%4==3 or epoch==epochs-1:
            with torch.no_grad():
                net.eval()
                val=np.mean([losses(name,net,v,ix,epoch).item() for ix in torch.arange(len(v['ref']),device=args.device).split(48)])
            if val<best and (name!='ncii' or epoch>=19):
                best=val;torch.save(net.state_dict(),OUT/f'{name}.pt')
    print(name,best,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('name',choices=['graph','pets','cophy','ncii']);ap.add_argument('--device',default='cuda')
    main(ap.parse_args())
