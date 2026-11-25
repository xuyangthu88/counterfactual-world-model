"""Train both short-stroke dynamics models on the same simulated contact menu."""
from copy import deepcopy
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .models import OUT, mlp


class SurfaceGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=mlp(20,5)
    def forward(self,x,k):
        out=self.net(torch.cat((x,k[...,None]),dim=-1))
        prior=torch.cat((x[...,14:16],x[...,16:17]/.47),dim=-1)*x[...,17:18]
        pred=out[...,:3]+prior
        return pred*out[...,3:4].sigmoid(),out[...,3],out[...,4].sigmoid()


def data(split,device,name):
    z=np.load(Path('data/crossbench')/f'{name}-{split}.npz')
    return [torch.from_numpy(z[k]).to(device) for k in ('x','y','physics','edge')]


def loss(m,d,ids):
    x,y,k,edge=[a[ids] for a in d]
    pred,logit,cov=m(x,(k.float()-1)[:,None].expand_as(x[...,0]))
    scale=x.new_tensor([1.,1.,.47]);e=(pred-y[...,:3])*scale
    center=e[:,1:2]
    return (F.smooth_l1_loss(e,torch.zeros_like(e),beta=.06)+
            .5*F.smooth_l1_loss(e-center,torch.zeros_like(e),beta=.06)+
            .12*F.binary_cross_entropy_with_logits(logit,edge.float())+
            .35*F.mse_loss(cov,y[...,3]))


def fit(device,name):
    from mbrl.models import GaussianMLP
    torch.manual_seed(11)
    d,v=data('train',device,name),data('val',device,name)
    m=SurfaceGraph().to(device);opt=torch.optim.AdamW(m.parameters(),lr=.0015)
    best=float('inf')
    for ep in range(160):
        for ids in torch.randperm(len(d[0]),device=device).split(32):
            l=loss(m,d,ids);opt.zero_grad();l.backward();opt.step()
        if ep in (80,120):
            for g in opt.param_groups:g['lr']*=.5
        if ep%4==3:
            with torch.no_grad():err=loss(m,v,torch.arange(len(v[0]),device=device)).item()
            if err<best:best=err; graph=deepcopy(m).eval()
    params={f'g{i}':p.detach().cpu().numpy() for i,p in enumerate(graph.net.parameters())}
    x=d[0].reshape(-1,19);y=d[1].reshape(-1,4);vx=v[0].reshape(-1,19);vy=v[1].reshape(-1,4)
    xm,xs,ym,ys=x.mean(0),x.std(0).clamp_min(.05),y.mean(0),y.std(0).clamp_min(.03)
    x=(x-xm)/xs;y=(y-ym)/ys;vx=(vx-xm)/xs;vy=(vy-ym)/ys
    pets=GaussianMLP(19,4,device,num_layers=3,ensemble_size=5,hid_size=128)
    opt=torch.optim.AdamW(pets.parameters(),lr=.001)
    bag=torch.randint(len(x),(5,len(x)),device=device);best=float('inf')
    for ep in range(110):
        for ids in torch.randperm(len(x),device=device).split(384):
            idx=bag[:,ids];l=pets.loss(x[idx],y[idx])[0];opt.zero_grad();l.backward();opt.step()
        if ep%4==3:
            with torch.no_grad():
                mu,_=pets(vx,use_propagation=False);err=(mu-vy).square().mean().item()
            if err<best:best=err; ens=deepcopy(pets).eval()
    for key,t in {'xm':xm,'xs':xs,'ym':ym,'ys':ys}.items():params[key]=t.cpu().numpy()
    for i,layer in enumerate([*[layer[0] for layer in ens.hidden_layers],ens.mean_and_logvar]):
        params[f'ew{i}']=layer.weight.detach().cpu().numpy()
        params[f'eb{i}']=layer.bias.detach().cpu().numpy()
    params['max_logvar']=ens.max_logvar.detach().cpu().numpy()
    params['min_logvar']=ens.min_logvar.detach().cpu().numpy()
    # Compare portable inference to the upstream network before running simulator workers.
    from .surface_numpy import graph_predict,ensemble_predict
    xx=v[0][0]
    with torch.no_grad():
        gp,_,_=graph(xx,torch.zeros_like(xx[:,0]))
        pm,pv=ens((xx-xm)/xs,use_propagation=False)
    ng=graph_predict(params,xx.cpu().numpy())
    np.testing.assert_allclose(ng[:,1,:3],gp.cpu().numpy(),atol=2e-5)
    nm,nv=ensemble_predict(params,xx.cpu().numpy())
    np.testing.assert_allclose(nm, (pm*ys+ym).cpu().numpy(),atol=2e-5)
    np.testing.assert_allclose(nv, ((.5*pv).exp()*ys).cpu().numpy(),atol=2e-5)
    np.savez_compressed(OUT/f'{name}-models.npz',**params)
    print('portable graph and Gaussian ensemble verified',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--name',default='surface',choices=['surface','tracked']);a=ap.parse_args()
    torch.set_num_threads(2);fit(a.device,a.name)
