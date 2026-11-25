"""One downstream value readout for both mechanisms and all planners."""
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from whatif.crossbench.models import mlp
from .model import DATA,OUT,body_delta,rotate,task_value


def value_features(s):
    goal=body_delta(s,s[...,9:12])
    rel=rotate(s[...,3:9].reshape(*s.shape[:-1],3,2)-s[...,:2].unsqueeze(-2),-s[...,2:3])
    return torch.cat((s[...,:2],s[...,2:3].cos(),s[...,2:3].sin(),goal[...,:2],goal[...,2:3].sin(),
                      goal[...,2:3].cos(),rel.flatten(-2),s[...,12:19]),-1)


def fit(device):
    torch.manual_seed(11)
    def data(split):
        states=[];future=[]
        for task in ('causalworld','pusht'):
            with np.load(DATA/f'return-{task}-{split}.npz') as z:
                states.append(z['state']);future.append(z['future'])
        s=torch.tensor(np.concatenate(states),device=device);y=torch.tensor(np.concatenate(future),device=device)
        native=task_value(s[:,None],y,y[...,12])
        score=.20*native[:,0]+.30*native[:,1]+.50*native[:,2]
        reached=(y[...,12]>(.9+.05*y[...,13])).any(1)
        return value_features(s),score+.50*reached-task_value(s,s,s[...,12])
    x,y=data('train');vx,vy=data('val');net=mlp(21,1,96).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.0015);best=np.inf
    for ep in range(135):
        for ii in torch.randperm(len(x),device=device).split(160):
            loss=F.smooth_l1_loss(net(x[ii]).squeeze(-1),y[ii],beta=.09)
            opt.zero_grad();loss.backward();opt.step()
        if ep in (70,105):
            for g in opt.param_groups:g['lr']*=.45
        if ep%3==2:
            with torch.no_grad():score=(net(vx).squeeze(-1)-vy).abs().mean().item()
            if score<best:
                best=score;torch.save(net.state_dict(),OUT/'value.pt')
    print('value',best,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();torch.set_num_threads(2);fit(a.device)
