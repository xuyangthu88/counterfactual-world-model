"""Fitted finite-horizon pose value for looking past an intermediate push."""
import argparse
from copy import deepcopy

import torch
from torch.nn import functional as F

from .models import OUT, mlp, panel, utility


def features(xy, yaw, cover, s):
    dx, dy = (s[...,9:11] - xy).unbind(-1)
    ca, sa = yaw.cos(), yaw.sin()
    turn = yaw - s[...,11]
    domain = s[...,13].expand_as(yaw)
    return torch.stack((dx, dy, ca*dx + sa*dy, ca*dy - sa*dx, turn.sin(),
                        turn.cos(), ca, sa, cover, domain), dim=-1)


def state_inputs(z):
    u = utility(z[...,12], z[...,:2], z[...,2], z)
    x = features(z[...,:2], z[...,2], z[...,12], z)
    done = z[...,12] > (.9 + .05*z[...,13])
    return x, u, done


def continuation(net, xy, yaw, cover, s):
    return net(features(xy, yaw, cover, s)).squeeze(-1).clamp(-.5,1.5)


def fit(device):
    torch.manual_seed(11)
    def inputs(split):
        d = panel(split,device)
        s = d['state'].flatten(0,1)
        end = d['end'].flatten(0,1)
        return state_inputs(s), state_inputs(end)
    train, val = inputs('train'), inputs('val')
    previous = None
    for h in range(1,4):
        def targets(data):
            (_,now,_),(x,u,done) = data
            with torch.no_grad():
                gain = 0. if previous is None else previous(x).squeeze(-1).clamp(-.5,1.5)
                v = u + .8 * (~done) * gain
                return v.max(-1).values-now
        y,vy = targets(train),targets(val)
        x = train[0][0]
        model = mlp(x.shape[-1],1,96).to(device)
        opt = torch.optim.AdamW(model.parameters(),lr=.0015)
        best = float('inf')
        for epoch in range(100):
            for ix in torch.randperm(len(x),device=device).split(128):
                loss = F.smooth_l1_loss(model(x[ix]).squeeze(-1),y[ix],beta=.1)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch%4 == 3:
                with torch.no_grad():
                    score = (model(val[0][0]).squeeze(-1)-vy).abs().mean().item()
                if score < best:
                    best,previous = score,deepcopy(model).eval()
        print(h,best,flush=True)
    torch.save(previous.state_dict(),OUT/'value.pt')


def load_value(device):
    net=mlp(10,1,96).to(device)
    net.load_state_dict(torch.load(OUT/'value.pt',weights_only=True,map_location=device))
    return net.eval()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args=ap.parse_args()
    torch.set_num_threads(2)
    fit(args.device)
