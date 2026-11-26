"""Train a shared world model on physical transitions and paired graph edits."""
import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F
from whatif.mechanism.model import labeled

from .collect import DOMAINS, OUT, PROTOCOL, MOTION_PROTOCOL
from .data import load, omission, pair_ids, subset
from .model import build_world, average, contrast

UNSEEN = ('unseen_domain','motion_unseen')


def paired_actions(x):
    return torch.triu_indices(x.size(1), x.size(1), offset=1, device=x.device)


def normal_nll(y, mean, scale, w):
    view = (len(w), *([1]*(mean.ndim-2)), w.size(-1))
    logp = w.clamp_min(1.e-9).log().reshape(view) - scale.log() - .5*((y[...,None]-mean)/scale).square()
    return -logp.logsumexp(-1)


def directed_truth(d):
    n = d['role'].size(1)
    i, j = pair_ids(n, d['x'].device)
    vector = d['off']
    b, k, p, t = vector.shape[:4]
    magnitude = vector.norm(dim=-1)
    diag = torch.arange(p, device=i.device)
    result = vector.new_zeros(b,k,t,n,n)
    result[...,i,j] = magnitude[:,:,diag,:,j].permute(1,2,3,0)
    result[...,j,i] = magnitude[:,:,diag,:,i].permute(1,2,3,0)
    return result


def objective(model, d, intervention, soft):
    removed = omission(d) if intervention and 'off' in d else None
    o = model(d, removed, soft=soft)
    i, j = paired_actions(d['a'])
    obj = d['role'].sum(-1)[:,None,None,:,None]
    movable = d['role'][:,None,None,:,0:1]
    err = o['pose']-d['pose']
    de = err[:,i]-err[:,j]
    receiver = (d['a'][:,i,:,0]+d['a'][:,j,:,0] == 0)[:, :, None, :, None]
    edge = labeled(d['role'])[:,None,None]
    contact = F.binary_cross_entropy_with_logits(o['edge'], d['edge'], reduction='none')
    change = (d['edge'][:,:,1:]-d['edge'][:,:,:-1]).abs()
    discord = (d['edge'][:,i]-d['edge'][:,j]).abs()
    affected = discord.any(-2) | discord.any(-1)
    weights = movable*(1 + .7*receiver + .8*affected[...,None])
    last = d['widths'].new_tensor([.7, .7, 1., 1.2, 1.4])[None,None,:,None,None]
    xyz, orient = err[...,:3], err[...,3:]
    diff_xyz, diff_orient = de[...,:3], de[...,3:]
    loss = (average(xyz.square()+.15*xyz.abs(), (obj+3*movable)*last)
            + .30*average(orient.square(), (obj+3*movable)*last)
            + .85*average(diff_xyz.square()+.15*diff_xyz.abs(), weights*last)
            + .20*average(diff_orient.square(), weights*last)
            + .58*average(contact, edge)
            + .25*average(contact[:,:,1:], edge*change)
            + .20*average(.5*(contact[:,i]+contact[:,j]), edge*discord)
            + .10*average(F.binary_cross_entropy_with_logits(o['graph'], d['fe'], reduction='none'),
                          edge[:,0,0])
            + .10*average(F.smooth_l1_loss(o['innovation'], torch.zeros_like(o['innovation']),
                                          beta=.15, reduction='none'), d['role'].sum(-1)[:,None,:,None])
            + .002*o['kl'])
    pose_dist = tuple(v[...,:3,:] for v in o['p_dist'])
    mean, scale, w = contrast(o['pose'][...,:3].detach(), pose_dist, o['w'], i,j)
    target = d['pose'][:,i,...,:3]-d['pose'][:,j,...,:3]
    loss = loss + .14*average(normal_nll(target, mean, scale, w), movable)
    if 'off' in d:
        true = directed_truth(d)
        loss = loss + .40*average(F.binary_cross_entropy_with_logits(o['causal'], (true>.1).to(true),
                                                                     reduction='none'), edge)
        loss = loss + .10*average(F.smooth_l1_loss(o['strength'], true.log1p(), beta=.15, reduction='none'), edge)
    else:
        reward = o['reward']-d['reward']
        mean, scale, w = contrast(o['reward'].detach(), o['r_dist'], o['w'], i,j)
        target = d['reward'][:,i]-d['reward'][:,j]
        loss = loss + (reward.square().mean() + .8*(reward[:,i]-reward[:,j]).square().mean()
                       + .10*normal_nll(target, mean, scale, w).mean())
    if removed is not None:
        gather = removed[...,None,None,None].expand(-1,-1,-1,*d['off'].shape[-3:])
        target = torch.gather(d['off'], 2, gather)
        error = F.smooth_l1_loss(o['edge_effect'], target, beta=.25, reduction='none')
        mask = obj.unsqueeze(2)*(1+2*(target.norm(dim=-1,keepdim=True)>.1))
        loss = loss + 1.10*average(error, mask*last.unsqueeze(2))
    return loss


@torch.no_grad()
def validation(model, sets):
    model.eval()
    metrics = {}
    for domain, d in sets.items():
        rms = []
        for ix in torch.arange(len(d['x']),device=d['x'].device).split(32):
            part = subset(d, ix)
            pred = model(part)['pose'][:,:,-1]
            i,j = paired_actions(pred)
            err = pred-part['pose'][:,:,-1]
            diff = (err[:,i,...,:3]-err[:,j,...,:3]).square().sum(-1)
            rms.append(average(diff, part['role'][:,None,:,0]).item())
        metrics[domain] = float(np.sqrt(np.average(rms, weights=[len(x) for x in torch.arange(len(d['x'])).split(32)])))
    return metrics


def train(kind, steps=None):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(PROTOCOL['seed'])
    dev = 'cuda'
    domains = DOMAINS[:2] if kind in UNSEEN else DOMAINS
    training = {dom: load(dom, 'train', dev) for dom in domains}
    valid = {dom: load(dom, 'val', dev) for dom in domains}
    net = build_world(kind).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=.0013, weight_decay=.005, fused=True)
    steps = steps or (PROTOCOL['budgets'] | MOTION_PROTOCOL['budgets'])[kind]
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=.0002)
    best = float('inf')
    history = []
    start = time.monotonic()
    OUT.mkdir(exist_ok=True)
    for step in range(1, steps+1):
        net.train()
        soft = max(0., .85*(1-(step-1)/500))
        rows = [subset(d, torch.randint(len(d['x']), (32,), device=dev)) for d in training.values()]
        loss = torch.stack([objective(net,d,kind!='no_edge_intervention',soft) for d in rows]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 4.)
        opt.step()
        schedule.step()
        if step%130==0 or step == steps:
            metrics = validation(net, valid)
            criterion = float(np.mean(list(metrics.values())))
            if criterion < best:
                best = criterion
                torch.save(dict(step=step, model=net.state_dict(), domains=domains), OUT/f'{kind}.pt')
            record = dict(step=step, loss=loss.item(), validation=metrics, criterion=criterion,
                          seconds=round(time.monotonic()-start, 1))
            history.append(record)
            print(record, flush=True)
            (OUT/f'{kind}-fit.json').write_text(json.dumps(history, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=('shared','no_edge_intervention','unseen_domain',*MOTION_PROTOCOL['architecture']))
    p.add_argument('--steps', type=int)
    args = p.parse_args()
    train(args.kind, args.steps)
