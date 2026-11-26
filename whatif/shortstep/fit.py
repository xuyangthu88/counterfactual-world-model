"""Train the signed message dynamics on normal and edited native trajectories."""
import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from whatif.effects.collect import DOMAINS, OUT as PREVIOUS, PROTOCOL as EARLIER
from whatif.effects.data import pair_ids, subset
from whatif.effects.fit import forecast_loss, paired_actions, validation
from whatif.effects.model import average, build_world
from whatif.mechanism.model import labeled
from .augment import rotate_layouts
from .collect import OUT, PROTOCOL
from .data import load, paired_removal
from .model import ShortWorld


def step_delta(y):
    return torch.diff(y,dim=-3,prepend=torch.zeros_like(y[...,:1,:,:]))


def smooth_error(e):
    return F.smooth_l1_loss(e,torch.zeros_like(e),beta=.25,reduction='none')


def native_loss(o,d):
    err=o['pose']-d['native_pose']
    pos=err[...,:3]
    i,j=paired_actions(d['a'])
    obj=d['role'].sum(-1)[:,None,None,:,None]
    move=d['role'][:,None,None,:,0:1]
    delta=(pos[:,i]-pos[:,j])
    speed=step_delta(delta)/d['native_widths'][:,None,:,None,None].sqrt()
    native=d['native_edge'].to(pos)
    edge=labeled(d['role'])[:,None,None]
    ce=F.binary_cross_entropy_with_logits(o['edge'],native,reduction='none')
    change=(native[:,:,1:]-native[:,:,:-1]).abs()
    loss=(.36*average(pos.square(),obj+2*move)+.09*average(err[...,3:].square(),obj+2*move)
          +.55*average(delta.square(),move)+.45*average(speed.square(),move)
          +.31*average(ce,edge)+.17*average(ce[:,:,1:],edge*change))
    if 'native_reward' in d:
        e=o['reward']-d['native_reward']
        loss=loss+.18*(e.square().mean()+(e[:,i]-e[:,j]).square().mean())
    return loss


def edit_loss(pred,target,removed,d,width):
    i,j=paired_actions(d['a'])
    p,q=pair_ids(d['x'].size(1),d['x'].device)
    edge=removed[:,0,0]
    ids=torch.arange(d['x'].size(1),device=edge.device)
    downstream=((ids!=p[edge,None])&(ids!=q[edge,None]))[:,None,None,:,None]
    obj=d['role'].sum(-1)[:,None,None,:,None]*(1+.5*downstream)
    e=pred-target
    effect_weight=1+2*(target.norm(dim=-1,keepdim=True)>.10)
    direction=lambda x: x/(.10+x.norm(dim=-1,keepdim=True))
    coupling=e[:,i]-e[:,j]
    native_scale=width[:,None,:,None,None].sqrt()
    return (.68*average(smooth_error(e),obj*effect_weight)
            +.52*average(smooth_error(coupling),obj)
            +.34*average(smooth_error(step_delta(e)/native_scale),obj)
            +.16*average((direction(pred)-direction(target)).square(),obj*effect_weight))


def objective(model,d,soft,dense):
    removed,target=paired_removal(d)
    o=model(d,removed,soft=soft,native=dense)
    loss=forecast_loss(o,d)
    if dense:
        o=o['native']
        loss=loss+native_loss(o,d)
        width=d['native_widths']
    else:
        target=target.index_select(2,d['end_idx'][0])
        width=d['widths']
    return loss+edit_loss(o['edge_effect'][:,:,0],target,removed,d,width)


def model_domains(kind):
    return DOMAINS[:2] if kind=='short_unseen' else DOMAINS


def train(kind):
    dev='cuda'
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(EARLIER['seed'])
    domains=model_domains(kind)
    init='motion_unseen-calibrated' if kind=='short_unseen' else 'motion_once-calibrated'
    parent=build_world(init).to(dev)
    params=torch.load(PREVIOUS/f'{init}.pt',map_location=dev)['model']
    parent.load_state_dict(params)
    train_data={x:load(x,'train',dev) for x in domains}
    valid_data={x:load(x,'val',dev,labels=False) for x in domains}
    scale=validation(parent,valid_data)
    del parent
    model=ShortWorld().to(dev)
    model.load_state_dict(params,strict=False)
    del params
    opt=torch.optim.AdamW([
        dict(params=[v for k,v in model.named_parameters() if not k.startswith('vector.')],lr=.00065),
        dict(params=model.vector.parameters(),lr=.00110),
    ],weight_decay=.005,fused=True)
    steps=PROTOCOL['budgets'][kind]
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00014)
    best=float('inf')
    history=[]
    start=time.monotonic()
    OUT.mkdir(exist_ok=True)
    for step in range(1,steps+1):
        model.train()
        soft=max(0., .30*(1-step/300))
        opt.zero_grad(set_to_none=True)
        values=[]
        for d in train_data.values():
            data=subset(d,torch.randint(len(d['x']),(24,),device=dev))
            if kind=='short_rotated':
                data=rotate_layouts(data,PROTOCOL['rotated_layout_probability'])
            loss=objective(model,data,soft,kind!='short_coarse')
            (loss/len(domains)).backward()
            values.append(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),4.)
        opt.step()
        sched.step()
        if step%100==0 or step==steps:
            metrics=validation(model,valid_data)
            criterion=float(np.mean([v/scale[k] for k,v in metrics.items()]))
            if criterion<best:
                best=criterion
                torch.save(dict(model=model.state_dict(),step=step,domains=domains,
                                initialization=init),OUT/f'{kind}.pt')
            record=dict(step=step,loss=float(torch.stack(values).mean()),validation=metrics,initial_validation=scale,
                        criterion=criterion,seconds=round(time.monotonic()-start,1))
            history.append(record)
            print(kind,record,flush=True)
            (OUT/f'{kind}-fit.json').write_text(json.dumps(history,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=PROTOCOL['budgets'])
    train(p.parse_args().kind)
