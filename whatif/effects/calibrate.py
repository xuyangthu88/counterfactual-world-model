"""Fit both marginal and paired distributions with the graph and means frozen."""
import argparse
import json

import torch

from .collect import DOMAINS, OUT, PROTOCOL, MOTION_PROTOCOL
from .data import load, subset
from .fit import UNSEEN, normal_nll, paired_actions
from .model import build_world, contrast, average


def loss(model, d):
    o=model(d)
    i,j=paired_actions(d['a'])
    w=o['w']

    def train_law(truth, mean, dist, mask):
        offset, scale, load=dist
        marginal=normal_nll(truth,mean[...,None]+offset,(scale.square()+load.square()).sqrt(),w)
        params=contrast(mean,dist,w,i,j)
        joint=normal_nll(truth[:,i]-truth[:,j],*params)
        return average(marginal,mask)+average(joint,mask)

    pm=o['pose'][...,:3]
    pd=tuple(v[...,:3,:] for v in o['p_dist'])
    pos=train_law(d['pose'][...,:3],pm,pd,d['role'][:,None,None,:,0:1])
    if 'off' in d:
        return pos
    return pos+.5*train_law(d['reward'],o['reward'],o['r_dist'],pm.new_ones(1))


def fit(kind):
    dev='cuda'
    checkpoint=torch.load(OUT/f'{kind}.pt',map_location=dev)
    model=build_world(kind).to(dev).eval()
    model.load_state_dict(checkpoint['model'])
    model.requires_grad_(False)
    for head in (model.position_dist,model.reward_dist,model.regime):
        head.requires_grad_(True)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.0007,fused=True)
    domains=DOMAINS[:2] if kind in UNSEEN else DOMAINS
    train={dom:load(dom,'train',dev) for dom in domains}
    val={dom:load(dom,'val',dev) for dom in domains}
    history=[]
    best=float('inf')
    for step in range(1,801):
        rows=[subset(d,torch.randint(len(d['x']),(48,),device=dev)) for d in train.values()]
        obj=torch.stack([loss(model,d) for d in rows]).mean()
        opt.zero_grad(set_to_none=True)
        obj.backward()
        opt.step()
        if step%100==0:
            with torch.no_grad():
                scores={name:float(loss(model,d)) for name,d in val.items()}
            criterion=sum(scores.values())/len(scores)
            history.append(dict(step=step,validation_nll=scores,criterion=criterion))
            print(kind, history[-1], flush=True)
            if criterion<best:
                best=criterion
                torch.save(dict(checkpoint,model=model.state_dict(),calibration_step=step),
                           OUT/f'{kind}-calibrated.pt')
            (OUT/f'{kind}-calibration.json').write_text(json.dumps(history,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=('shared','no_edge_intervention','unseen_domain',*MOTION_PROTOCOL['architecture']))
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(PROTOCOL['seed'])
    torch.set_float32_matmul_precision('high')
    fit(args.kind)
