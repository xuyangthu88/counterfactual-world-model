"""Compare the additional displacement readout with equal-budget continued training."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from whatif.effects.collect import DOMAINS, PROTOCOL as BASE, ROOT
from whatif.effects.data import subset
from whatif.effects.fit import validation
from whatif.shortstep.augment import rotate_layouts
from whatif.shortstep.calibrate import calibrate
from whatif.shortstep.collect import OUT as PREVIOUS, PROTOCOL as SHORT
from whatif.shortstep.data import load
from whatif.shortstep.fit import objective
from whatif.shortstep.model import ShortWorld
from .model import CumulativeWorld

OUT = ROOT/'results/cumulative'
PROTOCOL = json.loads(Path(__file__).with_name('protocol.json').read_text())
PARENT = 'short_rotated-calibrated'


def build(kind):
    return ShortWorld() if kind=='continued' else CumulativeWorld()


def domains(kind):
    return DOMAINS[:2] if kind=='cumulative_unseen' else DOMAINS


def fit(kind):
    dev='cuda'
    include=domains(kind)
    initialization='short_unseen-calibrated' if kind=='cumulative_unseen' else PARENT
    params=torch.load(PREVIOUS/f'{initialization}.pt',map_location=dev)['model']
    parent=ShortWorld().to(dev)
    parent.load_state_dict(params)
    train={d:load(d,'train',dev) for d in include}
    val={d:load(d,'val',dev,labels=False) for d in include}
    scale=validation(parent,val)
    del parent
    model=build(kind).to(dev)
    state=model.load_state_dict(params,strict=kind=='continued')
    assert not state.unexpected_keys and all(k.startswith('displacement.') for k in state.missing_keys)
    del params
    torch.manual_seed(BASE['seed'])
    rates={'backbone': .00048, 'displacement': .00115}
    groups={
        'backbone': [v for k,v in model.named_parameters() if not k.startswith('displacement.')],
        'displacement': [v for k,v in model.named_parameters() if k.startswith('displacement.')],
    }
    optimizer=torch.optim.AdamW([dict(params=p,lr=rates[name]) for name,p in groups.items() if p],
                                weight_decay=.005,fused=True)
    steps=PROTOCOL['budgets'][kind]
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.00010)
    best=float('inf')
    log=[]
    start=time.monotonic()
    OUT.mkdir(exist_ok=True)
    for step in range(1,steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risks=[]
        for d in train.values():
            batch=subset(d,torch.randint(len(d['x']),(24,),device=dev))
            batch=rotate_layouts(batch,SHORT['rotated_layout_probability'])
            loss=objective(model,batch,0.,True)
            (loss/len(include)).backward()
            risks.append(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),4.)
        optimizer.step()
        schedule.step()
        if step%100==0 or step==steps:
            metrics=validation(model,val)
            criterion=float(np.mean([v/scale[k] for k,v in metrics.items()]))
            if criterion<best:
                best=criterion
                torch.save(dict(step=step,model=model.state_dict(),domains=include,
                                initialization=initialization),OUT/f'{kind}.pt')
            log.append(dict(step=step,validation=metrics,initial_validation=scale,criterion=criterion,
                            loss=float(torch.stack(risks).mean()),seconds=round(time.monotonic()-start,1)))
            print(kind,log[-1],flush=True)
            (OUT/f'{kind}-fit.json').write_text(json.dumps(log,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=PROTOCOL['budgets'])
    p.add_argument('--calibrate',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(BASE['seed'])
    if args.calibrate:
        calibrate(args.kind,build(args.kind),domains=domains(args.kind),directory=OUT,
                  steps=PROTOCOL['calibration_steps'])
    else:
        fit(args.kind)
