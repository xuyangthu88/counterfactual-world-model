"""Calibrate the forecast law without moving the learned graph or the means."""
import argparse
import json

import torch

from whatif.effects.calibrate import loss
from whatif.effects.collect import PROTOCOL as EARLIER
from whatif.effects.data import subset
from .collect import OUT, PROTOCOL
from .data import load
from .fit import model_domains
from .model import ShortWorld


def calibrate(kind, model=None, *, domains=None, directory=OUT, steps=PROTOCOL['calibration_steps']):
    device='cuda'
    domains=model_domains(kind) if domains is None else domains
    checkpoint=torch.load(directory/f'{kind}.pt',map_location=device)
    model=(ShortWorld() if model is None else model).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval().requires_grad_(False)
    for head in (model.position_dist,model.reward_dist,model.regime):
        head.requires_grad_(True)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.00055,fused=True)
    train={d:load(d,'train',device,labels=False) for d in domains}
    valid={d:load(d,'val',device,labels=False) for d in domains}
    log=[]
    best=float('inf')
    for step in range(1,steps+1):
        rows=[subset(d,torch.randint(len(d['x']),(40,),device=device)) for d in train.values()]
        objective=torch.stack([loss(model,d) for d in rows]).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if step%100==0 or step==steps:
            with torch.no_grad():
                values={d:float(loss(model,b)) for d,b in valid.items()}
            value=sum(values.values())/len(values)
            if value<best:
                best=value
                torch.save(dict(checkpoint,model=model.state_dict(),calibration_step=step),
                           directory/f'{kind}-calibrated.pt')
            log.append(dict(step=step,validation=values,value=value))
            print(kind,log[-1],flush=True)
            (directory/f'{kind}-calibration.json').write_text(json.dumps(log,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=PROTOCOL['budgets'])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(EARLIER['seed'])
    calibrate(p.parse_args().kind)
