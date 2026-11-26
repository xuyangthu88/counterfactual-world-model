"""Score one shared checkpoint, its matched continuation, and the fixed references."""
import argparse
import json

import torch

from whatif.effects.collect import DOMAINS
from whatif.effects.evaluate import evaluate
from whatif.shortstep.data import load
from whatif.shortstep.evaluate import native_metrics, pushing_edits, restore as restore_short
from .fit import OUT, PARENT, PROTOCOL, build

EARLIER = ('shared-calibrated','motion_once-calibrated',PARENT,'short_unseen-calibrated')
TRAINED = tuple(PROTOCOL['budgets'])
REFERENCES = ('cophynet','pets','ncii')


def restore(name,device):
    kind=name.removesuffix('-calibrated')
    if kind not in TRAINED:
        return restore_short(name,device)
    model=build(kind).to(device)
    model.load_state_dict(torch.load(OUT/f'{name}.pt',map_location=device)['model'])
    return model.eval()


def selection():
    validation={}
    for kind in TRAINED:
        history=json.loads((OUT/f'{kind}-fit.json').read_text())
        best=min(history,key=lambda row:row['criterion'])
        validation[kind+'-calibrated']=best
    reference=validation['continued-calibrated']['initial_validation']
    validation[PARENT]=dict(validation=reference,initial_validation=reference,criterion=1.)
    eligible=[PARENT,'continued-calibrated','cumulative-calibrated']
    selected=min(eligible,key=lambda name:validation[name]['criterion'])
    heldout='cumulative_unseen-calibrated'
    heldout_parent='short_unseen-calibrated'
    reference=validation[heldout]['initial_validation']
    validation[heldout_parent]=dict(validation=reference,initial_validation=reference,criterion=1.)
    choice=dict(selected=selected,heldout_selected=min(
        (heldout_parent,heldout),key=lambda name:validation[name]['criterion']),validation=validation)
    (OUT/'selection.json').write_text(json.dumps(choice,indent=2)+'\n')
    return choice


def inputs(domain,split,device):
    return load(domain,split,device,labels=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    choice=selection()
    fresh=PROTOCOL['test_split']
    models=[*EARLIER,*(kind+'-calibrated' for kind in TRAINED)]
    results=evaluate([*models,*REFERENCES],('test',fresh),args.device,'metrics.json',
                     load_data=inputs,restore_model=restore,output_dir=OUT)
    for domain in DOMAINS:
        data=load(domain,fresh,args.device)
        for name in models:
            model=restore(name,args.device)
            row=results[domain][fresh][name]
            if domain!='cophy':
                row['edit']=pushing_edits(model,data,domain)
            if name not in EARLIER[:2]:
                row['native']=native_metrics(model,data,domain)
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print('selected',choice['selected'],flush=True)
