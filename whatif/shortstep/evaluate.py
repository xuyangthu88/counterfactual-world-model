"""Evaluate the frozen standard baselines and matched graph edits on the same draws."""
import argparse
import json

import torch
from torch.nn import functional as F

from whatif.effects.collect import DOMAINS
from whatif.effects.data import UNIT, subset
from whatif.effects.evaluate import (
    combine, evaluate, restore_world, root_mean, standardized_graph,
)
from whatif.effects.fit import paired_actions
from whatif.effects.model import average
from whatif.mechanism.model import labeled
from .collect import OUT, PROTOCOL
from .data import load, paired_removal
from .fit import step_delta
from .model import ShortWorld

NEW=tuple(PROTOCOL['budgets'])
PAST=('shared-calibrated','motion_once-calibrated','motion_unseen-calibrated')


def restore(name,device):
    if name.removesuffix('-calibrated') not in NEW:
        return restore_world(name,device)
    m=ShortWorld().to(device)
    m.load_state_dict(torch.load(OUT/f'{name}.pt',map_location=device)['model'])
    return m.eval()


def checkpoint_selection():
    variants=('shortstep','short_coarse','short_rotated')
    val={k:min(json.loads((OUT/f'{k}-fit.json').read_text()),key=lambda r:r['criterion']) for k in variants}
    chosen=min(val,key=lambda k:val[k]['criterion'])
    val=dict(selected=chosen,validation=val)
    (OUT/'selection.json').write_text(json.dumps(val,indent=2)+'\n')
    return chosen


@torch.no_grad()
def pushing_edits(net,d,domain):
    removed,truth=paired_removal(d)
    pred=[]
    for ix in torch.arange(len(d['x']),device=d['x'].device).split(24):
        out=net(subset(d,ix),removed[ix])
        pred.append(out['edge_effect'][:,:,0])
    y=truth.index_select(2,d['end_idx'][0])
    e=torch.cat(pred)-y
    obj=d['role'][:,None,None,:,0].bool().expand_as(y[...,0])
    magnitude=y.norm(dim=-1)
    significant=(magnitude>.10)&obj
    i,j=paired_actions(d['a'])
    affected=(significant[:,i]|significant[:,j])
    unit=UNIT[domain]
    direction=F.cosine_similarity(torch.cat(pred),y,dim=-1)
    return dict(edit_rmse=unit*root_mean(e,obj), paired_edit_rmse=unit*root_mean(e[:,i]-e[:,j],obj[:,i]),
                active_edit_rmse=unit*root_mean(e,significant), active_edits=int(significant.sum()),
                active_pairs=int(affected.sum()), active_edit_cosine=float(average(direction,significant)))


@torch.no_grad()
def native_metrics(net,d,domain):
    chunks=[]
    for ix in torch.arange(len(d['x']),device=d['x'].device).split(24):
        o=net(subset(d,ix),native=True)['native']
        chunks.append({k:o[k] for k in ('pose','edge')})
    o=combine(chunks)
    error=(o['pose']-d['native_pose'])[...,:3]
    i,j=paired_actions(d['a'])
    paired=error[:,i]-error[:,j]
    mover=d['role'][:,None,None,:,0]
    unit=UNIT[domain]
    y=d['native_edge'].bool()
    mask=labeled(d['role'])[:,None,None].bool().expand_as(y)
    p=o['edge'].sigmoid()
    onset=mask[:,:,1:]&~y[:,:,:-1]
    return dict(native_increment_diff_rmse=unit*root_mean(step_delta(paired),mover),
                native_delta_rmse=[unit*root_mean(paired[:,:,t],mover[:,:,0])
                                   for t in range(paired.size(2))],
                native_contact_ap=standardized_graph(y,p,mask)['ap'],
                native_new_contact_ap=standardized_graph(y[:,:,1:],p[:,:,1:],onset)['ap'])


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--device',default='cuda')
    args=a.parse_args()
    selected=checkpoint_selection()
    methods=[*PAST,*(f'{k}-calibrated' for k in NEW),'cophynet','pets','ncii']
    fresh=PROTOCOL['test_split']
    def inputs(dom,split,dev):
        return load(dom,split,dev,labels=False)

    result=evaluate(methods,('test',fresh),args.device,'metrics.json',load_data=inputs,
                    restore_model=restore,output_dir=OUT)
    for dom in DOMAINS:
        data=load(dom,fresh,args.device)
        for name in methods[:-3]:
            net=restore(name,args.device)
            stat=result[dom][fresh][name]
            if dom!='cophy':
                stat['edit']=pushing_edits(net,data,dom)
            if name not in PAST:
                stat['native']=native_metrics(net,data,dom)
            print('short step scores',dom,name,stat.get('edit'),stat.get('native'),flush=True)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('selected',selected,flush=True)
