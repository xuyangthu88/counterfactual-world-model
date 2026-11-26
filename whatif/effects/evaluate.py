"""Score held-out native action pairs, interaction edits, and predicted contacts."""
import argparse
import json
import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_curve

from whatif.mechanism.model import labeled
from .baseline import cophy_class, cophy_predict, pets_predict, pets_restore
from .collect import DOMAINS, OUT, PROTOCOL
from .data import load, pair_ids, subset, UNIT
from .fit import directed_truth, paired_actions
from .model import build_world, average, contrast


def root_mean(error, mask):
    return math.sqrt(float(average(error.square().sum(-1), mask)))


def gaussian_abs(x, sd):
    z = x/sd
    return sd*math.sqrt(2/math.pi)*torch.exp(-z.square()/2)+x*torch.erf(z/math.sqrt(2))


def crps(truth, loc, scale, w):
    scale = scale.clamp_min(1.e-6)
    w = w.reshape(len(w), *([1]*(loc.ndim-2)), -1)
    term1 = (w*gaussian_abs(loc-truth[...,None],scale)).sum(-1)
    m12 = loc.unsqueeze(-1)-loc.unsqueeze(-2)
    s12 = (scale.square().unsqueeze(-1)+scale.square().unsqueeze(-2)).sqrt()
    term2 = (w.unsqueeze(-1)*w.unsqueeze(-2)*gaussian_abs(m12,s12)).sum((-1,-2))
    return term1-.5*term2


def empirical_crps(truth, draw):
    first = (draw-truth[...,None]).abs().mean(-1)
    draw = draw.sort(-1).values
    n = draw.size(-1)
    ranks = torch.arange(1,n+1,device=draw.device)
    second = (draw*(2*ranks-n-1)).sum(-1)/n**2
    return first-second


def standardized_graph(truth, score, mask, threshold=None):
    y = truth[mask].cpu().numpy().astype(bool)
    p = score[mask].cpu().numpy()
    if len(np.unique(y))!=2:
        return dict(ap=None, ber=None, cut=threshold, positives=int(y.sum()), count=len(y))
    if threshold is None:
        fp,tp,cuts = roc_curve(y,p,drop_intermediate=False)
        threshold = float(cuts[np.argmax(tp-fp)])
        if not math.isfinite(threshold):
            threshold = .5
    on = p>=threshold
    return dict(ap=float(average_precision_score(y,p)), ber=float((on[~y].mean()+(~on[y]).mean())*.5),
                cut=threshold, positives=int(y.sum()), count=len(y))


def contact_graph(d, o, cuts):
    mask = labeled(d['role'])[:,None,None].bool().expand_as(d['edge'])
    y = d['edge'].bool()
    p = o['edge'].sigmoid()
    stat={}
    stat['contact']=standardized_graph(y,p,mask,cuts.get('contact'))
    stat['new_contact']=standardized_graph(y[:,:,1:],p[:,:,1:],mask[:,:,1:]&~y[:,:,:-1],
                                           cuts.get('new_contact'))
    stat['lost_contact']=standardized_graph(~y[:,:,1:],1-p[:,:,1:],mask[:,:,1:]&y[:,:,:-1],
                                            cuts.get('lost_contact'))
    i,j = paired_actions(d['a'])
    both_absent = ~y[:,i,:-1] & ~y[:,j,:-1]
    changed = y[:,i,1:] != y[:,j,1:]
    actor = d['a'][:,i,:,0].bool() | d['a'][:,j,:,0].bool()
    via_others = ~actor[:,:,:,None] & ~actor[:,:,None,:]
    unique = torch.ones_like(y[0,0,0]).triu(1)
    eligible = mask[:,i,1:] & both_absent & changed & via_others[:,:,None] & unique
    sign = y[:,i,1:].float()-y[:,j,1:].float()
    gap = ((p[:,i,1:]-p[:,j,1:])*sign)[eligible]
    on, off = (p[:,i,1:]>=.5), (p[:,j,1:]>=.5)
    split = ((on==y[:,i,1:])&(off==y[:,j,1:]))[eligible]
    stat['remote_contact_disagreement']=dict(count=int(eligible.sum()),
        layouts=int(eligible.flatten(1).any(-1).sum()),
        order=float(((gap>0).float()+.5*(gap==0)).mean()) if gap.numel() else None,
        hard=float(split.float().mean()) if split.numel() else None)
    if 'off' in d:
        true = directed_truth(d)[:,:,-1]>.10
        actual=o.get('causal',o['edge'])[:,:,-1].sigmoid()
        stat['directed']=standardized_graph(true,actual,mask[:,:,-1],cuts.get('directed'))
    return stat


def prediction(d, o, domain, flavor):
    unit = UNIT[domain]
    i,j = paired_actions(d['a'])
    mover = d['role'][:,None,:,0].bool()
    pos = o['pose']
    truth = d['pose'][...,:pos.size(-1)]
    err = pos-truth
    diff = err[:,i]-err[:,j]
    y = truth[:,i]-truth[:,j]
    end = diff[:,:,-1,:,:3]
    stat=dict(
        position_rmse=unit*root_mean(err[:,:,-1,:,:3],mover),
        delta_rmse=unit*root_mean(end,mover),
        pose_rmse_by_window=[unit*root_mean(err[:,:,t,:,:3],mover) for t in range(err.size(2))],
        delta_rmse_by_window=[unit*root_mean(diff[:,:,t,:,:3],mover) for t in range(err.size(2))],
    )
    if err.size(-1)==9:
        stat.update(orientation_rmse=root_mean(err[:,:,-1,:,3:],mover))
    if 'off' in d:
        n = err.size(-2)
        depth = torch.arange(n,device=err.device)
        a = d['a'][:,i,:,0].argmax(-1)
        b = d['a'][:,j,:,0].argmax(-1)
        remote = ((depth-a[...,None]).abs()>=2)&((depth-b[...,None]).abs()>=2)&mover
        stat.update(remote_delta_rmse=unit*root_mean(end,remote),remote_action_pairs=int(remote.sum()))
    large = y[:,:,-1,:,:3].norm(dim=-1)
    measured = mover.expand_as(large)
    tail = (large >= large[measured].quantile(.95)) & measured
    sq = end.square().sum(-1)
    stat.update(largest_effect_rmse=unit*root_mean(end,tail),
                largest_effect_error_share=float((sq*tail).sum()/(sq*measured).sum().clamp_min(1.e-12)),
                largest_effect_pairs=int(tail.sum()))
    discrepancy=(d['edge'][:,i]!=d['edge'][:,j]).any(2).any(-2) | (d['edge'][:,i]!=d['edge'][:,j]).any(2).any(-1)
    changed=mover&discrepancy
    stat.update(contact_disagreement_delta_rmse=unit*root_mean(end,changed) if changed.any() else None,
                contact_disagreement_action_pairs=int(changed.sum()))
    if o.get('reward') is not None and 'off' not in d:
        w = d['widths']/d['widths'].sum(1,keepdim=True)
        r = ((o['reward']-d['reward'])*w[:,None]).sum(-1)
        stat.update(reward_rmse=float(r.square().mean().sqrt()),
                    reward_delta_rmse=float((r[:,i]-r[:,j]).square().mean().sqrt()))

    pair_target = y[:,:,-1,:,:3]
    def score(x):
        return unit*float(average(x, mover[...,None]))
    if flavor=='ours':
        for independent in (False,True):
            gaussian = tuple(v[:,:,-1,:,:3] for v in o['p_dist'])
            args = contrast(pos[:,:,-1,:,:3],gaussian,o['w'],i,j,independent)
            suffix = '_independent' if independent else ''
            stat['delta_crps'+suffix]=score(crps(pair_target,*args))
            if 'off' not in d:
                args = contrast(o['reward'],o['r_dist'],o['w'],i,j,independent)
                rd = d['reward'][:,i]-d['reward'][:,j]
                stat['reward_delta_crps'+suffix]=float((crps(rd,*args)*w[:,None]).sum(-1).mean())
    elif flavor=='pets':
        draw = o['draws'][:,:,:,-1,:,:3].movedim(2,-1)
        a,b = draw[:,i],draw[:,j]
        stat['delta_crps']=score(empirical_crps(pair_target,(a.unsqueeze(-1)-b.unsqueeze(-2)).flatten(-2)))
        if 'off' not in d:
            draw = o['reward_draws'].movedim(2,-1)
            a,b = draw[:,i],draw[:,j]
            rd = d['reward'][:,i]-d['reward'][:,j]
            stat['reward_delta_crps']=float((empirical_crps(rd,(a.unsqueeze(-1)-b.unsqueeze(-2)).flatten(-2))
                                            *w[:,None]).sum(-1).mean())
    else:
        stat['delta_crps']=score(end.abs())
    return stat


def combine(parts):
    result={}
    for key in parts[0]:
        item=parts[0][key]
        if isinstance(item, tuple):
            result[key]=tuple(torch.cat([p[key][i] for p in parts]) for i in range(len(item)))
        elif item is not None:
            result[key]=torch.cat([p[key] for p in parts])
    return result


@torch.no_grad()
def edits(model,d):
    n=d['role'].size(1)
    ps,pd=pair_ids(n,d['x'].device)
    parts=[]
    for ix in torch.arange(len(d['x']),device=d['x'].device).split(12):
        part=subset(d,ix)
        all_pairs=torch.arange(len(ps),device=ix.device)[None,None].expand(len(ix),d['a'].size(1),-1)
        parts.append(model(part,all_pairs)['edge_effect'])
    pred=torch.cat(parts)
    y=d['off']
    present=d['role'].sum(-1).bool()
    mask=(present[:,ps]&present[:,pd])[:,None,:,None,None]&present[:,None,None,None,:]
    significant=(y.norm(dim=-1)>.1)&mask
    ids=torch.arange(n,device=pred.device)
    indirect=(ids!=ps[:,None])&(ids!=pd[:,None])
    remote=significant&indirect[None,None,:,None,:]
    direction=torch.nn.functional.cosine_similarity(pred,y,dim=-1)
    i,j=paired_actions(d['a'])
    e=pred-y
    return dict(edited_trajectory_rmse=root_mean(pred-y,mask),
                paired_edit_rmse=root_mean(e[:,i]-e[:,j],mask),
                affected_trajectory_rmse=root_mean(pred-y,significant),
                downstream_trajectory_rmse=root_mean(pred-y,remote),
                downstream_effect_recall=float(average((pred.norm(dim=-1)>.1).float(),remote)),
                downstream_direction_cosine=float(average(direction,remote)),
                affected=int(significant.sum()), downstream=int(remote.sum()))


def restore_world(name, device):
    checkpoint=torch.load(OUT/f'{name}.pt',map_location=device)
    model=build_world(name).to(device)
    model.load_state_dict(checkpoint['model'])
    return model.eval()


@torch.no_grad()
def evaluate(names, splits, device, destination, *, load_data=load, restore_model=restore_world,
             output_dir=OUT):
    torch.set_num_threads(2)
    variants=('motion_once','motion_twice')
    if all(f'{kind}-calibrated' in names for kind in variants):
        val={kind:min(json.loads((OUT/f'{kind}-fit.json').read_text()),key=lambda r:r['criterion'])
             for kind in variants}
        selected=min(val,key=lambda kind:val[kind]['criterion'])
        (OUT/'motion-selection.json').write_text(json.dumps(dict(selected=selected,validation=val),indent=2)+'\n')
    results={}
    for domain in DOMAINS:
        sets={s:load_data(domain,s,device) for s in ('val',*splits)}
        for name in names:
            if name=='cophynet':
                model=cophy_class()().to(device)
                model.load_state_dict(torch.load(OUT/f'cophynet-{domain}.pt',map_location=device))
                def run(d):
                    return {'pose':cophy_predict(model,d)['pose'], 'reward':None}
            elif name=='pets':
                model=pets_restore(domain,device)
                # Upstream TS∞ uses the default generator for member assignment.
                torch.manual_seed(345)
                rng=torch.Generator(device=device).manual_seed(345)
                def run(d):
                    return pets_predict(model,d,rng)
            elif name=='ncii':
                from .ncii import restore as ncii_restore, predict as ncii_predict
                model=ncii_restore(domain,device)
                def run(d):
                    return ncii_predict(model,d)
            else:
                model=restore_model(name,device)
                def run(d):
                    o=model(d)
                    return {k:o[k] for k in ('pose','reward','edge','causal','w','p_dist','r_dist')}
            model.eval()
            graph=name not in ('pets','cophynet')
            cuts={}
            for split,d in sets.items():
                if split=='val' and not graph and 'val' not in splits:
                    continue
                parts=[run(subset(d,ix)) for ix in torch.arange(len(d['x']),device=device).split(24)]
                o=combine(parts)
                stat=contact_graph(d,o,cuts) if graph else {}
                if split=='val':
                    cuts={k:v.get('cut') for k,v in stat.items()}
                if split not in splits:
                    continue
                if name!='ncii':
                    stat.update(prediction(d,o,domain,name if not graph else 'ours'))
                if 'off' in d and graph and name!='ncii':
                    stat.update(edits(model,d))
                stat['layouts']=len(d['x'])
                results.setdefault(domain,{}).setdefault(split,{})[name]=stat
                print(domain,split,name,stat.get('delta_rmse'),stat.get('contact',{}).get('ap'),flush=True)
    for splits_result in results.values():
        for panel in splits_result.values():
            for model in panel.values():
                if 'contact' not in model or 'delta_rmse' not in model or not all(x in panel for x in ('cophynet','pets','ncii')):
                    continue
                baseline=min(('cophynet','pets'),key=lambda method:panel[method]['delta_rmse'])
                edge='directed' if 'directed' in model else 'contact'
                model['comparison']=dict(
                    prediction_reference=baseline,
                    delta_rmse_reduction_pct=100*(1-model['delta_rmse']/panel[baseline]['delta_rmse']),
                    graph_reference='ncii',
                    ber_reduction_pct=100*(1-model[edge]['ber']/panel['ncii'][edge]['ber']),
                )
    (output_dir/destination).write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--models',nargs='+',default=(
        'shared-calibrated','no_edge_intervention-calibrated','unseen_domain-calibrated','cophynet','pets','ncii'))
    parser.add_argument('--splits',nargs='+',default=('test','transfer'))
    parser.add_argument('--output',default='metrics.json')
    a=parser.parse_args()
    evaluate(a.models,a.splits,a.device,a.output)
