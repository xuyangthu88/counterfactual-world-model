"""Separate same-event action contrasts, prospective contact graphs, and next-event actions."""
import argparse
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .models import (OUT, TASKS, ContactModel, NullPoint, actions, belief, cophy_class,
                     cophy_predict, difference, fresh, geometry_cover, panel, particles, utility)


def load_model(name,device):
    m=(NullPoint() if name=='ncii' else cophy_class()(7) if name=='cophy' else ContactModel()).to(device)
    m.load_state_dict(torch.load(OUT/f'{name}.pt',weights_only=True,map_location=device))
    return m.eval()


def bmr(y,p,th):
    y=y.astype(bool);label=p>=th
    return float(.5*((~label[y]).mean()+label[~y].mean()))


def classify(y,p,th):
    y=y.ravel().astype(bool);p=p.ravel(); z=p>=th
    return dict(balanced_error=bmr(y,p,th),positive_rate=float(y.mean()),
                precision=float(y[z].mean()) if z.any() else 0.,recall=float(z[y].mean()),
                fpr=float(z[~y].mean()),ap=float(average_precision_score(y,p)),
                auroc=float(roc_auc_score(y,p)),edges=len(y))


def pose_effect(delta,y):
    return difference(delta, y)


def metrics(pred,y,w):
    # scene, alternative, particle, pose
    pos=(pred[...,:2]*w[:,None,:,None]).sum(2)
    residual=(pos-y[...,:2]).norm(dim=-1)
    angles=pred[...,2]
    yaw=torch.atan2((angles.sin()*w[:,None]).sum(2),(angles.cos()*w[:,None]).sum(2))
    yawerr=torch.atan2((yaw-y[...,2]).sin(),(yaw-y[...,2]).cos()).abs().rad2deg()
    x=pred[...,:2].contiguous()
    dist=torch.cdist(x,x,compute_mode='donot_use_mm_for_euclid_dist')
    es=((x-y[:,:,None,:2]).norm(dim=-1)*w[:,None]).sum(2)-.5*(dist*w[:,None,:,None]*w[:,None,None,:]).sum((2,3))
    sse=residual.square().mean()
    sst=(y[...,:2]-y[...,:2].mean((0,1))).square().sum(-1).mean()
    return dict(xy=float(residual.mean()),xy_rmse=float(sse.sqrt()),r2=float(1-sse/sst),
                yaw_deg=float(yawerr.mean()),energy=float(es.mean())),residual.mean(1)


def posterior(d,delta,sig):
    i=torch.arange(len(d['ref']),device=delta.device)
    obs=difference(d['end'][i,0,d['ref'],:3],d['state'][:,0,:3])
    return belief(delta,d['ref'],obs,delta.new_tensor([sig,sig,1.6*sig]))


def action_scores(s,delta,cover,w,rule,blend=0.,value=None,strength=0.):
    xy=s[:,None,None,:2]+delta[...,:2]
    yaw=s[:,None,None,2]+delta[...,2]
    if blend:
        cover = torch.lerp(cover, geometry_cover(xy, yaw, s[:,None,None]), blend)
    if rule=='distance':
        v=-(xy-s[:,None,None,9:11]).norm(dim=-1)
    else:
        v=utility(cover,xy,yaw,s[:,None,None])
        if strength:
            from .value import continuation
            gain=continuation(value,xy,yaw,cover,s[:,None,None])
            v=v+strength*(cover<=(.9+.05*s[:,None,None,13]))*gain
        if rule=='paired':
            # Same physical particle for the candidate and nominal reference.
            v=v+.22*torch.tanh((v-v[:,0:1])/.20)
    return (v*w[:,None]).sum(2)


def real_score(d,choices):
    i=torch.arange(len(choices),device=choices.device)
    y=d['end'][i,1,choices];s=d['state'][:,1]
    return utility(y[:,12],y[:,:2],y[:,2],s)


@torch.no_grad()
def evaluate(device):
    names=['ours','null_only','cw_only','tee_only','ncii','cophy']
    models={key:load_model(key,device) for key in names}
    val=panel('val',device)
    settings={}
    grids=np.linspace(.015,.985,130)
    for name,m in models.items():
        if name=='cophy':
            i=torch.arange(len(val['ref']),device=device)
            s0=val['state'][:,0]; r=val['ref']; obs=val['end'][i,0,r]
            delta=cophy_predict(m,s0,obs,r,s0)[0]
            observed=difference(obs[:,:3],s0[:,:3])
            true=difference(val['end'][:,0,:,:3],obs[:,None,:3])
            anchors={'observed':observed, 'predicted':delta[i,r]}
            settings[name]={'anchor':min(anchors,key=lambda k:
                (difference(delta,anchors[k][:,None])[...,:2]-true[...,:2]).norm(dim=-1).mean().item())}
            continue
        s=val['state'][:,0]
        if name=='ncii':
            a=actions(s).expand(len(s),-1,-1)
            p=m.edges(s[:,None].expand(-1,a.shape[1],-1),a).sigmoid()
            cfg={}
        else:
            delta,_,edges=particles(m,s)
            # Single-domain checkpoints must calibrate without seeing the held-out domain.
            mask=(s[:,13]==(name=='tee_only')) if name.endswith('_only') and name!='null_only' else torch.ones(len(s),device=device,dtype=torch.bool)
            i=torch.arange(len(s),device=device)
            true=difference(val['end'][:,0,:,:3],val['end'][i,0,val['ref'],:3][:,None])
            cf=difference(delta,delta[i,val['ref']][:,None])
            def err(sig):
                w=posterior(val,delta,sig)
                return (((cf[...,:2]*w[:,None,:,None]).sum(2)-true[...,:2]).norm(dim=-1)[mask]).mean().item()
            sigma=min([.055,.11,.22,.44],key=err)
            w=posterior(val,delta,sigma)
            p=(edges.sigmoid()*w[:,None,:,None]).sum(2)
            d1,c1,_=particles(m,val['state'][:,1])
            rules=['paired'] if name=='ours' else ['task','paired']
            def achieved(candidate):
                rule, blend = candidate
                a=action_scores(val['state'][:,1],d1,c1,fresh(w),rule,blend).argmax(1)
                return real_score(val,a)[mask].mean().item()
            rule,blend=max(((r,b) for r in rules for b in (0.,.5,1.)),key=achieved)
            cfg=dict(sigma=sigma,rule=rule,blend=blend)
        mask=(s[:,13]==(name=='tee_only')) if name in ('cw_only','tee_only') else torch.ones(len(s),device=device,dtype=torch.bool)
        alt=torch.arange(len(actions(s)),device=device)[None]!=val['ref'][:,None]
        labels=val['edge'][:,0][mask][alt[mask]].cpu().numpy().ravel()
        scores=p[mask][alt[mask]].cpu().numpy().ravel()
        cfg['threshold']=float(min(grids,key=lambda th:bmr(labels,scores,th)))
        settings[name]=cfg
    output={};stored={}
    for task in TASKS:
        d=panel('test',device,(task,)); s=d['state'][:,0]
        i=torch.arange(len(s),device=device);ref=d['ref'];obs=d['end'][i,0,ref]
        a=actions(s).expand(len(s),-1,-1)
        alt=torch.arange(a.shape[1],device=device)[None]!=ref[:,None]
        true=difference(d['end'][:,0,:,:3],obs[:,None,:3])[alt].reshape(len(i),-1,3)
        labels=d['edge'][:,0][alt].reshape(len(i),-1,3)
        out={}
        for name,m in models.items():
            entry={}
            if name=='ncii':
                p=m.edges(s[:,None].expand(-1,a.shape[1],-1),a).sigmoid()
            elif name=='cophy':
                delta=cophy_predict(m,s,obs,ref,s)[0]
                anchor=delta[i,ref] if settings[name]['anchor']=='predicted' else difference(obs[:,:3],s[:,:3])
                cf=difference(delta,anchor[:,None])[alt].reshape(len(i),-1,1,3)
                w=s.new_ones(len(i),1)
            else:
                delta,_,edges=particles(m,s)
                w=posterior(d,delta,settings[name]['sigma'])
                p=(edges.sigmoid()*w[:,None,:,None]).sum(2)
                cf=difference(delta,delta[i,ref][:,None])[alt].reshape(len(i),-1,9,3)
                d1,c1,_=particles(m,d['state'][:,1])
                opts={}
                for rule in ('distance','task','paired'):
                    chosen=action_scores(d['state'][:,1],d1,c1,fresh(w),rule,settings[name]['blend']).argmax(1)
                    y=d['end'][i,1,chosen]
                    opts[rule]=float(y[:,12].mean())
                entry['offline_next_coverage']=opts
                if name=='ours':
                    uniform=torch.ones_like(w)/w.shape[1]
                    entry['prior_prediction']=metrics(cf,true,uniform)[0]
                    prior_p=edges.sigmoid().mean(2)[alt].reshape(len(i),-1,3).cpu().numpy()
                    entry['graph_before_observation']=classify(labels.cpu().numpy(),prior_p,settings[name]['threshold'])
            if name!='cophy':
                prob=p[alt].reshape(len(i),-1,3)
                y=labels.cpu().numpy();v=prob.cpu().numpy()
                entry['graph']=classify(y,v,settings[name]['threshold'])
                entry['per_source_positive']=y.mean(axis=(0,1)).tolist()
                entry['switch_rate']=float((labels!=d['edge'][i,0,ref][:,None]).float().mean())
                ref_true=d['edge'][i,0,ref][:,None].cpu().numpy()
                ref_prob=p[i,ref][:,None].cpu().numpy()
                changes={}
                for kind,truth,score in [('new',y & ~ref_true,v*(1-ref_prob)),
                                         ('broken',~y & ref_true,(1-v)*ref_prob)]:
                    changes[kind]={'ap':float(average_precision_score(truth.ravel(),score.ravel())),
                                   'base_rate':float(truth.mean())}
                entry['edge_changes']=changes
                stored[f'{task}_{name}_edges']=v
            if name!='ncii':
                entry['prediction'],per=metrics(cf,true,w)
                stored[f'{task}_{name}_error']=per.cpu().numpy()
            out[name]=entry
        stored[f'{task}_labels']=labels.cpu().numpy()
        stored[f'{task}_seed']=d['seed'].cpu().numpy()
        stored[f'{task}_stratum']=d['protocol'].cpu().numpy()
        output[task]=out
    (OUT/'calibration.json').write_text(json.dumps(settings,indent=2)+'\n')
    (OUT/'metrics.json').write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(OUT/'test_arrays.npz',**stored)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    torch.set_num_threads(2);evaluate(args.device)
