import argparse
import json
import numpy as np
import torch
from whatif.crossbench.evaluate import bmr,classify,metrics
from .model import OUT,panel,grid,wrap,log_measured,posterior,unpack,task_value
from .planning import Models,graph_scores,menus,mechanical


def contrast(z,b):
    delta=z[...,:3]-b[...,:3]
    return torch.cat((delta[...,:2],wrap(delta[...,2:3])),-1)


def positions(d):
    s=d['state'];a=d['cmd'];ref=d['ref'];i=torch.arange(len(s),device=s.device)
    obs=d['end'][i,ref]
    alt=torch.arange(12,device=s.device)[None]!=ref[:,None]
    return s,a,i,ref,obs,alt


@torch.no_grad()
def calibrate(model,device):
    d=panel('val',device);s,a,i,r,obs,alt=positions(d)
    z,edges,_,_=grid(model.graph,s,a)
    target=contrast(d['end'],obs[:,None])
    cf=contrast(z,z[i,r][:,None]);opts={}
    for sig in (.055,.10,.18,.32):
        w=posterior(s,obs,z,r,sig)
        y=(cf[...,:2]*w[:,None,:,None]).sum(2)-target[...,:2]
        g=[y[s[:,13]==k].norm(dim=-1).mean() for k in (0,1)]
        opts[sig]=float(sum(g))
    sigma=min(opts,key=opts.get)
    w=posterior(s,obs,z,r,sigma)
    p=(edges.sigmoid()*w[:,None,:,None]).sum(2)
    n=model.ncii.edges(s[:,None].expand(-1,12,-1),a).sigmoid()
    y=d['edge'][alt].cpu().numpy().ravel()
    def th(score):
        pred=score[alt].cpu().numpy().ravel()
        return float(min(np.linspace(.03,.97,96),key=lambda t:bmr(y,pred,t)))
    co,_=model.cophy(s,obs,a[i,r],s,a)
    choice={}
    for name,b in [('observed',obs),('predicted',co[i,r])]:
        err=contrast(co,b[:,None])-target
        choice[name]=float(err[alt][:,:2].norm(dim=-1).mean())
    cfg=dict(sigma=sigma,graph=th(p),ncii=th(n),cophy=min(choice,key=choice.get))
    (OUT/'calibration.json').write_text(json.dumps(cfg,indent=2)+'\n');print(cfg,flush=True)


@torch.no_grad()
def test(model,device):
    cfg=json.loads((OUT/'calibration.json').read_text());out={};arr={}
    for task in ('causalworld','pusht'):
        d=panel('test',device,(task,));s,a,i,r,obs,alt=positions(d);target=contrast(d['end'],obs[:,None])
        z,edges,peak,rank=grid(model.graph,s,a);w=posterior(s,obs,z,r,cfg['sigma'])
        true=target[alt].reshape(len(s),11,3)
        labels=d['edge'][alt].reshape(len(s),11,3).cpu().numpy()
        base=d['edge'][i,r][:,None].cpu().numpy()
        validation=panel('val',device,(task,))
        source_rates=validation['edge'].float().mean((0,1)).cpu().numpy()
        # Predict by source identity alone to expose easy always-pusher/never-wall cases.
        reference=np.broadcast_to(source_rates,labels.shape)
        summary={'source_only':{'edges':classify(labels,reference,.5)}};pack={}
        u=task_value(s[:,None],d['end'],d['peak'])
        arr[task+'_labels']=labels;arr[task+'_seed']=d['seed'].cpu().numpy();arr[task+'_stratum']=d['protocol'].cpu().numpy()
        p=(edges.sigmoid()*w[:,None,:,None]).sum(2)
        n=model.ncii.edges(s[:,None].expand_as(d['end']),a).sigmoid()
        for name,score in [('graph',p),('ncii',n)]:
            v=score[alt].reshape(len(s),11,3).cpu().numpy()
            result=classify(labels,v,cfg[name])
            old=score[i,r][:,None].cpu().numpy()
            changed=(labels!=base)
            changes=(v>=cfg[name])!=(old>=cfg[name])
            result['switch_rate']=float(changed.mean())
            result['switch_recall']=float(changes[changed].mean())
            result['switch_precision']=float(changed[changes].mean()) if changes.any() else 0.
            result['source_prevalence']=labels.mean((0,1)).tolist()
            summary[name]={'edges':result};arr[task+'_'+name+'_edges']=v
        contrast_graph=contrast(z,z[i,r][:,None])
        uniform=torch.ones_like(w)/w.shape[1]
        noobs=(log_measured(s).softmax(-1)).repeat_interleave(3,-1)/3
        for name,mass in [('graph',w),('uniform',uniform),('pre_probe',noobs)]:
            pack[name]=(contrast_graph,mass)
        co,_=model.cophy(s,obs,a[i,r],s,a)
        anchored=obs if cfg['cophy']=='observed' else co[i,r]
        pack['cophy']=(contrast(co,anchored[:,None])[:,:,None],s.new_ones(len(s),1))
        # Common random draws and member across each action pair, without factual PETS reweighting.
        state=s[:,None].expand_as(d['end']);flat=state.reshape(-1,19)
        mu,sd=model.predict(flat,a.reshape(-1,7))
        torch.manual_seed(11)
        eps=torch.randn(5,1,1,3,16,device=device)
        dy=mu.reshape(5,len(s),12,1,16)+sd.reshape(5,len(s),12,1,16)*eps
        zz,pk=unpack(state[:,:,None].expand(-1,-1,15,-1),dy.permute(1,2,0,3,4).reshape(len(s),12,15,16))
        pack['pets']=(contrast(zz,zz[i,r][:,None]),uniform)
        direct={'graph':(z,w),'pets':(zz,uniform),'cophy':(co[:,:,None],s.new_ones(len(s),1))}
        for name,(candidates,mass) in direct.items():
            raw_pose=candidates[alt].reshape(len(s),11,-1,candidates.shape[-1])[...,:3]
            actual_pose=d['end'][alt].reshape(len(s),11,19)[...,:3]
            summary.setdefault(name,{})['candidate_pose']=metrics(raw_pose,actual_pose,mass)[0]
        for name,(cf,mass) in pack.items():
            vals,per=metrics(cf[alt].reshape(len(s),11,-1,3),true,mass)
            summary.setdefault(name,{})['prediction']=vals
            arr[task+'_'+name+'_error']=per.cpu().numpy()
        # Withhold the factual action from all rank diagnostics; compare actual utilities of chosen siblings.
        graphu=.55*task_value(s[:,None,None],z,peak)+.45*rank
        co_abs=state.clone();co_abs[...,:3]=co
        from whatif.crossbench.models import geometry_cover
        cov=geometry_cover(co[...,:2],co[...,2],state);co_abs[...,12]=cov
        ranks={'graph':(graphu*w[:,None]).sum(2),'uniform':(graphu*uniform[:,None]).sum(2),
               'pets':task_value(state[:,:,None],zz,pk).mean(2),'cophy':task_value(state,co_abs,cov),
               'physics':mechanical(state,a)}
        optimum=u.masked_fill(~alt,-torch.inf).max(1).values
        for name,scores in ranks.items():
            selected=scores.masked_fill(~alt,-torch.inf).argmax(1)
            reg=optimum-u[i,selected]
            summary.setdefault(name,{})['regret']=float(reg.mean())
            arr[task+'_'+name+'_regret']=reg.cpu().numpy()
        strata=d['protocol'].cpu().numpy()
        checks={}
        for group in np.unique(strata):
            chosen=strata==group
            entry={}
            for name in ('graph','pre_probe','uniform','pets','cophy'):
                entry[name]=float(arr[task+'_'+name+'_error'][chosen].mean())
            checks[str(int(group))]=entry
        summary['xy_by_protocol']=checks
        anchor=s.new_tensor([0.,.5*.08**.1,.08**.1,(.08**.1+.55**.1)/2,.55**.1])
        if task=='pusht':
            true_rho=s.new_tensor([0.,.25**.1])[d['protocol']]
            pre=log_measured(s).softmax(-1)
            after=w.reshape(-1,5,3).sum(-1)
            summary['retention']={}
            for key,prob in [('uniform',torch.ones_like(pre)/5),('pre_probe',pre),('post_probe',after)]:
                for g in (0,1):
                    subset=d['protocol']==g
                    val=(prob@anchor)[subset]
                    summary['retention'][key+str(g)]=dict(estimate=float(val.mean()),mae=float((val-true_rho[subset]).abs().mean()))
        out[task]=summary
    (OUT/'metrics.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(OUT/'paired.npz',**arr)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['calibrate','test']);ap.add_argument('--device',default='cuda')
    args=ap.parse_args();torch.set_num_threads(2)
    (calibrate if args.mode=='calibrate' else test)(Models(args.device),args.device)
