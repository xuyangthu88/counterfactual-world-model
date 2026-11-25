import argparse
import json
import numpy as np
import torch
from whatif.crossbench.evaluate import classify,metrics
from whatif.crossbench.models import geometry_cover
from whatif.shared.evaluate import contrast
from whatif.shared.model import posterior,task_value,unpack
from whatif.shared.planning import Models
from .filter import OUT,fit_data,blank,update,gain,correct


@torch.no_grad()
def evaluate(device,split):
    model=Models(device);which=json.loads((OUT/'freeze.json').read_text())['model']
    cfg=json.loads((OUT/'fit.json').read_text())['shortlist'][which]
    thresholds=json.load(open('results/shared/calibration.json'))
    raw={};results={}
    for task in ('causalworld','pusht'):
        d,ev,z,e,p,rank,u,_=fit_data(model,task,split);s=d['state'];a=d['cmd'];ref=d['ref']
        i=torch.arange(len(s),device=device);s0=d['previous'];b=d['b']
        bel=update(blank(s),ev,cfg);w=bel[0].repeat_interleave(3,-1)/3
        g=gain(bel,cfg).repeat_interleave(3,-1)[:,None]
        zz=correct(s[:,None,None],a[:,:,None],z,e,g)
        true=contrast(d['end'],d['end'][i,ref,None])
        alt=torch.arange(12,device=device)[None]!=ref[:,None]
        truth=true[alt].reshape(len(s),11,3); obs=d['end'][i,ref]
        z0=model.graph
        from whatif.shared.model import grid
        root=grid(z0,s0,d['previous_actions'][i,b,None])[0]
        legacy=posterior(s0,s,root,torch.zeros_like(b),.055).reshape(-1,5,3).sum(-1).repeat_interleave(3,-1)/3
        cophy,_=model.cophy(s0,s,d['previous_actions'][i,b],s,a)
        co_same,_=model.cophy(s,obs,a[i,ref],s,a)
        st=s[:,None].expand_as(d['end']);mu,sd=model.predict(st.reshape(-1,19),a.reshape(-1,7))
        torch.manual_seed(11)
        noise=torch.randn(5,1,1,1,16,device=device)*s.new_tensor([-1.22474487,0.,1.22474487])[None,None,None,:,None]
        dy=mu.reshape(5,len(s),12,1,16)+sd.reshape(5,len(s),12,1,16)*noise
        pets,pk=unpack(st[:,:,None].expand(-1,-1,15,-1),dy.permute(1,2,0,3,4).reshape(len(s),12,15,16))
        unif=torch.ones_like(w)/15
        pack={'whatif':(zz,w),'uniform':(z,unif),'legacy':(z,legacy),'pets':(pets,unif),'cophy':(cophy[:,:,None],s.new_ones(len(s),1))}
        result={}
        for name,(pred,mass) in pack.items():
            contrasts=contrast(pred,pred[i,ref,None])
            vals,per=metrics(contrasts[alt].reshape(len(s),11,-1,3),truth,mass)
            az=pred[alt].reshape(len(s),11,-1,pred.shape[-1])[...,:3]
            actual=d['end'][alt].reshape(len(s),11,19)[...,:3]
            result[name]={'future':vals,'absolute':metrics(az,actual,mass)[0]}
            raw[task+'_'+name+'_error']=per.cpu().numpy()
            retrospect_mass=mass
            if name=='whatif':
                er=contrast(pred[i,ref],obs[:,None])
                ll=-.5*(er/s.new_tensor([.055,.055,.088])).square().sum(-1)
                retrospect_mass=(mass.clamp_min(1e-10).log()+ll).softmax(-1)
            if name=='cophy':
                contrasts=contrast(co_same,obs[:,None])[:,:,None]
            result[name]['same_event']=metrics(contrasts[alt].reshape(len(s),11,-1,3),truth,retrospect_mass)[0]
        uhat=.55*task_value(s[:,None,None],zz,p)+.45*rank
        cs=st.clone();cs[...,:3]=cophy
        cv=geometry_cover(cophy[...,:2],cophy[...,2],st);cs[...,12]=cv
        scores={'whatif':(uhat*w[:,None]).sum(2),'uniform':((.55*task_value(s[:,None,None],z,p)+.45*rank)*unif[:,None]).sum(2),
                'pets':task_value(st[:,:,None],pets,pk).mean(2),'cophy':task_value(st,cs,cv)}
        for name,value in scores.items():
            choice=value.argmax(1)
            result[name]['prospective_regret']=float((u.amax(1)-u[i,choice]).mean())
        labels=d['edge'][alt].reshape(len(s),11,3).cpu().numpy()
        base=d['edge'][i,ref,None].cpu().numpy()
        for name,pr in [('whatif',(e.sigmoid()*w[:,None,:,None]).sum(2)),('ncii',model.ncii.edges(st,a).sigmoid())]:
            score=pr[alt].reshape(len(s),11,3).cpu().numpy()
            th=thresholds['graph' if name=='whatif' else 'ncii']
            result.setdefault(name,{})['edges']=classify(labels,score,th)
            flips=(score>=th)!=(pr[i,ref,None].cpu().numpy()>=th)
            gt=base!=labels
            result[name]['edges']['switch_recall']=float(flips[gt].mean())
            result[name]['edges']['switch_rate']=float(gt.mean())
        # Diagnose single-disturbance integration as an explanation on the validation panel.
        point=.55*task_value(s[:,None,None],z,p)+.45*rank
        bank=point.reshape(len(s),12,5,3)
        oldmass=legacy.reshape(len(s),5,3).sum(-1)
        best=(bank.mean(-1)*oldmass[:,None]).sum(-1).argmax(1)
        torch.manual_seed(11)
        disagree=[]
        for trial in range(32):
            which_u=torch.randint(3,(len(s),1,5,1),device=device).expand(-1,12,-1,-1)
            sampled=(bank.gather(-1,which_u).squeeze(-1)*oldmass[:,None]).sum(-1)
            disagree.append((sampled.argmax(1)!=best).float().mean())
        result['sampling']={'legacy_effective_k':float((1/oldmass.square().sum(1)).mean()),
                            'new_effective_k':float((1/bel[0].square().sum(1)).mean()),
                            'one_draw_top_action_disagreement':float(torch.stack(disagree).mean())}
        raw[task+'_stratum']=d['protocol'].cpu().numpy()
        raw[task+'_seed']=d['seed'].cpu().numpy()
        results[task]=result
    (OUT/f'pred-{split}.json').write_text(json.dumps(results,indent=2)+'\n')
    np.savez_compressed(OUT/f'paired-{split}.npz',**raw)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('split',choices=['val','test']);ap.add_argument('--device',default='cuda')
    a=ap.parse_args();torch.set_num_threads(2);evaluate(a.device,a.split)
