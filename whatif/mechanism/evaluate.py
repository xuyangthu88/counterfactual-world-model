"""Validation-only edge thresholds; held-out direct contacts, excisions and paired outcomes."""
import json
import math
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_curve
from .model import labeled,pool
from .train import build,read,sub,hindsight,DOMAINS,DATA,OUT,NAMES


def graph(y,p,threshold=None):
    y,p=np.asarray(y).astype(bool),np.asarray(p)
    if threshold is None:
        fpr,tpr,cut=roc_curve(y,p)
        best=np.flatnonzero((fpr+1-tpr)<=(fpr+1-tpr).min()+1e-12)
        finite=best[np.isfinite(cut[best])]
        threshold=float(cut[finite[len(finite)//2]]) if len(finite) else float(p.max()+1)
    on=p>=threshold
    fpr=float(on[~y].mean())
    fnr=float(1-on[y].mean())
    tp=(on&y).sum()
    return dict(threshold=threshold,positive=float(y.mean()),n=int(y.size),ap=float(average_precision_score(y,p)),
                ber=(fpr+fnr)/2,precision=float(tp/max(1,on.sum())),recall=1-fnr,fpr=fpr,
                f1=float(2*tp/max(1,on.sum()+y.sum())),shd_rate=float((on!=y).mean()))


def mixture_crps(mu,sd,y):
    def normabs(z,s):
        safe=s.clamp_min(1.e-7)
        t=z/safe
        return z*(2*torch.special.ndtr(t)-1)+safe*math.sqrt(2/math.pi)*torch.exp(-.5*t*t)
    first=normabs(mu-y,sd).mean(0)
    second=normabs(mu[:,None]-mu[None,:],(sd[:,None].square()+sd[None,:].square()).sqrt()).mean((0,1))
    return first-.5*second


def predictions(name,model,d,i,j):
    if name=='mechanism':
        stack=[]
        gen=torch.Generator().manual_seed(939)
        two=torch.randn(2,*d['x'].shape[:2],4,generator=gen)
        for epsilon in torch.cat([two,-two]):
            o=model(dict(d,eps=epsilon))
            (dm,ds),(rm,rs)=model.contrast(o,i,j)
            stack.append((o['y'],o['r'],dm,ds+.12,rm,rs+.055))
        y,r,dm,ds,rm,rs=map(torch.stack,zip(*stack))
        return y.mean(0),r.mean(0),dm,ds,rm,rs
    o=model(d)
    y,r=o['y'],o['r']
    if name=='pets':
        ensemble=o['ensemble']
        sd=(.5*o['lv']).exp()
        movable=d['role'][:,None,:,0]
        # Use a common standardized Gaussian shock and member for both actions.
        rm=pool(ensemble[:, :, i,...,9:]-ensemble[:, :, j,...,9:],movable)
        rs=pool((sd[:,:,i,...,9:]-sd[:,:,j,...,9:]).abs(),movable)
        return y,r,ensemble[:,:,i,...,:9]-ensemble[:,:,j,...,:9],(sd[:,:,i,...,:9]-sd[:,:,j,...,:9]).abs(),rm,rs
    dm=(y[:,i]-y[:,j])[None]
    rm=(r[:,i]-r[:,j])[None]
    return y,r,dm,torch.zeros_like(dm),rm,torch.zeros_like(rm)


def metrics(name,model,d,domain):
    unit={'causalworld':65.,'pusht':100.,'cophy':1.}[domain]
    k=d['a'].size(1)
    i,j=torch.triu_indices(k,k,1)
    y,r,mu,sd,rm,rs=predictions(name,model,d,i,j)
    dy=d['y'][:,i]-d['y'][:,j]
    dr=d['r'][:,i]-d['r'][:,j]
    move=d['role'][:,None,:,0].bool()
    def selected(v,mask):
        return v[mask.expand_as(v)]
    def position(err,mask):
        dist=selected(err[...,:3].norm(dim=-1),mask)*unit
        return float(dist.square().mean().sqrt())
    def reward_rmse(error,index):
        return float(error[...,index].square().mean().sqrt())
    pred=mu.mean(0)
    result=dict(future_position_rmse=position(y-d['y'],move),effect_position_rmse=position(pred-dy,move),
                future_pose_mae=float(selected((y-d['y']).abs(),move[...,None]).mean()),
                effect_xyz_crps=float(selected(mixture_crps(mu[...,:3],sd[...,:3],dy[...,:3]),move[...,None]).mean()*unit))
    # Alternate branch zero is revealed as a factual observation; the other three are hidden.
    idx=torch.zeros(len(d['x']),dtype=torch.long)
    retrospective=hindsight(d,torch.ones_like(idx,dtype=torch.bool),idx)
    choices=torch.arange(1,k)
    _,_,post,_,post_r,_=predictions(name,model,retrospective,choices,torch.zeros_like(choices))
    result['anchored_position_rmse']=position(d['y'][:,:1]+post.mean(0)-d['y'][:,1:],move)
    if domain=='cophy':
        # CoPhy do() directly relocates one block. Test transport to untouched blocks separately.
        untouched=(d['a'][:,i,: ,0]+d['a'][:,j,:,0])==0
        result['untouched_effect_rmse']=position(pred-dy,move&untouched)
    else:
        result.update(native_reward_rmse=reward_rmse(r-d['r'],0),
                      native_reward_effect_rmse=reward_rmse(rm.mean(0)-dr,0),
                      native_reward_effect_crps=float(mixture_crps(rm[...,0],rs[...,0],dr[...,0]).mean()),
                      terminal_score_rmse=reward_rmse(r-d['r'],1),
                      anchored_native_reward_rmse=reward_rmse(d['r'][:,:1]+post_r.mean(0)-d['r'][:,1:],0))
        score=d['r'][:,:,1]
        result['terminal_regret']=float((score.max(1).values-score.gather(1,r[:,:,1].argmax(1,keepdim=True))[:,0]).mean())
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(3)
    edges,pred={},{}
    for name in NAMES:
        m=build(name,'cpu')
        m.load_state_dict(torch.load(OUT/f'{name}.pt',map_location='cpu'))
        m.eval()
        for domain in DOMAINS:
            val,test=[read(s,'cpu',[domain]) for s in ['val','test']]
            if name in ('mechanism','ncii','ncii_contact'):
                scores=[]
                for split,d in zip(['val','test'],[val,test]):
                    o=m(d)
                    receiver=d['role'][:,None,None,:,0]
                    mask=(labeled(d['role'])[:,None]*receiver).bool().expand_as(d['e'])
                    probability=o['edge'].sigmoid().numpy()
                    native=d['e'].numpy()
                    labels={'contact':(native,probability,mask.numpy())}
                    if domain=='cophy':
                        archive=np.load(DATA/f'excision-{split}.npz')
                        assert np.array_equal(archive['seed'],d['seed'].numpy())
                        labels['excision']=(archive['impact']>.10,probability,mask.numpy())
                    if name=='mechanism':
                        labels['factual']=(d['fe'].numpy(),o['fact'].sigmoid().numpy(),mask[:,0].numpy())
                    scores.append(labels)
                out={}
                for key,(y,p,mask) in scores[0].items():
                    cut=graph(y[mask],p[mask])['threshold']
                    y,p,mask=scores[1][key]
                    out[key]=graph(y[mask],p[mask],cut)
                    if domain=='cophy' and key=='excision':
                        for h in [3,4]:
                            keep=(test['role'][:,:,0].sum(1).numpy()==h)
                            out[key+str(h)]=graph(y[keep][mask[keep]],p[keep][mask[keep]],cut)
                edges.setdefault(domain,{})[name]=out
            if name in ('mechanism','cophynet','pets'):
                pred.setdefault(domain,{})[name]=metrics(name,m,test,domain)
    for file,table in [('edges',edges),('prediction',pred)]:
        (OUT/f'{file}.json').write_text(json.dumps(table,indent=2,allow_nan=False))
        print(file,flush=True)


if __name__=='__main__':
    main()
