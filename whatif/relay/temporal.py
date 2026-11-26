"""Score native edge persistence, births, losses and within-layout changes under intervention."""
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from whatif.hitting.data import take
from whatif.mechanism.model import labeled
from .model import restore
from .data import dataset
from .native import DOMAINS,OUT,ENDS

def rate(a,b):
    return float(a/b) if b else None

def score(y,p):
    y=y.astype(bool)
    return dict(n=int(y.size),positive=int(y.sum()),ap=float(average_precision_score(y,p)) if y.any() else None)

def transition(y,p,valid,mask,d,dom):
    past,now=y[:,:,:-1],y[:,:,1:]
    before,after=p[:,:,:-1],p[:,:,1:]
    valid=valid[:,:,1:]&valid[:,:,:-1]
    valid=valid[...,None,None]&mask[:,None,None]
    birth=(~past)&now
    death=past&(~now)
    result={}
    for key,subset,label,prob in [('birth',~past,now,after),('retained',past,now,after),('loss',past,~now,1-after)]:
        use=valid&subset
        result[key]=score(label[use],prob[use])
    for key,truth,guess in [('birth',birth,(before<.5)&(after>=.5)),('loss',death,(before>=.5)&(after<.5))]:
        hits=int((valid&truth&guess).sum())
        result[key].update(hard_correct=hits,hard_predicted=int((valid&guess).sum()),
                           hard_recall=rate(hits,(valid&truth).sum()),hard_precision=rate(hits,(valid&guess).sum()))
    i,j=np.triu_indices(y.shape[1],1)
    # For feedback plans exclude every actuator on either endpoint, even if this
    # particular future first command selected another finger.
    actors=d['role'][...,1].cpu().numpy().astype(bool)
    active=d['a'][...,0].cpu().numpy().astype(bool) if dom=='cophy' else np.broadcast_to(actors[:,None],(*y.shape[:2],y.shape[-1]))
    excluded=active[:,i]|active[:,j]
    remote=~(excluded[...,None]|excluded[...,None,:])[:,:,None]
    valid_pair=valid[:,i]&valid[:,j]&remote
    use=valid_pair&(~past[:,i])&(~past[:,j])&(now[:,i]!=now[:,j])
    direction=now[:,i].astype(float)-now[:,j]
    diff=after[:,i]-after[:,j]
    oriented=(direction*diff)[use]
    positive=np.where(direction>0,after[:,i],after[:,j])
    negative=np.where(direction>0,after[:,j],after[:,i])
    result['remote_action_birth']=dict(n=int(use.sum()),layouts=int(use.any(axis=(1,2,3,4)).sum()),
        correct=rate((oriented>1e-6).sum()+.5*(abs(oriented)<=1e-6).sum(),len(oriented)),
        mean_signed=float(oriented.mean()) if len(oriented) else None,
        hard_separation=rate(((positive>=.5)&(negative<.5)&use).sum(),len(oriented)),
        initial_max=float(np.abs(p[:,i,0]-p[:,j,0])[np.broadcast_to(remote[:,:,0]&mask[:,None],diff[:,:,0].shape)].max(initial=0)))
    return result,use

@torch.no_grad()
def run():
    torch.set_num_threads(3)
    records={}
    example=None
    for kind in ('dynamic','fixed','unpaired'):
        net=restore(kind,'cuda')
        table={}
        for dom in ('cophy',*DOMAINS):
            for ph in (['short'] if dom=='cophy' else ['short','early','late']):
                d=dataset(dom,ph,'test','cuda')
                valid=d['valid'].bool().cpu().numpy()
                mask=(labeled(d['role'])*d['role'][:,None,:,0]).bool().cpu().numpy()
                groups=[]
                for ix in torch.arange(len(valid),device='cuda').split(40):
                    g=net.core(take(d,ix))
                    groups.append([g[k].sigmoid().cpu().numpy() for k in ('edge_time','causal_time')])
                ps=[np.concatenate(parts) for parts in zip(*groups)]
                tag=dom+'-'+ph
                table[tag]={}
                for key,y,p in [('contact',d['contact']>0,ps[0]),('deletion',d['effects']>.1,ps[1])][:2 if dom=='cophy' else 1]:
                    lab=y.cpu().numpy()
                    stat,use=transition(lab,p,valid,mask,d,dom)
                    table[tag][key]=stat
                    if kind=='dynamic' and dom=='pusht' and ph=='early':
                        # First in dataset order among true contrasting pairs where the
                        # new edge was also opened by the model on just the positive branch.
                        i,j=np.triu_indices(p.shape[1],1)
                        for row,pair,t,src,dst in np.argwhere(use):
                            a,b=i[pair],j[pair]
                            if not lab[row,a,t+1,src,dst]:
                                a,b=b,a
                            if p[row,a,t,src,dst]>=.5 or p[row,a,t+1,src,dst]<.5 or p[row,b,t+1,src,dst]>=.5:
                                continue
                            example=dict(seed=int(d['seed'][row]),period=int(t+1),window_ends=ENDS['early'],wall=int(src),arms=[int(a),int(b)],
                                x=d['x'][row].cpu().tolist(),positions=d['traj'][row,[a,b],:,0,:2].cpu().tolist(),
                                truth=lab[row,[a,b],:,src,dst].astype(int).tolist(),prob=p[row,[a,b],:,src,dst].tolist(),
                                pusher=lab[row,[a,b],:,1,0].astype(int).tolist(),pusher_prob=p[row,[a,b],:,1,0].tolist())
                            break
                print(kind,tag,flush=True)
        records[kind]=table
    (OUT/'temporal.json').write_text(json.dumps(records,indent=1,allow_nan=False))
    if example:
        (OUT/'example.json').write_text(json.dumps(example,indent=1))
if __name__=='__main__':
    run()
