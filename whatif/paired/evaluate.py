"""Fresh native prediction, object/contact/excision graphs, and success-transition calibration."""
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from whatif.mechanism.train import OUT as OLD,build,goal_training
from whatif.mechanism.evaluate import graph,metrics,predictions
from whatif.mechanism.directed import DirectedNCII
from whatif.mechanism.model import Mechanism,labeled
from .fit import current,world
from .data import OUT
from .decision import ModalAction,decision_head,panel,targets,statistics,choices


def frozen(net,path):
    net.load_state_dict(torch.load(path,map_location='cpu'))
    return net.eval()


def graph_metrics(model,val,test,which,cut=None):
    for split,d in [('val',val),('test',test)]:
        if split=='val' and cut is not None:
            continue
        mask=(labeled(d['role'])[:,None]*d['role'][:,None,None,:,0]).bool().expand_as(d['e'])
        truth=d['e'] if which=='edge' else d['impact']>.1
        pred=model(d)[which].sigmoid()
        summary=graph(truth[mask].numpy(),pred[mask].numpy(),cut)
        cut=summary['threshold']
    return summary


def remote(model,d):
    k=d['a'].size(1)
    i,j=torch.triu_indices(k,k,1)
    means=predictions('mechanism',model,d,i,j)[2].mean(0)
    err=(means-(d['y'][:,i]-d['y'][:,j]))[...,:3].square().sum(-1)
    ia=d['a'][:,i,:,0].argmax(-1)
    ib=d['a'][:,j,:,0].argmax(-1)
    depth=torch.arange(d['x'].size(1))
    sep=torch.minimum((depth-ia[...,None]).abs(),(depth-ib[...,None]).abs())
    mask=(sep>=2)&d['role'][:,None,:,0].bool()
    squared=(err*mask).sum((1,2)).numpy()
    counts=mask.sum((1,2)).numpy()
    return float(np.sqrt(squared.sum()/counts.sum())),squared,counts


def laplace_crps(o,y):
    p=o['mass'].double()
    loc=o['mu'].double()+p.new_tensor([0,-1,1,0])
    b=o['scale'].double()
    t=(loc-y[...,None]).abs()
    first=(p*(t+b*(-t/b).exp())).sum(-1)
    a=b.unsqueeze(-1)
    b=b.unsqueeze(-2)
    gap=(loc.unsqueeze(-1)-loc.unsqueeze(-2)).abs()
    denom=a.square()-b.square()
    equal=(a-b).abs()<1e-6
    middle=.5*(a+b)
    term=(a.pow(3)*(-gap/a).exp()-b.pow(3)*(-gap/b).exp())/torch.where(equal,1.,denom)
    term=torch.where(equal,(1.5*middle+.5*gap)*(-gap/middle).exp(),term)
    return first-.5*(p.unsqueeze(-1)*p.unsqueeze(-2)*(gap+term)).sum((-1,-2))


@torch.no_grad()
def main():
    torch.set_num_threads(3)
    historical=json.loads((OLD/'edges.json').read_text())
    directed=json.loads((OLD/'directed/metrics.json').read_text())
    restored=json.loads((OLD/'demonstrations/prediction.json').read_text())
    records={}
    distances={}
    current_model=world()
    for dom in ['cophy','causalworld','pusht']:
        val,test=[goal_training(current(dom,s,'cpu'),False) for s in ('val','test')]
        candidates=[('transport',current_model,'mechanism'),('contact_path',world(True),'mechanism')]
        if dom=='cophy':
            candidates+= [('stage20',frozen(Mechanism(directed=True),OLD/'directed/graph.pt'),'mechanism'),
                          ('ncii_excision',frozen(DirectedNCII(),OLD/'directed/ncii.pt'),'ncii')]
        else:
            candidates+= [(name,frozen(build(name,'cpu'),OLD/'demonstrations'/f'{name}.pt'),'mechanism' if name=='mechanism' else name)
                          for name in ['mechanism','cophynet','pets']]
            candidates+= [(name,frozen(build(name,'cpu'),OLD/f'{name}.pt'),'ncii') for name in ['ncii','ncii_contact']]
        for name,model,mode in candidates:
            results={}
            if mode not in ('pets','cophynet'):
                cut=None
                if dom=='cophy' and name in ('stage20','ncii_excision'):
                    cut=directed['directed_graph' if name=='stage20' else 'directed_ncii']['contact']['threshold']
                elif name in ('ncii','ncii_contact'):
                    cut=historical[dom][name]['contact']['threshold']
                elif name=='mechanism':
                    cut=restored[dom][name]['contact']['threshold']
                results['contact']=graph_metrics(model,val,test,'edge',cut)
                if dom=='cophy':
                    oldcut=(directed['directed_graph' if name=='stage20' else 'directed_ncii']['causal']['threshold']
                            if name in ('stage20','ncii_excision') else None)
                    results['excision']=graph_metrics(model,val,test,'causal',oldcut)
            if mode!='ncii':
                results.update(metrics(mode,model,test,dom))
                if dom=='cophy':
                    result,sq,cnt=remote(model,test)
                    results.update(distant_effect_rmse=result,distant_receiver_pairs=int(cnt.sum()))
                    distances[name]=sq
                    distances['count']=cnt
            records.setdefault(dom,{})[name]=results
        print(dom,flush=True)
    idx=np.random.default_rng(271).integers(0,len(distances['count']),(4000,len(distances['count'])))
    for other in ['contact_path','stage20']:
        def boot(name):
            return np.sqrt(distances[name][idx].sum(1)/distances['count'][idx].sum(1))
        records['cophy']['transport']['difference_vs_'+other]=np.quantile(boot('transport')-boot(other),[.025,.975]).tolist()
    head=decision_head()
    xx,yy,priors=panel(current_model,ModalAction(),'test','cpu')
    config=json.loads((OUT/'selection.json').read_text())['config']
    for dom,idx in zip(['causalworld','pusht'],torch.arange(len(xx)).chunk(2)):
        x,y,prior=xx[idx],yy[idx],priors[idx]
        i,j=torch.triu_indices(x.size(1),x.size(1),1)
        o=head(x,i,j)
        s,aux=targets(y)
        truth=(s+aux)[:,i]-(s+aux)[:,j]
        v=statistics(o)
        r=dict(return_effect_rmse=float((v['mean']-truth).square().mean().sqrt()),
               return_crps=float(laplace_crps(o,truth).mean()))
        ix=torch.arange(x.size(1))
        o=head(x,ix,torch.zeros_like(ix))
        for label,index in [('rescue',2),('break',1)]:
            observed=(s[:,1:]>s[:,:1]) if label=='rescue' else (s[:,1:]<s[:,:1])
            p=o['mass'][:,1:,index]
            r[label]=dict(ap=float(average_precision_score(observed.flatten(),p.flatten())),
                          rate=float(observed.float().mean()),brier=float((observed.float()-p).square().mean()))
        for policy in ['marginal','pair_mean','pair_law','pair_kl']:
            a=choices(o,prior,policy,config)
            g=s+aux
            r[policy]=dict(gain=float((g.gather(1,a[:,None])-g[:,:1]).mean()),
                           switched=float((a!=0).float().mean()))
        records[dom]['return']=r
    (OUT/'offline.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    np.savez_compressed(OUT/'remote-errors.npz',**distances)


if __name__=='__main__':
    main()
