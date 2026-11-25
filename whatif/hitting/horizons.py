"""First-segment contact predictions and a matched frozen Stage 24 long-horizon comparison."""
import json
import torch
from whatif.mechanism.model import labeled,pool
from whatif.mechanism.evaluate import graph
from whatif.recourse.accuracy import forecasts,stats
from whatif.recourse.fit import restore as previous
from .data import data,take,load_np
from .fit import restore
from .native import DOMAINS,DATA,OUT,BANK

@torch.no_grad()
def older(dom):
    net=previous('cuda')
    d=data(dom,'late','test','cuda')
    raw=load_np(DATA/f'late-{dom}-test.npz')
    mask=raw['mid_alive'].astype(bool)
    x=torch.tensor(raw['final'][mask],device='cuda')
    rewards=torch.tensor(raw['native'][mask],device='cuda')
    target=torch.cat([pool(x,d['role'][:,None,:,0])[...,:3],rewards[...,None]],-1)
    parts=[]
    for ix in torch.arange(len(x),device='cuda').split(48):
        row=take(d,ix)
        z=net.encode(row,27)[0]
        known=(z+net.after(z))[:,None]
        mid=pool(row['x'],row['role'][...,0])
        parts.append([t.cpu() for t in forecasts(net,known,torch.ones_like(mid[:,:1]),mid)])
    out=[torch.cat(ts) for ts in zip(*parts)]+[(torch.arange(-1,2)[:,None]+net.grid.cpu()).flatten()]
    return stats(out,d['outcome'].cpu(),target.cpu(),100 if dom=='pusht' else 65)

@torch.no_grad()
def prefix(model,dom,split,cut):
    d=data(dom,'early',split,'cuda')
    preds=[]
    for ix in torch.arange(len(d['x']),device='cuda').split(48):
        row=take(d,ix)
        pred=model(row,True)[1]['program_edge'][:,1:].sigmoid()
        preds.append(pred.cpu())
    p=torch.cat(preds)
    lab=torch.tensor(load_np(DATA/f'contact-{dom}-{split}.npz')['contact']).reshape_as(p)
    mask=(labeled(d['role'])[:,None]*d['role'][:,None,None,:,0]).bool().expand_as(p).cpu()
    return graph(lab[mask],p[mask],cut)

if __name__=='__main__':
    torch.set_num_threads(2)
    results={}
    for dom in DOMAINS:
        entry=dict(stage24_late=older(dom))
        for k in ('marginal','joint'):
            model=restore(k,True,'cuda')
            v=prefix(model,dom,'val',None)
            entry[k]=prefix(model,dom,'test',v['threshold'])
        results[dom]=entry
    (OUT/'horizons.json').write_text(json.dumps(results,indent=2,allow_nan=False))
