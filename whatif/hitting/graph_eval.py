"""Contact and intervention structure, with identical atomic evidence for every checkpoint."""
import json
import torch
import numpy as np
from whatif.mechanism.evaluate import metrics
from whatif.mechanism.directed import DirectedNCII
from whatif.mechanism.train import goal_training,OUT as OLD,build
from whatif.paired.fit import current,load_rows
from whatif.paired.evaluate import frozen,graph_metrics,remote
from whatif.recourse.fit import restore as previous
from whatif.recourse.native import OUT as PRIOR
from .data import load_np
from .continuations import midpoint
from .fit import restore
from .native import DOMAINS,DATA,OUT

def atomic(dom,split):
    if dom=='cophy':
        d=(current(dom,'val','cpu') if split=='val' else load_rows(DATA/'cophy-test.npz',dom,'cpu',DATA/'excision-test.npz'))
    else:
        raw=load_np(DATA/f'late-{dom}-{split}.npz')
        ix=np.flatnonzero(raw['mid_alive'][::2])*2
        r={k:v[ix] for k,v in raw.items()}
        d=midpoint(r,'cpu')
        for k in ('hx','ha','hy'):
            del d[k]
        d.update({k:torch.tensor(r[k]).float() for k in ('e','r','y')})
        d.update(fe=torch.zeros_like(d['e'][:,0]),reward_mask=torch.ones(len(ix)))
    return goal_training(d,False)

@torch.no_grad()
def run():
    torch.set_num_threads(3)
    old=json.loads((PRIOR/'graphs.json').read_text())
    models={k:restore(k).core for k in ('joint','marginal')}
    out={}
    for dom in ('cophy',*DOMAINS):
        val,test=[atomic(dom,s) for s in ('val','test')]
        options=[(k,'mechanism',v) for k,v in models.items()]+[('stage24','mechanism',previous().core)]
        if dom=='cophy':
            options+=[('ncii_excision','ncii',frozen(DirectedNCII(),OLD/'directed/ncii.pt'))]
        else:
            for k in ('ncii','ncii_contact','cophynet','pets'):
                path=OLD/f'{k}.pt' if k.startswith('ncii') else OLD/'demonstrations'/f'{k}.pt'
                options.append((k,'ncii' if k.startswith('ncii') else k,frozen(build(k,'cpu'),path)))
        table={}
        for name,mode,model in options:
            entry=dict(layouts=len(test['x']))
            if mode in ('ncii','mechanism'):
                for k,label in [('edge','contact'),('causal','excision')][:2 if dom=='cophy' else 1]:
                    threshold=None if name in models else old[dom]['joint' if name=='stage24' else name][label]['threshold']
                    entry[label]=graph_metrics(model,val,test,k,threshold)
            if mode!='ncii':
                entry.update(metrics(mode,model,test,dom))
                if dom=='cophy':
                    value,_,counts=remote(model,test)
                    entry.update(remote=value,remote_pairs=int(counts.sum()))
            table[name]=entry
        out[dom]=table
        print(dom,len(test['x']),flush=True)
    (OUT/'graphs.json').write_text(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__':
    run()
