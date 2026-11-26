"""Identical atomic interventions for structure, future state, reward and counterfactual effects."""
import json
import torch
from whatif.mechanism.evaluate import metrics
from whatif.mechanism.directed import DirectedNCII
from whatif.mechanism.train import goal_training,OUT as OLD,build
from whatif.paired.evaluate import frozen,graph_metrics,remote
from whatif.hitting.fit import restore as oldworld
from whatif.hitting.native import OUT as PRIOR
from .native import DOMAINS,OUT
from .model import restore
from .data import dataset

def atomic(dom,split):
    d=dataset(dom,'short',split,'cpu')
    for k in ('hx','ha','hy'):
        del d[k]
    return goal_training(d,False)

@torch.no_grad()
def run(extra=False):
    torch.set_num_threads(3)
    old=json.loads((PRIOR/'graphs.json').read_text())
    names=('dynamic-full','fixed-full','dynamic','fixed') if extra else ('dynamic','fixed','unpaired')
    models={k:restore(k).core for k in names}
    records={}
    for dom in ('cophy',*DOMAINS):
        val,test=[atomic(dom,s) for s in ('val','extra' if extra else 'test')]
        options=[(k,'mechanism',v) for k,v in models.items()]+[('stage25','mechanism',oldworld('joint').core)]
        for k in ('cophynet','pets'):
            options.append((k,k,frozen(build(k,'cpu'),OLD/'demonstrations'/f'{k}.pt')))
        if dom=='cophy':
            options.append(('ncii_excision','ncii',frozen(DirectedNCII(),OLD/'directed/ncii.pt')))
        else:
            for k in ('ncii','ncii_contact'):
                options.append((k,'ncii',frozen(build(k,'cpu'),OLD/f'{k}.pt')))
        tab={}
        for name,mode,net in options:
            r=dict(layouts=len(test['x']))
            if mode in ('ncii','mechanism'):
                for key,label in [('edge','contact'),('causal','excision')][:2 if dom=='cophy' else 1]:
                    cut=None if name in models else old[dom]['joint' if name=='stage25' else name][label]['threshold']
                    r[label]=graph_metrics(net,val,test,key,cut)
            if mode!='ncii':
                r.update(metrics(mode,net,test,dom))
                if dom=='cophy' and mode=='mechanism':
                    dist,_,n=remote(net,test)
                    r.update(remote=dist,count=int(n.sum()))
            tab[name]=r
        records[dom]=tab
        print(dom,flush=True)
    (OUT/('atomic-extra.json' if extra else 'atomic.json')).write_text(json.dumps(records,indent=1,allow_nan=False))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--extra',action='store_true')
    run(parser.parse_args().extra)
