"""Fresh same-layout atomic interventions. Compare all heads on identical native labels."""
import json
import torch
from whatif.paired.fit import world as old_world,current,load_rows
from whatif.mechanism.evaluate import metrics
from whatif.mechanism.directed import DirectedNCII
from whatif.mechanism.train import goal_training,OUT as OLD,build
from whatif.paired.data import OUT as PAIR
from whatif.paired.evaluate import frozen,graph_metrics,remote
from .fit import restore
from .native import DOMAINS,DATA,OUT
from .load import read,take,late

def atomic(dom,split):
    if dom=='cophy':
        d=(current(dom,'val','cpu') if split=='val' else
           load_rows(DATA/'cophy-test.npz',dom,'cpu',DATA/'excision-test.npz'))
    else:
        all_rows=read(dom,split)
        # One sampled first policy per seed, before seeing its outcomes.
        stride=3 if split=='val' else 4
        first=take(all_rows,torch.arange(0,len(all_rows['x']),stride))
        row=take(first,first['midscore'][:,0]==0)
        d=late(row)
        # Atomic comparison has the last single factual transition for all models.
        for k in ('hx','ha','hy'):
            d.pop(k)
        d.update(e=row['late_e'],r=row['late_r'],y=row['late_y'],seed=row['seed'].float(),
                 fe=torch.zeros_like(row['late_e'][:,0]),reward_mask=torch.ones_like(row['alive']))
    return goal_training(d,False)

@torch.no_grad()
def run():
    torch.set_num_threads(3)
    older=json.loads((PAIR/'offline.json').read_text())
    core=restore().core
    records={}
    for dom in ('cophy',*DOMAINS):
        val,test=[atomic(dom,s) for s in ('val','test')]
        candidates=[('joint','mechanism',core),('transport','mechanism',old_world())]
        if dom=='cophy':
            candidates.append(('ncii_excision','ncii',frozen(DirectedNCII(),OLD/'directed/ncii.pt')))
        else:
            for k in ('ncii','ncii_contact','cophynet','pets'):
                ckpt=OLD/f'{k}.pt' if k.startswith('ncii') else OLD/'demonstrations'/f'{k}.pt'
                candidates.append((k,'ncii' if k.startswith('ncii') else k,frozen(build(k,'cpu'),ckpt)))
        table={}
        for name,mode,model in candidates:
            r=dict(layouts=len(test['x']))
            if mode in ('ncii','mechanism'):
                for k,label in [('edge','contact'),('causal','excision')][:2 if dom=='cophy' else 1]:
                    cut=None if name=='joint' else older[dom][name][label]['threshold']
                    r[label]=graph_metrics(model,val,test,k,cut)
            if mode!='ncii':
                r.update(metrics(mode,model,test,dom))
                if dom=='cophy':
                    dist,_,count=remote(model,test)
                    r.update(remote=dist,remote_pairs=int(count.sum()))
            table[name]=r
        records[dom]=table
    (OUT/'graphs.json').write_text(json.dumps(records,indent=2,allow_nan=False))

if __name__=='__main__':
    run()
