"""Evaluate the deployment archive itself on CoPhy, both pushing domains, and actual recourse graphs."""
import json
import numpy as np
import torch
from whatif.mechanism.model import labeled
from whatif.mechanism.evaluate import metrics
from whatif.paired.evaluate import remote,graph_metrics
from whatif.hitting.data import take,load_np
from .atomic import atomic
from .backup import world,records,menu
from .data import dataset
from .forecast import evaluate
from .temporal import transition
from .native import DATA,OUT,DOMAINS

@torch.no_grad()
def run():
    torch.set_num_threads(3)
    net=world('cpu')
    prior=json.loads((OUT/'atomic-extra.json').read_text())
    result={}
    for dom in ('cophy',*DOMAINS):
        d=atomic(dom,'extra')
        r=metrics('mechanism',net.core,d,dom)
        for k,key in [('edge','contact'),('causal','excision')][:2 if dom=='cophy' else 1]:
            r[key]=graph_metrics(net.core,d,d,k,prior[dom]['dynamic-full'][key]['threshold'])
        if dom=='cophy':
            r['remote'],_,_=remote(net.core,d)
        result[dom]=dict(atomic=r)
    net=net.cuda()
    for dom in DOMAINS:
        v=menu(records(dom,'test','cuda'))
        m=(labeled(v['role'])*v['role'][:,None,:,0]).bool().cpu().numpy()
        obs=load_np(DATA/f'backup-contact-{dom}.npz')
        if not np.array_equal(v['seed'].cpu().numpy(),obs['seed']):
            raise ValueError('misaligned graph reference')
        logits=torch.cat([net.core(take(v,ix))['edge_time'].sigmoid().cpu() for ix in torch.arange(len(m),device='cuda').split(24)]).numpy()
        result[dom]['recourse_contact']=transition(obs['contact']>0,logits,obs['valid']>0,m,v,dom)[0]
        result[dom]['late']=evaluate(net,dataset(dom,'late','test','cuda'),'final',100 if dom=='pusht' else 65)
    (OUT/'final.json').write_text(json.dumps(result,indent=1,allow_nan=False))
if __name__=='__main__':
    run()
