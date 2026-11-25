"""Select the deployment readout using held-out native branches and a common validation menu."""
import json
import torch
from .native import DOMAINS,OUT
from .fit import restore
from .backup import restore as backed
from .load import early,late,read,take

@torch.no_grad()
def run():
    net=restore('cuda')
    backup=backed('cuda')
    result={}
    for dom in DOMAINS:
        d=read(dom,'val','cuda')
        b=len(d['x'])//3
        options=d['first'].reshape(b,3)
        ix=torch.arange(b,device='cuda')
        fscore={}
        sscore={}
        for rows in ix.split(16):
            de=early(take(d,3*rows))
            values=net.first_scores(de)
            z,_=net.encode(de,8)
            values.update(backup.scores(backup(z)))
            for k,v in values.items():
                fscore.setdefault(k,[]).append(v.gather(-1,options[rows]))
        for r in torch.arange(3*b,device='cuda').split(40):
            row=take(d,r)
            for k,v in net.last_scores(late(row),row['first']).items():
                sscore.setdefault(k,[]).append(v)
        result[dom]={}
        for k,v in fscore.items():
            i=3*ix+torch.cat(v).argmax(1)
            mode='mean' if k.startswith('backup_') else k
            j=torch.cat(sscore[mode])[i].argmax(1)
            y=d['outcome'][i,j]
            g=y[:,0]+.5*y[:,1]+.25*y[:,2]
            result[dom][k]=dict(success=int(y[:,0].sum()),G=float(g.mean()))
    arms=list(result[DOMAINS[0]])
    best=max(arms,key=lambda arm:(sum(result[d][arm]['success'] for d in DOMAINS),
                                  sum(result[d][arm]['G'] for d in DOMAINS)))
    (OUT/'validation.json').write_text(json.dumps(dict(by_domain=result,selected=best),indent=2))
if __name__=='__main__':
    torch.set_num_threads(2)
    run()
