"""Fit directed endpoint sensitivities using genuine pair-excision targets."""
import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
from .model import Mechanism,pairs,mlp,labeled
from .baselines import NCII,node_inputs
from .train import BATCH,EPOCHS,LR,SEED,DATA,OUT,avg,read,hindsight,sub,cost
from .native import D,A
from .evaluate import graph,metrics
from .figures import plt,bars,save

DEST=OUT/'directed'


class DirectedNCII(NCII):
    def __init__(self):
        super().__init__()
        self.transfer=mlp(2*(3*D+2*A+3),2)

    def forward(self,d):
        o=super().forward(d)
        b,k,n,_=d['a'].shape
        e,mag=self.transfer(pairs(node_inputs(d))).reshape(b,k,n,n,2).unbind(-1)
        prob=o['edge'].sigmoid()*(e+1.5).sigmoid()
        o.update(causal=torch.logit(prob.clamp(1e-6,1-1e-6)),impact=F.softplus(mag))
        return o


def dataset(split,device):
    if split=='test':
        items=dict(np.load(DATA/'directed/cophy-test.npz'))
        d={k:torch.tensor(v,device=device).float() for k,v in items.items()}
        d.update(reward_mask=torch.zeros_like(d['seed']),same=torch.ones_like(d['seed']))
    else:
        d=read(split,device)
    # The raw intervention data use the exact same layout seeds as the CoPhy portion.
    archive=DATA/f'excision-{split}.npz' if split=='val' else DATA/f'directed/excision-{split}.npz'
    raw=np.load(archive)
    which=(d['reward_mask']==0)
    if not np.array_equal(d['seed'][which].cpu(),raw['seed']):
        raise ValueError('mismatched excision trajectories')
    impact=torch.zeros_like(d['e'])
    impact[which]=torch.as_tensor(raw['impact'],device=device)
    d.update(impact=impact,excision_mask=which.float())
    return d


def objective(name,model,d,train):
    o=model(d)
    original=cost('mechanism' if name=='graph' else 'ncii_contact',model,d,train,o)
    mask=labeled(d['role'])[:,None]*d['excision_mask'][:,None,None,None]
    binary=F.binary_cross_entropy_with_logits(o['causal'],(d['impact']>.10).float(),reduction='none')
    strength=F.smooth_l1_loss(o['impact'],torch.log1p(d['impact']),beta=.2,reduction='none')
    return original+1.2*avg(binary,mask)+.45*avg(strength,mask)


def fit(name,device):
    torch.manual_seed(SEED)
    tr,va=dataset('train',device),dataset('val',device)
    net=(Mechanism(directed=True) if name=='graph' else DirectedNCII()).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=LR)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.00015)
    best=float('inf')
    history=[]
    for epoch in range(EPOCHS):
        net.train()
        for ix in torch.randperm(len(tr['x']),device=device).split(BATCH):
            d=sub(tr,ix)
            ref=torch.randint(4,(len(ix),),device=device)
            d=hindsight(d,torch.rand(len(ix),device=device)<.35,ref)
            loss=objective(name,net,d,True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            opt.step()
        schedule.step()
        if epoch%3==2 or epoch==0:
            net.eval()
            with torch.no_grad():
                # The three environments are contiguous; evaluate the joint objective on one batch.
                score=objective(name,net,va,False).item()
            history.append([epoch+1,score])
            if score<best:
                best=score
                torch.save(net.state_dict(),DEST/f'{name}.pt')
            print(name,epoch+1,round(score,4),flush=True)
    (DEST/f'{name}-fit.json').write_text(json.dumps(history))


@torch.no_grad()
def evaluate():
    val=dataset('val','cpu')
    val=sub(val,val['excision_mask'].bool())
    test=dataset('test','cpu')
    results={}
    for name,net,path in [('contact_graph',Mechanism(),OUT/'mechanism.pt'),
                          ('directed_graph',Mechanism(directed=True),DEST/'graph.pt'),
                          ('directed_ncii',DirectedNCII(),DEST/'ncii.pt')]:
        net.load_state_dict(torch.load(path,map_location='cpu'))
        net.eval()
        scores={}
        cuts={}
        for split,d in [('val',val),('test',test)]:
            o=net(d)
            mask=labeled(d['role'])[:,None].bool().expand_as(d['e'])
            for key,t,p in [('contact',d['e'],o['edge']),('causal',d['impact']>.10,o.get('causal',o['edge']))]:
                g=graph(t[mask].numpy(),p[mask].sigmoid().numpy(),cuts.get(key))
                if split=='val':
                    cuts[key]=g['threshold']
                else:
                    scores[key]=g
            if split=='test' and 'impact' in o:
                scores['log_impact_mae']=float((o['impact']-d['impact'].log1p()).abs()[mask].mean())
        if name!='directed_ncii':
            scores['prediction']=metrics('mechanism',net,test,'cophy')
        results[name]=scores
    (DEST/'metrics.json').write_text(json.dumps(results,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(8.8,3.3))
    for a,k,title in zip(ax,['contact','causal'],['接触 AP','有向删边效应 AP']):
        bars(a,['仅接触训练','有向图','同监督 NCII'],[r[k]['ap'] for r in results.values()],title)
    save(fig,'20-directed.png')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('operation',choices=['graph','ncii','evaluate'])
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    torch.set_num_threads(3)
    DEST.mkdir(parents=True,exist_ok=True)
    if args.operation=='evaluate':
        evaluate()
    else:
        fit(args.operation,args.device)


if __name__=='__main__':
    main()

