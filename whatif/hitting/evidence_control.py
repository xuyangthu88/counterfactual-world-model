"""Identical first decisions and seven potential successors; use only the measured motion to update belief."""
import argparse
import json
import torch
from .native import OUT,DATA,DOMAINS
from .data import data,take
from .fit import restore as world
from .policy import features,selector,ordered
from .continuations import winner
from .evidence_data import panel,worlds
from .evidence import restore,readout_features,MODES

@torch.no_grad()
def first():
    net=world('joint',True,'cuda')
    policy=selector(True,'cuda')
    m=winner(True)
    result={}
    for dom in DOMAINS:
        d=data(dom,'early','control','cuda',live=True,path=DATA/f'evidence-initial-{dom}.npz')
        picks=[]
        for ix in torch.arange(len(d['x']),device='cuda').split(64):
            f=features(net,take(d,ix),True)
            score=f['m'][...,0] if m=='direct' else policy(f,m)
            picks.extend(score.argmax(1).tolist())
        result[dom]=picks
    (OUT/'evidence-first.json').write_text(json.dumps(result))

@torch.no_grad()
def second():
    nets=worlds()
    belief,readout=restore()
    firsts=json.loads((OUT/'evidence-first.json').read_text())
    pick=json.loads((OUT/'belief-selection.json').read_text())['mode']
    result={}
    for dom in DOMAINS:
        d=panel(dom,'live',*nets)
        f=readout_features(belief(d),d)
        scores={m:readout(f,m) for m in MODES}
        scores['stage25']=f['base']
        scores['selected']=scores[pick]
        result[dom]={m:[[max(k-1,0),a if k else 0] for k,a in zip(firsts[dom],v.argmax(1).tolist())] for m,v in scores.items()}
        result[dom]['actor']=[[0,0]]*len(d['seed'])
    (OUT/'evidence-second.json').write_text(json.dumps(result))

@torch.no_grad()
def accuracy():
    nets=worlds()
    belief,readout=restore()
    result={}
    for dom in DOMAINS:
        d=panel(dom,'test',*nets)
        s=d['target'][...,0]
        i,j=ordered(s.size(1),s.device)
        target=(s[:,i]-s[:,j]+1).long()
        o=belief(d)
        p=o['logits'].sigmoid()
        scores={}
        for mode in ('prior','moment','posterior'):
            w=o[mode]
            marginal=(p*w[:,None]).sum(-1)
            plus=(p[:,i]*(1-p[:,j])*w[:,None]).sum(-1)
            minus=((1-p[:,i])*p[:,j]*w[:,None]).sum(-1)
            law=torch.stack([minus,1-(minus+plus),plus],-1).clamp_min(1e-8)
            sd=(marginal[:,i]-marginal[:,j])-(s[:,i]-s[:,j])
            scores[mode]=dict(success_brier=float((marginal-s).square().mean()),effect_rmse=float(sd.square().mean().sqrt()),
                effect_nll=float(-law.gather(-1,target[...,None]).log().mean()),
                rescued_brier=float((plus-(target==2).float()).square().mean()))
        scores['kl_posterior_prior']=float((o['posterior']*(o['posterior'].clamp_min(1e-8).log()-o['prior'].clamp_min(1e-8).log())).sum(1).mean())
        result[dom]=scores
    (OUT/'belief-prediction.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['first','second','accuracy'])
    torch.set_num_threads(2)
    {'first':first,'second':second,'accuracy':accuracy}[p.parse_args().stage]()
