"""Condition the prospective joint law of a measured motion and seven binary outcomes."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from whatif.mechanism.model import mlp
from .native import OUT,DOMAINS
from .data import take
from .evidence_data import panel,worlds
from .policy import ordered,loss as action_loss

H=6
MODES=('prior','moment','posterior','law','kl')

class Belief(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry=mlp(64,H*(1+2*12),80)
        self.prob=mlp(64*3,H,80)
        self.point=mlp(64+12,H,48)
        self.register_buffer('center',torch.zeros(12))
        self.register_buffer('scale',torch.ones(12))

    def forward(self,d):
        a,b,w=d['first'],d['z'],(d['w']-self.center)/self.scale
        theta=self.geometry(a).unflatten(-1,(H,25))
        logpi=theta[...,0].log_softmax(-1)
        mu=theta[...,1:13]
        sd=.12+F.softplus(theta[...,13:])
        logdensity=(-.5*((w[:,None]-mu)/sd).square()-sd.log()).sum(-1)
        joint=logpi+logdensity
        post=joint.softmax(-1)
        mean=(logpi.exp()[...,None]*mu).sum(1)
        moment=(logpi+self.point(torch.cat([a,w-mean],-1))).softmax(-1)
        l=a[:,None].expand_as(b)
        logits=self.prob(torch.cat([l,b,b-l],-1))
        return dict(prior=logpi.exp(),posterior=post,moment=moment,logits=logits,
                    log_density=joint.logsumexp(-1),logpost=joint.log_softmax(-1))

def objective(o,target):
    s=target[...,0]
    branch=-F.binary_cross_entropy_with_logits(o['logits'],s[...,None].expand_as(o['logits']),reduction='none')
    joint_nll=-(o['logpost']+branch.sum(1)).logsumexp(-1).mean()/s.size(1)
    p=o['logits'].sigmoid()
    def bce(q):
        return F.binary_cross_entropy((p*q[:,None]).sum(-1).clamp(1e-6,1-1e-6),s)
    return -.24*o['log_density'].mean()/12+joint_nll+bce(o['posterior'])+.5*bce(o['moment'])+.2*bce(o['prior'])

def readout_features(o,d):
    p=o['logits'].sigmoid()
    b,k,_=p.shape
    i,j=ordered(k,p.device)
    def compare(q):
        # One shared latent h per layout, with conditionally independent branch outcomes.
        plus=(p[:,i]*(1-p[:,j])*q[:,None]).sum(-1)
        minus=((1-p[:,i])*p[:,j]*q[:,None]).sum(-1)
        law=torch.stack([minus,1-minus-plus,plus],-1).clamp_min(1e-7)
        return (p*q[:,None]).sum(-1),law.reshape(b,k,k-1,3)
    out=dict(base=d['base'])
    out['prior'],prior=compare(o['prior'])
    out['moment'],_=compare(o['moment'])
    out['posterior'],post=compare(o['posterior'])
    out['shape']=post
    out['kl']=(post*(post.log()-prior.log())).sum(-1,keepdim=True).clamp_min(0)
    return out

class Update(nn.Module):
    def __init__(self):
        super().__init__()
        self.msg=mlp(6,16,24)
        self.update=mlp(2+32,1,32)
        self.register_buffer('scale',torch.tensor(1.))
    def forward(self,f,mode):
        base=f['base']
        p=f[mode] if mode in ('prior','moment') else f['posterior']
        b,k=p.shape
        i,j=ordered(k,p.device)
        delta=(p[:,i]-p[:,j]).reshape(b,k,k-1,1)
        db=(base[:,i]-base[:,j]).reshape_as(delta)
        shape=f['shape'] if mode in ('law','kl') else torch.zeros_like(f['shape'])
        kl=f['kl']/self.scale if mode=='kl' else torch.zeros_like(f['kl'])
        msg=self.msg(torch.cat([delta,db,shape,kl],-1))
        return base+self.update(torch.cat([base[...,None],p[...,None],msg.mean(2),msg.amax(2)],-1))[...,0]

def restore(dev='cuda'):
    belief=Belief().to(dev)
    belief.load_state_dict(torch.load(OUT/'belief.pt',map_location=dev)['state'])
    score=Update().to(dev)
    score.load_state_dict(torch.load(OUT/'belief-readout.pt',map_location=dev)['state'])
    return belief.eval(),score.eval()

def train():
    torch.manual_seed(271)
    torch.set_num_threads(2)
    dev='cuda'
    nets=worlds(dev)
    tr,pol,val=[[panel(dom,s,*nets,dev) for dom in DOMAINS] for s in ('world','policy','val')]
    def batch(rows):
        ds=[take(d,torch.randint(len(d['w']),(48,),device=dev)) for d in rows]
        return {k:torch.cat([d[k] for d in ds]) for k in ds[0]}
    belief=Belief().to(dev)
    w=torch.cat([d['w'] for d in tr])
    belief.center.copy_(w.mean(0))
    belief.scale.copy_(w.std(0).clamp_min(.08))
    def fit(net,steps,phase,train_loss,val_loss):
        opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.012)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00008)
        best=float('inf')
        hist=[]
        for n in range(1,steps+1):
            v=train_loss()
            opt.zero_grad()
            v.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.)
            opt.step()
            sched.step()
            if n%80==0:
                with torch.no_grad():
                    v=float(val_loss())
                hist.append([n,v])
                if v<best:
                    best=v
                    torch.save(dict(step=n,state=net.state_dict()),OUT/f'{phase}.pt')
        net.load_state_dict(torch.load(OUT/f'{phase}.pt',map_location=dev)['state'])
        return hist
    def worldloss(d):
        return objective(belief(d),d['target'])
    hist=fit(belief,1440,'belief',lambda:worldloss(batch(tr)),lambda:sum(worldloss(d) for d in val)/2)
    with torch.no_grad():
        def prepare(ds):
            return [dict(readout_features(belief(d),d),target=d['target'],w=d['w']) for d in ds]
        pol,val=prepare(pol),prepare(val)
    net=Update().to(dev)
    net.scale.copy_(torch.cat([d['kl'].flatten() for d in pol]).mean().clamp_min(.01))
    def ploss(d):
        return sum(action_loss(net(d,m),d['target']) for m in MODES)/len(MODES)
    phist=fit(net,800,'belief-readout',lambda:ploss(batch(pol)),lambda:sum(ploss(d) for d in val)/2)
    ranking={}
    for mode in ('stage25',*MODES):
        selected=[]
        for d in val:
            v=d['base'] if mode=='stage25' else net(d,mode)
            y=d['target'][torch.arange(len(v),device=dev),v.argmax(1)]
            selected.append(dict(success=int(y[:,0].sum()),n=len(y),value=float((y[:,0]+.05*(y[:,1]+y[:,2])).mean().detach())))
        ranking[mode]=selected
    pick=max(ranking,key=lambda m:(sum(r['success']/r['n'] for r in ranking[m]),sum(r['value'] for r in ranking[m])))
    (OUT/'belief-selection.json').write_text(json.dumps(dict(mode=pick,validation=ranking,world=hist,readout=phist),indent=2))
    print('belief',pick,ranking,flush=True)

if __name__=='__main__':
    train()
