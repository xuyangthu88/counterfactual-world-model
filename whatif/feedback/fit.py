"""Learn multi-horizon consequences and a joint discrete law on matched feedback policies."""
import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
from whatif.mechanism.model import labeled, pool
from whatif.mechanism.train import avg
from .model import FeedbackWorld, ATOMS, preference
from .smoother import HistoryFacts
from .native import OUT, DATA, DOMAINS, command

def bank(expanded=True, history=False):
    name = 'expanded-bank.json' if expanded else 'bank.json'
    if history:
        name = 'history-bank.json'
    a = np.array(json.loads((OUT/name).read_text())['bank'], np.float32)
    return np.pad(a,((0,0),(0,11-a.shape[1])))

def data(domain, split, device, expanded=True, history=False):
    name = ('expanded-' if expanded and split!='observe' else '')+domain+'-'+split+'.npz'
    if history:
        suffix = 'control' if split=='observe' else split
        name = f'history-{domain}-{suffix}.npz'
    with np.load(DATA/name) as z:
        raw = {k:z[k] for k in z.files}
    weights = bank(expanded,history)
    if split=='observe':
        raw['a'] = np.stack([[command(x,r,g,w) for w in weights]
                              for x,r,g in zip(raw['x'],raw['role'],raw['goal'])])
    elif not expanded or 'native' not in raw:
        raw['native'] = np.zeros_like(raw['outcome'][...,0])
        raw['native'][:,:,0] = raw['r'][:,:,0]
        raw['known'] = np.zeros_like(raw['native'])
        raw['known'][:,:,0] = 1
    else:
        raw['known'] = np.ones_like(raw['native'])
    raw['weights'] = np.broadcast_to(weights,(len(raw['x']),*weights.shape)).copy()
    return {k:torch.tensor(v,device=device,dtype=torch.float32) for k,v in raw.items()}

def subset(d, ix):
    return {k:v[ix] for k,v in d.items()}

def pairs(k,device):
    return tuple(torch.triu_indices(k,k,1,device=device))

def objective(net, d):
    o = net(d)
    role = d['role']
    mover = role[...,0]
    i,j = pairs(d['a'].size(1),d['x'].device)
    mu,scale,p = net.contrasts(o,i,j,mover)
    edge_mask = labeled(role)
    e = F.binary_cross_entropy_with_logits(o['edge'],d['contact'],reduction='none')
    fact = F.binary_cross_entropy_with_logits(o['fact'],d['fe'],reduction='none')
    pose = F.smooth_l1_loss(o['y'],d['future'],beta=.18,reduction='none')
    pos = pool(d['future'][...,:3],mover[:,None,None,:])
    target = torch.cat([pos,d['native'][...,None]],-1)
    diff = target[:,i]-target[:,j]
    valid = torch.ones_like(diff)
    valid[...,3] = d['known'][:,i]*d['known'][:,j]
    err = mu-diff
    native = avg((o['scalar'][...,3]-d['native']).square(),d['known'])
    outcome = d['outcome']
    last = outcome[:,:,-1]
    disc = (last[:,i,0]-last[:,j,0]).long()+1
    aux = last[...,1]*.5+last[...,2]*.25
    t = ((aux[:,i]-aux[:,j]+.75)*(ATOMS-1)/1.5).clamp(0,ATOMS-1)
    low = t.floor().long()
    within = p.gather(-2,disc[...,None,None].expand(-1,-1,1,ATOMS)).squeeze(-2)
    def nll(ix):
        return -within.gather(-1,ix[...,None]).squeeze(-1).clamp_min(1e-8).log()
    ce = ((1-t.frac())*nll(low)+t.frac()*nll((low+1).clamp_max(ATOMS-1))).mean()
    scale_loss = .5*(err/scale).square()+scale.log()
    # Distinguish the rare success reversals even when many return differences lie near zero.
    event = p.sum(-1).clamp_min(1e-8)
    discord = disc!=1
    conditional = (event[...,2]/(event[...,0]+event[...,2])).clamp(1e-6,1-1e-6)
    sign_loss = avg(F.binary_cross_entropy(conditional,(disc==2).float(),reduction='none'),discord)
    return (ce + .55*sign_loss + F.binary_cross_entropy_with_logits(o['scalar'][...,0],outcome[...,0])
            + 2*F.mse_loss(o['scalar'][...,1:3].sigmoid(),outcome[...,1:])
            + avg(pose,mover[:,None,None,:,None])+.5*avg(e,edge_mask[:,None,None])
            +.15*avg(fact,edge_mask)+native+.4*avg(err.abs(),valid)
            +.08*avg(scale_loss,valid)+.002*o['kl'])

@torch.no_grad()
def validate(net, d):
    losses, utility = [], []
    for ids in torch.arange(len(d['x']),device=d['x'].device).split(40):
        row = subset(d,ids)
        losses.append(objective(net,row).item()*len(ids))
        o = net(row)
        i,j = pairs(row['a'].size(1),ids.device)
        p = net.contrasts(o,i,j,row['role'][...,0])[2]
        scores = preference(net,o,i,j,p)
        last = row['outcome'][:,:,-1]
        g = last[...,0]+.5*last[...,1]+.25*last[...,2]
        utility.append(torch.stack([g.gather(1,s.argmax(1,keepdim=True))[:,0]-g[:,0]
                                    for s in scores.values()],-1))
    return sum(losses)/len(d['x']),torch.cat(utility).mean(0).tolist()

def run(device, epochs, history=False):
    torch.set_num_threads(2)
    torch.manual_seed(271)
    tag = 'history-' if history else ''
    splits = ([data(d,'train',device,history=True) for d in DOMAINS] if history else
              [data(d,'train',device,exp) for exp in (False,True) for d in DOMAINS])
    train = {k:torch.cat([s[k] for s in splits]) for k in splits[0]}
    val = [data(d,'val',device,history=history) for d in DOMAINS]
    net = FeedbackWorld().to(device)
    if history:
        net.facts = HistoryFacts().to(device)
        net.load_state_dict(torch.load(OUT/'world.pt',map_location=device)['state'], strict=False)
        train = subset(train,train['alive'].bool())
        val = [subset(d,d['alive'].bool()) for d in val]
    opt = torch.optim.AdamW(net.parameters(), lr=.0008 if history else .0014)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00016)
    best = np.inf
    log = []
    for epoch in range(1,epochs+1):
        net.train()
        for ix in torch.randperm(len(train['x']),device=device).split(80):
            loss = objective(net,subset(train,ix))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),4.)
            opt.step()
        sched.step()
        net.eval()
        result = [validate(net,s) for s in val]
        value = np.mean([v for v,_ in result])
        log.append(dict(epoch=epoch,loss=value,by_domain=result))
        if value<best:
            best = value
            torch.save({'epoch':epoch,'state':net.state_dict()},OUT/(tag+'world.pt'))
        if epoch%4==0 or epoch==1:
            print(epoch,result,flush=True)
            (OUT/(tag+'fit.json')).write_text(json.dumps(log,indent=1))
    (OUT/(tag+'fit.json')).write_text(json.dumps(log,indent=1))

def restore(device='cpu', history=False):
    net = FeedbackWorld().to(device)
    if history:
        net.facts = HistoryFacts().to(device)
    saved = torch.load(OUT/('history-world.pt' if history else 'world.pt'),map_location=device)
    net.load_state_dict(saved['state'])
    return net.eval()

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device',default='cuda')
    p.add_argument('--epochs',type=int,default=64)
    p.add_argument('--history',action='store_true')
    args = p.parse_args()
    run(args.device,args.epochs,args.history)
