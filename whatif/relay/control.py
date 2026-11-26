"""Commit choices from real factual prefixes before each native segment is executed."""
import argparse
import json
import numpy as np
import torch
from whatif.hitting.fit import restore as old25
from whatif.hitting.policy import features as old_features,selector as old_selector
from whatif.hitting.evidence import restore as old_belief,readout_features as belief_features
from whatif.recourse.fit import restore as old24
from whatif.recourse.anchor import restore as old_anchor,full_feature
from whatif.hitting.data import menus,take
from whatif.mechanism.model import pool
from .native import DOMAINS,DATA,OUT,CONTROL,VALIDATION,FOLLOWUP
from .data import live
from .model import restore
from .policy import MODES,load,features
READ=('direct',*MODES)

def rank(world,policy,others,d):
    f=features(world,d)
    vals={m:policy(f,m)[0] for m in MODES}
    vals['direct']=f['m'][...,0]
    for k,net in others.items():
        vals[k]=1-net(d)[0]['p'][...,-1]
    return {m:v.argmax(1).tolist() for m,v in vals.items()}

def freeze(k,mode):
    return [[max(i-1,0),mode if i else 'hold'] for i in k]

def belief_update(old,belief,readout,batch,base):
    x=batch['hy'][:,0]
    role=batch['role']
    a=np.stack([menus(v,w,u) for v,w,u in zip(x.cpu().numpy(),role.cpu().numpy(),batch['goal'].cpu().numpy())])
    a=torch.tensor(a,device=x.device)
    e=dict(x=x,f0=batch['hx'][:,0],fa=batch['ha'][:,0],f1=x,role=role,goal=batch['goal'],same=batch['same'],
           a=torch.cat([a[:,:1],a],1))
    z=old.encode(e,True)[0]
    z=old.program_graph(e,z,old.core.abduct(e)[:3])[0][:,1:]
    mover=pool(x,role[...,0])
    rot=torch.stack([mover[:,3:6],mover[:,6:9],torch.cross(mover[:,3:6],mover[:,6:9],dim=-1)],-1)
    change=batch['x']-x
    w=torch.cat([(pool(change,role[...,0]).unflatten(-1,(3,3))@rot).flatten(1),
                 (pool(change[...,:3],role[...,1])[:,None]@rot)[:,0]],-1)
    d=dict(z=z,first=z[torch.arange(len(x),device=x.device),batch['first']],w=w,base=base)
    return readout(belief_features(belief(d),d),'kl').argmax(1).tolist()

@torch.no_grad()
def run(first,validation,followup=False):
    prefix='backup-' if followup else 'val-' if validation else ''
    base,n=FOLLOWUP if followup else VALIDATION if validation else CONTROL
    dev='cuda'
    net,decision=restore('dynamic',dev),load(dev)
    if followup:
        from .backup import world,read_policy,menu,programs
        net=world(dev)
        backup_policy=read_policy(dev)
    others={k:restore(k,dev) for k in ('fixed','unpaired')}
    old_early=old25('joint',True,dev)
    prev=old24(dev)
    if first:
        anchor=old_anchor(dev)
    else:
        orders=json.loads((OUT/(prefix+'first.json')).read_text())
        late=old25('joint',False,dev)
        legacy=old_selector(False,dev)
        belief,update=old_belief(dev)
    records={}
    for dom in DOMAINS:
        tag='initial' if first else 'mid'
        d,raw=live(dom,'early' if first else 'late',DATA/f'{prefix}{dom}-{tag}.npz',dev)
        records[dom]={}
        lookup={}
        for ix in torch.arange(len(d['x']),device=dev).split(24):
            v=take(d,ix)
            choices=rank(net,decision,others,v)
            if first:
                de={k:v[k] for k in ('x','role','goal','a','f0','fa','f1','same','seed')}
                full=full_feature(prev,de)
                z=prev.encode(de,8)[0]
                k24=anchor.decisions(torch.cat([full,z],1))['anchor_marginal'].argmax(1).tolist()
                de['a']=torch.cat([de['a'][:,:1],de['a']],1)
                k25=(1-old_early(de,True)[0]['p'][...,-1]).argmax(1).tolist()
                arms={f'both_{m}':[[a,m] for a in ids] for m,ids in choices.items()}
                arms.update({f'hold_{m}':[[a,'hold'] for a in ids] for m,ids in choices.items()})
                arms.update({f'late_{m}':freeze(k25,m) for m in READ})
                arms.update(stage25=freeze(k25,'stage25'),stage25belief=freeze(k25,'stage25belief'),
                            stage24=freeze(k24,'stage24'),actor=[[0,'hold']]*len(ix),fixed_policy=[[1,'hold']]*len(ix))
                if followup:
                    arms={k:arms[k] for k in ('both_kl','both_direct','late_law','stage25','stage25belief','stage24','actor','fixed_policy')}
                    f=features(net,menu(programs(v)))
                    for mode in READ:
                        q=f['m'][...,0] if mode=='direct' else backup_policy(f,mode)[0]
                        arms['backup_'+mode]=freeze(q.argmax(1).tolist(),'law')
                for name,seq in arms.items():
                    records[dom].setdefault(name,[]).extend(seq)
            else:
                base25=legacy(old_features(late,v,False),'means')
                choices['stage25']=base25.argmax(1).tolist()
                choices['stage25belief']=belief_update(old_early,belief,update,v,base25)
                choices['stage24']=prev.last_scores(v,v['first'])['mean'].argmax(1).tolist()
                for j,(s,a) in enumerate(zip(v['seed'].tolist(),v['first'].tolist())):
                    lookup[s,a]={m:q[j] for m,q in choices.items()}
        if not first:
            ids={(int(s),int(a)):bool(ok) for s,a,ok in zip(raw['seed'],raw['first'],raw['mid_alive'])}
            for m,seq in orders[dom].items():
                records[dom][m]=[[a,a if mode=='hold' or not ids[base+i,a] else lookup[base+i,a][mode]]
                                 for i,(a,mode) in enumerate(seq)]
    (OUT/(prefix+('first.json' if first else 'second.json'))).write_text(json.dumps(records))

def select():
    outcomes=json.loads((OUT/'val-episodes.json').read_text())
    r={}
    for mode in [f'{s}_{m}' for s in ('both','hold','late') for m in READ]:
        r[mode]=[np.mean(outcomes[d][mode],axis=0).tolist() for d in DOMAINS]
    key=lambda m:(sum(t[0] for t in r[m]),sum(t[1]+t[2] for t in r[m]))
    pure=max((m for m in r if not m.startswith('late')),key=key)
    mid=max((m for m in r if m.startswith('late')),key=key)
    (OUT/'selection.json').write_text(json.dumps(dict(primary=pure,common_first=mid,validation=r),indent=1))
    print('selected',pure,mid,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=['first','second','select'])
    p.add_argument('--validation',action='store_true')
    p.add_argument('--followup',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2)
    if args.job=='select':
        select()
    else:
        run(args.job=='first',args.validation,args.followup)
