import argparse
from contextlib import ExitStack
import csv
import json
import numpy as np
import torch
from whatif.shared.actions import menu
from whatif.shared.control import NativeBatch
from whatif.shared.model import grid,posterior
from whatif.shared.planning import Models,policy_scores
from .collect import START,OUT
from .filter import blank,evidence,update,gain
from .planner import cem,enumerate_cw


@torch.no_grad()
def run(args):
    m=Models(args.device)
    source=json.loads((OUT/'fit.json').read_text())['shortlist']
    configs={**source,'uniform':dict(sigma=.16,strength=0.,gain=0.)}
    seeds=list(range(START[args.split],START[args.split]+args.count))
    def tensor(x):return torch.as_tensor(np.array(x),dtype=torch.float32,device=args.device)
    results=[];steps=[]
    with ExitStack() as stack:
        native=NativeBatch(args,stack);native.start=seeds[0]
        for task in args.tasks:
            methods=args.policies
            if args.split=='control':
                locked=json.loads((OUT/'freeze.json').read_text())
                methods=[locked['model'] if n=='ours' else locked[task]['pets'] if n=='pets' else n for n in methods]
            for name in dict.fromkeys(methods):
                cfg=configs.get(name,configs['mass'])
                start=native.reset(task,seeds,'transfer_val' if args.split=='controlval' else 'control')
                s=tensor([start[i][0] for i in seeds]);prev=s.clone()
                oldact=tensor([menu(z)[1] for z in s.cpu().numpy()])
                bel=blank(s);legacy=bel[0].clone()
                cap=420 if task=='causalworld' else 280;goal=.9 if task=='causalworld' else .95
                hits=s[:,12].cpu().numpy()>goal;used=np.zeros(len(s),int);n=0
                while np.any((~hits)&(used<cap)):
                    ids=np.flatnonzero((~hits)&(used<cap));ss=s[ids];cmd=tensor([menu(z) for z in ss.cpu().numpy()])
                    batch=tuple(x[ids] for x in bel)
                    if n==0:
                        selected=np.ones(len(ids),int)
                    elif name in ('physics','cophy'):
                        selected=policy_scores(m,ss,cmd,legacy[ids],task,.25 if name=='cophy' else 0.,False,name,
                                               prev[ids],oldact[ids]).argmax(-1).cpu().numpy()
                    elif name in configs and task=='causalworld':
                        selected=enumerate_cw(m,ss,cmd,batch,cfg).cpu().numpy()
                    elif name=='legacy' and task=='causalworld':
                        selected=policy_scores(m,ss,cmd,legacy[ids],task,1.,True,'graph',prev[ids],oldact[ids]).argmax(-1).cpu().numpy()
                    else:
                        selected=[]
                        for j,i in enumerate(ids):
                            rng=11+seeds[i]*73+n
                            if name in ('legacy','pets5'):
                                hist=legacy[i] if name=='legacy' else None
                                choice=m.plan(ss[j],task,rng,.35,hist,terminal=task=='causalworld')
                            elif name=='balanced_legacy':
                                choice=cem(m,ss[j],task,rng,legacy[i],torch.zeros_like(legacy[i]))
                            elif name=='pets15':
                                choice=cem(m,ss[j],task,rng,terminal=task=='causalworld')
                            else:
                                choice=cem(m,ss[j],task,rng,batch[0][j],gain(batch,cfg)[j])
                            selected.append(choice)
                        selected=np.array(selected)
                    mv=[(seeds[i],j,min(42,cap-used[i])) for i,j in zip(ids,selected)]
                    records=native.step(mv,measure=True)
                    obs=tensor([records[seeds[i]][0] for i in ids])
                    chosen=torch.as_tensor(selected,device=ss.device); ii=torch.arange(len(ss),device=ss.device)
                    fact=cmd[ii,chosen]
                    if name in configs:
                        z,edges,_,_=grid(m.graph,ss,fact[:,None])
                        meas=tensor([records[seeds[i]][3:5] for i in ids])
                        ev=evidence(ss,obs,fact,z[:,0],edges[:,0],meas)
                        nxt=update(batch,ev,cfg)
                        for x,v in zip(bel,nxt):x[ids]=v
                    elif name in ('legacy','balanced_legacy'):
                        z=grid(m.graph,ss,fact[:,None])[0]
                        w=posterior(ss,obs,z,torch.zeros_like(chosen),.055,legacy[ids])
                        legacy[ids]=w.reshape(-1,5,3).sum(-1)
                    for j,i in enumerate(ids):
                        used[i]+=records[seeds[i]][1];hits[i]=float(obs[j,12])>goal
                        steps.append([task,name,seeds[i],n+1,int(selected[j]),int(used[i]),float(obs[j,12])])
                    prev[ids]=ss;s[ids]=obs;oldact[ids]=fact;n+=1
                for j,i in enumerate(seeds):
                    results.append(dict(task=task,policy=name,seed=i,protocol=start[i][1],success=int(hits[j]),coverage=float(s[j,12]),steps=int(used[j])))
                print(task,name,int(hits.sum()),float(s[:,12].mean()),float(used.mean()),flush=True)
                with (OUT/f'{args.tag}.csv').open('w',newline='') as f:
                    wr=csv.DictWriter(f,fieldnames=results[0],lineterminator='\n');wr.writeheader();wr.writerows(results)
    with (OUT/f'{args.tag}-steps.csv').open('w',newline='') as f:
        wr=csv.writer(f,lineterminator='\n');wr.writerow(['task','policy','seed','decision','action','steps','coverage']);wr.writerows(steps)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('split',choices=['controlval','control']);ap.add_argument('tag')
    ap.add_argument('--tasks',nargs='+',default=['causalworld','pusht']);ap.add_argument('--policies',nargs='+',required=True)
    ap.add_argument('--count',type=int,default=64);ap.add_argument('--workers',type=int,default=10)
    ap.add_argument('--device',default='cuda');ap.add_argument('--sim-python',default='/tmp/whatif-push-venv/bin/python')
    args=ap.parse_args();torch.set_num_threads(2);run(args)
