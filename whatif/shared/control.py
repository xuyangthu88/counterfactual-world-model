"""Report real transitions after each candidate policy's own choices."""
import argparse
import csv
import json
import os
import subprocess
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from .actions import OUT,START,menu
from .model import grid,posterior
from .planning import Models,policy_scores


class NativeBatch:
    def __init__(self,args,stack):
        root=Path.cwd()
        env={**os.environ,'PYTHONPATH':str(root/'work/stage7/upstream/pusht')+os.pathsep+str(root),
             'SDL_VIDEODRIVER':'dummy','PYGAME_HIDE_SUPPORT_PROMPT':'1','OPENBLAS_NUM_THREADS':'1'}
        log=stack.enter_context(open(root/'work/stage8'/f'workers-{args.split}-{args.tag}.log','w'))
        self.workers=[stack.enter_context(subprocess.Popen([args.sim_python,'-m','whatif.shared.worker'],
                      env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1))
                      for _ in range(args.workers)]
        stack.callback(self.close)
    def close(self):
        for w in self.workers:
            if w.poll() is None:
                w.stdin.write('{"op":"quit"}\n');w.stdin.flush()
    def call(self,jobs):
        for w,j in zip(self.workers,jobs):
            w.stdin.write(json.dumps(j)+'\n');w.stdin.flush()
        result={}
        for w in self.workers:
            line=w.stdout.readline()
            if not line:raise RuntimeError('Native worker stopped. See work/stage8/worker logs.')
            for i,*row in json.loads(line): result[i]=row
        return result
    def reset(self,task,seeds,split,fixed_coast=None):
        return self.call([dict(op='reset',task=task,split=split,seeds=seeds[i::len(self.workers)],fixed_coast=fixed_coast)
                          for i in range(len(self.workers))])
    def step(self,moves,measure=False):
        groups=[[] for _ in self.workers]
        for seed,idx,budget in moves:groups[(seed-self.start)%len(groups)].append([seed,int(idx),int(budget)])
        return self.call([dict(op='step',moves=g,measure=measure) for g in groups])


@torch.no_grad()
def run(args):
    model=Models(args.device)
    cal=json.loads((OUT/'calibration.json').read_text());sigma=cal['sigma']
    name_choices=None
    if args.split=='control':
        name_choices=json.loads((OUT/'planning.json').read_text())
    if 'previous' in args.policies:
        from whatif.crossbench.evaluate import load_model,action_scores
        from whatif.crossbench.models import particles,difference
        from whatif.crossbench.value import load_value
        prev_net=load_model('ours',args.device);prev_value=load_value(args.device)
        prev_cfg=json.loads(Path('results/crossbench/calibration.json').read_text())['ours']
    logs=[];traces=[]
    def tensor(a):return torch.as_tensor(np.array(a),device=args.device,dtype=torch.float32)
    seeds=list(range(START[args.split],START[args.split]+args.count))
    with ExitStack() as stack:
        native=NativeBatch(args,stack);native.start=seeds[0]
        for task in args.tasks:
            for method in args.policies:
                if args.split=='controlval':
                    choices={'graph':[(w,h) for w in (.12,.38,.72,1.) for h in (0,1)],
                             'cophy':[(w,0) for w in (.25,.65,1.)],
                             'pets':[(w,1) for w in (.35,1.)],
                             'cemgraph':[(w,2) for w in (.35,.70,1.)],
                             'returngraph':[(w,3) for w in (.35,.70)],
                             'returnpets':[(w,3) for w in (.35,1.)],
                             'physics':[(0,0)]}[method]
                elif args.split=='transfer_val':
                    choices=json.loads((OUT/'shortlist.json').read_text())[task][method]
                else:
                    original='graph' if method in ('prior','poseonly') else method
                    choices=[(0,0)] if method=='previous' else [name_choices[task][original]]
                for weight,look in choices:
                    start=native.reset(task,seeds,args.split)
                    current=tensor([start[i][0] for i in seeds])
                    previous=current.clone()
                    refcmd=tensor([menu(z)[1] for z in current.cpu().numpy()])
                    limit=420 if task=='causalworld' else 280
                    goal=.9 if task=='causalworld' else .95
                    used=np.zeros(len(seeds),int);hits=current[:,12].cpu().numpy()>goal
                    initial=hits.copy()
                    best=current[:,12].cpu().numpy().copy()
                    hist=torch.full((len(seeds),5),.2,device=args.device)
                    oldhist=torch.full((len(seeds),3),1/3,device=args.device)
                    p=0
                    while ((used<limit)&~hits).any():
                        ids=np.flatnonzero((used<limit)&~hits);s=current[ids]
                        a=tensor([menu(st) for st in s.cpu().numpy()])
                        if p==0:
                            chosen=np.ones(len(ids),int)
                        elif method=='previous':
                            delta,cov,_=particles(prev_net,s[:,:17])
                            chosen=action_scores(s[:,:17],delta,cov,oldhist[ids].repeat_interleave(3,-1)/3,
                                                 prev_cfg['rule'],prev_cfg['blend'],prev_value,.4).argmax(-1).cpu().numpy()
                        elif method in ('pets','returnpets','cemgraph','returngraph') or look>=2:
                            use_graph=method not in ('pets','returnpets')
                            chosen=[]
                            for st,i in zip(s,ids):
                                belief=(torch.ones_like(hist[i])*.2 if method=='prior' else hist[i]) if use_graph else None
                                chosen.append(model.plan(st,task,11+seeds[i]*73+p,weight,belief,terminal=look==3))
                            chosen=np.array(chosen)
                        else:
                            weights=hist[ids] if method!='prior' else torch.ones_like(hist[ids])*.2
                            score=policy_scores(model,s,a,weights,task,weight,look,method,previous[ids],refcmd[ids])
                            chosen=score.argmax(-1).cpu().numpy()
                        moves=[(seeds[i],j,min(42,limit-used[i])) for i,j in zip(ids,chosen)]
                        observed=native.step(moves)
                        nexts=tensor([observed[seeds[i]][0] for i in ids])
                        if method in ('graph','cemgraph','returngraph','prior','poseonly'):
                            pred=grid(model.graph,s,a)[0]
                            post=posterior(s,nexts,pred,torch.as_tensor(chosen,device=s.device),sigma,hist[ids],passive=method!='poseonly')
                            hist[ids]=post.reshape(-1,5,3).sum(-1)
                        if method=='previous':
                            old=particles(prev_net,s[:,:17])[0]
                            err=difference(old[torch.arange(len(ids),device=s.device),torch.as_tensor(chosen,device=s.device)],
                                           difference(nexts[:,:3],s[:,:3])[:,None])
                            ll=-.5*(err/(s.new_tensor([1.,1.,1.6])*prev_cfg['sigma'])).square().sum(-1)
                            oldhist[ids]=(.85*oldhist[ids].clamp_min(1e-6).log()+torch.logsumexp(ll.reshape(-1,3,3),-1)).softmax(-1)
                        previous[ids]=s;current[ids]=nexts
                        refcmd[ids]=a[torch.arange(len(ids),device=s.device),torch.as_tensor(chosen,device=s.device)]
                        for ii,(i,j) in enumerate(zip(ids,chosen)):
                            _,n,peak=observed[seeds[i]]
                            used[i]+=n;best[i]=max(best[i],peak);hits[i]=float(nexts[ii,12])>goal
                            traces.append((task,method,weight,look,seeds[i],p+1,int(j),used[i],float(nexts[ii,12])))
                        p+=1
                    rows=[dict(task=task,policy=method,weight=weight,look=look,seed=i,protocol=start[i][1],initial_success=int(initial[j]),success=int(hits[j]),
                               steps=int(used[j]),coverage=float(current[j,12]),peak=float(best[j])) for j,i in enumerate(seeds)]
                    logs.extend(rows)
                    print(task,method,weight,look,int(hits.sum()),round(float(current[:,12].mean()),4),round(used.mean(),1),flush=True)
                    with (OUT/f'{args.tag}.csv').open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(logs)
    with (OUT/f'{args.tag}-steps.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['task','policy','weight','look','seed','decision','action','native_steps','coverage']);writer.writerows(traces)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('split',choices=['controlval','transfer_val','control']);ap.add_argument('tag')
    ap.add_argument('--policies',nargs='+',default=['graph','cophy','pets','physics'])
    ap.add_argument('--tasks',nargs='+',default=['causalworld','pusht']);ap.add_argument('--count',type=int,default=48)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--device',default='cuda')
    ap.add_argument('--sim-python',default='/tmp/whatif-push-venv/bin/python')
    args=ap.parse_args();torch.set_num_threads(2);run(args)
