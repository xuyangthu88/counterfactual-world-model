"""PETS trajectory sampling with upstream CEM; common-action real closed-loop testing."""
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from .evaluate import action_scores,load_model
from .models import OUT, ID, TASKS, actions, difference, geometry_cover, panel, particles, utility
from .tasks import BASE
from .train import pets_features


class PETS:
    def __init__(self,device):
        from mbrl.models import GaussianMLP
        from mbrl.planning.trajectory_opt import CEMOptimizer
        self.net=GaussianMLP(22,len(ID),device,num_layers=3,ensemble_size=5,hid_size=128)
        checkpoint=torch.load(OUT/'pets.pt',map_location=device,weights_only=True)
        self.net.load_state_dict(checkpoint.pop('net'));self.net.eval()
        for k,v in checkpoint.items():setattr(self,k,v)
        self.horizon=3
        self.cem=CEMOptimizer(3,.15,48,[[-1.15,-.42,.34]]*3,[[1.15,1.65,1.10]]*3,.1,device)
        self.mid=self.xm.new_tensor([[0.,0.,.65]]*3)
        self.scale=self.xm.new_tensor([1.15,.8,.5])

    def calibrate(self, d):
        # Select the native-geometry readout on the same validation panel as our controller.
        s=d['state'][:,1]
        a=actions(s).expand(len(s),-1,-1)
        zs=s[:,None].expand(len(s),a.shape[1],-1)
        x=(pets_features(zs,a)-self.xm)/self.xs
        mean,_=self.net(x.reshape(-1,22),use_propagation=False)
        delta=(mean*self.ys+self.ym).reshape(5,len(s),a.shape[1],-1)
        xy=zs[None,...,:2]+delta[...,:2]
        yaw=zs[None,...,2]+delta[...,2]
        learned=(zs[None,...,12]+delta[...,ID.index(12)]).clamp(0.,1.)
        area=geometry_cover(xy,yaw,zs[None])
        ids=torch.arange(len(s),device=s.device)
        def score(blend):
            choices=utility(torch.lerp(learned,area,blend),xy,yaw,zs[None]).mean(0).argmax(1)
            y=d['end'][ids,1,choices]
            return utility(y[:,12],y[:,:2],y[:,2],s).mean().item()
        return max((0.,.5,1.),key=score)

    def plan(self,s,seed):
        # Model member fixed per particle (TS-infinity); common standardized
        # process draws across competing candidate sequences at each update.
        torch.manual_seed(seed)
        lib=actions(s)
        noise=torch.randn(self.horizon,5,1,len(ID),device=s.device)
        def discretize(x):
            return (((x[...,None,:]-lib)/self.scale).square().sum(-1)).argmin(-1)
        def objective(seq):
            cmds=lib[discretize(seq)]
            z=s.expand(5,len(cmds),-1)
            total=0.
            for j in range(self.horizon):
                a=cmds[None,:,j].expand(5,-1,-1)
                x=(pets_features(z,a)-self.xm)/self.xs
                mean,logvar=self.net(x,use_propagation=False)
                change=(mean+noise[j]*(.5*logvar).exp())*self.ys+self.ym
                updated=z.clone()
                updated[...,ID]=z[...,ID]+change
                updated[...,12]=updated[...,12].clamp(0,1)
                z=updated
                cov=z[...,12]
                if self.blend:
                    cov=torch.lerp(cov,geometry_cover(z[...,:2],z[...,2],z),self.blend)
                total=total+(.85**j)*utility(cov,z[...,:2],z[...,2],z)
            return total.mean(0)
        seq=self.cem.optimize(objective,self.mid)
        return int(discretize(seq[0]))


@torch.no_grad()
def run(args):
    model=load_model('ours',args.device)
    cfg=json.loads((OUT/'calibration.json').read_text())['ours']
    pets=PETS(args.device)
    pets.blend=pets.calibrate(panel('val',args.device))
    (OUT/'pets_calibration.json').write_text(json.dumps({'blend':pets.blend})+'\n')
    from .value import load_value
    policies=args.policies
    value=load_value(args.device) if any(p.startswith('value') for p in policies) else None
    traces=[]
    root=Path.cwd()
    simenv={**os.environ,'PYTHONPATH':str(root/'work/stage7/upstream/pusht')+os.pathsep+str(root),
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','SDL_VIDEODRIVER':'dummy','PYGAME_HIDE_SUPPORT_PROMPT':'1'}
    for task in args.tasks:
        h=10 if task=='causalworld' else 20
        goal=.9 if task=='causalworld' else .95
        start=BASE[args.split] if args.start is None else args.start
        seeds=list(range(start,start+args.layouts))
        episodes={(p,i):dict(history=[0],posterior=None,previous=None,finished=False)
                  for p in policies for i in seeds}
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/'jobs.json';dest=Path(tmp)/'out.json'
            for t in range(h):
                keys=sorted({(i,tuple(v['history'])) for (_,i),v in episodes.items() if not v['finished']})
                if not keys:break
                source.write_text(json.dumps([[task,i,his,args.split] for i,his in keys]))
                subprocess.run([args.sim_python,'-m','whatif.crossbench.rollout_worker',str(source),str(dest),
                                '--workers',str(args.workers)],check=True,env=simenv)
                observed=dict(zip(keys,json.loads(dest.read_text())))
                for p in policies:
                    group=[i for i in seeds if not episodes[p,i]['finished']]
                    if not group:continue
                    rec=[episodes[p,i] for i in group]
                    values=[observed[i,tuple(v['history'])] for i,v in zip(group,rec)]
                    def ten(x):return torch.as_tensor(np.array(x),dtype=torch.float32,device=args.device)
                    curr=ten([r[1] for r in values])
                    prev=ten([r[0] if v['previous'] is None else v['previous'] for r,v in zip(values,rec)])
                    ids=torch.arange(len(group),device=curr.device)
                    ref=torch.tensor([v['history'][-1] for v in rec],device=curr.device)
                    if p=='pets':
                        choice=[pets.plan(s,11+int(i)*31+t) if s[12]<=goal and t+1<h else 0
                                for s,i in zip(curr,group)]
                    else:
                        pred,_,_=particles(model,prev)
                        err=difference(pred[ids,ref],difference(curr[:,:3],prev[:,:3])[:,None])
                        sigma=curr.new_tensor([1.,1.,1.6])*cfg['sigma']
                        loglike=-.5*(err/sigma).square().sum(-1)
                        hist=ten([np.ones(3)/3 if v['posterior'] is None else v['posterior'] for v in rec])
                        logp=.85*hist.clamp_min(1e-6).log()+torch.logsumexp(loglike.reshape(-1,3,3),dim=-1)
                        prob=logp.softmax(-1)
                        w=prob.repeat_interleave(3,-1)/3
                        if p=='prior':w=torch.ones_like(w)/9
                        delta,cov,_=particles(model,curr)
                        rule=p if p in ('distance','task') else cfg['rule']
                        strength=float(p.removeprefix('value')) if p.startswith('value') else 0.
                        choice=action_scores(curr,delta,cov,w,rule,cfg['blend'],value,strength).argmax(-1).tolist()
                        for v,pp in zip(rec,prob.cpu().numpy()):v['posterior']=pp
                    for j,(i,v,r) in enumerate(zip(group,rec,values)):
                        cover=float(r[1][12])
                        v['finished']=cover>goal
                        traces.append(dict(task=task,policy=p,scene=i,protocol=r[3],macro=t+1,
                                           action=v['history'][-1],steps=r[2],coverage=cover,
                                           distance=float(np.linalg.norm(np.array(r[1][:2])-r[1][9:11])),
                                           success=int(v['finished'])))
                        v['previous']=r[1];v['history'].append(choice[j])
                print(task,t+1,len(keys),flush=True)
    with (OUT/args.output).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=traces[0],lineterminator='\n')
        writer.writeheader();writer.writerows(traces)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--sim-python',required=True);ap.add_argument('--device',default='cpu')
    ap.add_argument('--layouts',default=64,type=int);ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--split',choices=['control','controlval'],default='control')
    ap.add_argument('--policies',nargs='+',default=['ours','pets','task','distance','prior'])
    ap.add_argument('--output',default='rollouts.csv')
    ap.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    ap.add_argument('--start',type=int)
    args=ap.parse_args();torch.set_num_threads(2);run(args)
