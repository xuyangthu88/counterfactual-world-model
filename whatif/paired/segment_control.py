"""Execute a selected three-macro option with feedback for the actuator's actual position."""
import argparse
import json
import multiprocessing as mp
import numpy as np
from .data import OUT
from .segments import bank,LENGTH

START=2723000
POLICIES=['actor','stage21','marginal','pair_mean','pair_law','pair_kl',
          'combined_marginal','combined_law','combined_kl']


class SegmentPlanner:
    def __init__(self,model,actor,head,policy,config):
        from whatif.mechanism.return_model import backbone,load_head
        self.model,self.actor,self.head=model,actor,head
        self.policy,self.config=policy,config
        self.bank=bank()
        if policy.startswith('combined'):
            self.teacher=backbone('graph')
            self.old=load_head('graph')
        self.reset()

    def reset(self):
        self.cursor=0
        self.counts=dict(options=0,retrieved=0,locked_steps=0,teacher_switches=0,kl_disagreements=0)

    def act(self,d,prior,decision):
        import torch
        from whatif.mechanism.return_model import features,score
        from .decision import trajectory_features
        from .option import choose
        x,role,g=[d[k][0].numpy() for k in ['x','role','goal']]
        if self.cursor:
            cmd=self.bank.command(self.choice,self.cursor,x,role,g,self.node,self.paths)
            self.cursor=(self.cursor+1)%LENGTH
            self.counts['locked_steps']+=1
            return cmd
        combined=self.policy.startswith('combined')
        if combined:
            idx=score(self.old,features(self.teacher,d),prior,'delta_kl').item()
            if idx!=0:
                self.counts['teacher_switches']+=1
                return d['a'][0,idx].numpy()
        self.node,self.paths=self.bank.plans(x,role,g)
        first=np.stack([self.bank.command(i,0,x,role,g,self.node,self.paths) for i in range(4)])
        bundle=dict(d,a=torch.tensor(first)[None],plans=torch.tensor(self.paths)[None],
                    node=torch.tensor([self.node]),remaining=torch.tensor([35.-decision]))
        f=trajectory_features(self.model,self.actor,bundle)
        i=torch.arange(4)
        o=self.head(f,i,torch.zeros_like(i))
        zeros=torch.zeros(1,4)
        which=self.policy.replace('combined_','')
        which={'law':'pair_law','kl':'pair_kl'}.get(which,which)
        self.choice=choose(o,zeros,which,self.config)
        self.counts['options']+=1
        self.counts['retrieved']+=int(self.choice!=0)
        self.counts['kl_disagreements']+=int(choose(o,zeros,'pair_law',self.config)!=choose(o,zeros,'pair_kl',self.config))
        self.cursor=1 if (self.choice or not combined) else 0
        return first[self.choice]


def main():
    from .control import batch
    p=argparse.ArgumentParser()
    p.add_argument('--sim-python',required=True)
    p.add_argument('--workers',type=int,default=24)
    args=p.parse_args()
    config=json.loads((OUT/'segment-selection.json').read_text())['config']
    opts=dict(config=config,segments=True,id=0)
    jobs=[(dom,policy,START+i,8,args.sim_python,opts) for i in range(0,128,8)
          for dom in ('causalworld','pusht') for policy in POLICIES]
    rows=[]
    with mp.get_context('fork').Pool(args.workers) as workers:
        for part in workers.imap_unordered(batch,jobs):
            rows+=part
            print('episodes',len(rows),flush=True)
    rows.sort(key=lambda r:(r['domain'],r['policy'],r['seed']))
    (OUT/'segment-control.json').write_text(json.dumps(rows,separators=(',',':')))


if __name__=='__main__':
    main()
