"""Particle-coupled action effects and upstream PETS trajectory sampling."""
import numpy as np
import torch
from whatif.crossbench.tasks import ACTIONS
from whatif.crossbench.models import geometry_cover
from .actions import SITES,FACTORS
from .model import (OUT,SharedGraph,CoPhy,Null,rotate,wrap,center,body_delta,grid,
                    log_measured,posterior,represent,unpack,task_value,physical)


def menus(s,task):
    if task=='causalworld':
        cmds=s.new_tensor(ACTIONS)
        d=s[...,9:11]-s[...,:2]
        phi=torch.atan2(d[...,1],d[...,0])[...,None]+cmds[:,0]-s[...,2:3]
        n=torch.stack((phi.cos(),phi.sin()),-1)
        side=torch.stack((-n[...,1],n[...,0]),-1)
        r=-n+cmds[:,1:2]*side
        tips=rotate(s[...,3:9].reshape(*s.shape[:-1],3,2)-s[...,:2].unsqueeze(-2),-s[...,2:3])
        dist=(r.unsqueeze(-2)-tips.unsqueeze(-3)).norm(dim=-1).amin(-1)
        arm=r[...,0]*n[...,1]-r[...,1]*n[...,0]
        return torch.cat((r,n,cmds[:,2].expand_as(arm)[...,None],arm[...,None],dist[...,None]),-1)
    pts=s.new_tensor(SITES);r,n=pts[:,:2],pts[:,2:]
    arm=r[:,0]*n[:,1]-r[:,1]*n[:,0]
    h=torch.cat((n,arm[:,None]/.47),-1)
    e=body_delta(s,s[...,9:12]);err=torch.cat((e[...,:2],.47*e[...,2:3]),-1)
    dot=err@h.T;norm=h.square().sum(1)
    drive=(dot/norm).clamp(.018,.36)
    agent=rotate(s[...,3:5]-s[...,:2],-s[...,2])-s.new_tensor([0.,.45])
    dest=r-.17*n;travel=(dest-agent.unsqueeze(-2)).norm(dim=-1)
    world=s[...,:2].unsqueeze(-2)+rotate(dest+s.new_tensor([0.,.45]),s[...,2:3])
    score=2*dot*drive-norm*drive.square()-.013*travel-10*(world.abs()>2.39).any(-1)
    chosen=[]
    for _ in range(4):
        j=score.argmax(-1);chosen.append(j)
        exclude=((n[j]@n.T)>.99)&((arm[j][...,None]-arm).abs()<.22)
        score=score.masked_fill(exclude,-1e4)
    ids=torch.stack(chosen,-1).repeat_interleave(3,-1)
    length=drive.gather(-1,ids)*s.new_tensor(FACTORS).repeat(4)
    return torch.cat((pts[ids],length[...,None],arm[ids][...,None],travel.gather(-1,ids)[...,None]),-1)


def mechanical(s,a):
    domain=s[...,13]
    # The executor already compensates for coasting in the commanded stroke.
    # Estimate where it will finish relative to the remaining target error.
    known=s[...,18]
    gain=(.66+.50*s[...,17]*known)*domain+(1-domain)*.95
    e=body_delta(s,s[...,9:12])
    drift=rotate(s[...,14:16],-s[...,2])*(1.+3*s[...,17]*known)[...,None]
    delta=a[...,2:4]*a[...,4:5]*gain[...,None]+drift
    theta=a[...,5]*a[...,4]*gain/(.50-.2791*domain)
    ang=wrap(e[...,2]-theta-s[...,16]*(1.+3*s[...,17]*known))
    return -((e[...,:2]-delta).square().sum(-1)+(.47*domain*ang).square()).clamp_min(1e-8).sqrt()-.012*a[...,6]


def scale(scores):
    return (scores-scores.mean(-1,keepdim=True))/(scores.amax(-1,keepdim=True)-scores.amin(-1,keepdim=True)).clamp_min(.035)


def weighted_state(z,w):
    avg=(z*w[:,None,:,None]).sum(2)
    avg[...,2]=torch.atan2((z[...,2].sin()*w[:,None]).sum(2),(z[...,2].cos()*w[:,None]).sum(2))
    return avg


def graph_scores(m,s,a,w,task,future=True):
    z,_,peak,rank=grid(m,s,a)
    u=.55*task_value(s[:,None,None],z,peak)+.45*rank
    u=u+.075*((u-u[:,1:2])/.12).tanh()
    now=(u*w[:,None]).sum(-1)
    if not future:return now,now
    # Roll out one further contact selected by the common physical controller.
    avg=weighted_state(z,w).reshape(-1,19)
    actions=menus(avg,task)
    best=mechanical(avg[:,None],actions).argmax(-1)
    cmd=actions[torch.arange(len(avg),device=s.device),best][:,None]
    nz,_,p,r=grid(m,avg,cmd)
    continuation=(.55*task_value(avg[:,None,None],nz,p)+.45*r)[:,0]
    continuation=continuation.reshape(*a.shape[:-1],-1)
    return now,.65*now+.35*(continuation*w[:,None]).sum(-1)


class Models:
    def __init__(self,device):
        from mbrl.models import GaussianMLP
        from mbrl.planning.trajectory_opt import CEMOptimizer
        self.device=device
        for name,cls in [('graph',SharedGraph),('cophy',CoPhy),('ncii',Null)]:
            net=cls().to(device)
            net.load_state_dict(torch.load(OUT/f'{name}.pt',map_location=device,weights_only=True))
            setattr(self,name,net.eval())
        self.pets=GaussianMLP(40,16,device,num_layers=3,ensemble_size=5,hid_size=144)
        ck=torch.load(OUT/'pets.pt',map_location=device,weights_only=True);self.pets.load_state_dict(ck.pop('net'))
        self.pets.eval()
        for k,v in ck.items():setattr(self,k,v)
        self.cem=CEMOptimizer(3,.15,48,[[-1.,-1.]]*3,[[1.,1.]]*3,.12,device,clipped_normal=True)
        self.start=self.xm.new_tensor([[-.70,0.]]*3)
    def predict(self,s,a):
        x=(represent(s,a)-self.xm)/self.xs
        mu,lv=self.pets(x,use_propagation=False)
        return mu*self.ys+self.ym,(lv*.5).exp()*self.ys
    def value_readout(self,z):
        from .value import value_features
        from whatif.crossbench.models import mlp
        if not hasattr(self,'value_net'):
            self.value_net=mlp(21,1,96).to(z.device).eval()
            self.value_net.load_state_dict(torch.load(OUT/'value.pt',map_location=z.device,weights_only=True))
        return self.value_net(value_features(z)).squeeze(-1).clamp(-1.,1.65)

    def plan(self,s,task,seed,weight,hist=None,terminal=False):
        torch.manual_seed(seed)
        noise=torch.randn(3,5,1,16,device=s.device)
        def ids(p):
            return ((p[...,0]+1)*1.999).long().clamp(0,3)*3+((p[...,1]+1)*1.499).long().clamp(0,2)
        offsets=torch.randint(3,(3,5,1),device=s.device).float()-1
        ks=s.new_tensor([-1.,-.5,0.,.5,1.])[:,None]
        def simulate(choices):
            st=s.expand(5,len(choices),-1)
            total=0.;alive=st.new_ones(st.shape[:-1])
            for h in range(3):
                opts=menus(st,task)
                idx=choices[:,h].expand(5,-1)
                a=opts.gather(-2,idx[...,None,None].expand(-1,-1,1,7)).squeeze(-2)
                if hist is None:
                    mu,sd=self.predict(st,a)
                    nxt,peak=unpack(st,mu+sd*noise[h])
                    gain=task_value(st,nxt,peak)
                else:
                    shape=st.shape[:-1]
                    nxt,_,peak,rank=self.graph(st,a,ks.expand(shape),offsets[h].expand(shape))
                    gain=.55*task_value(st,nxt,peak)+.45*rank
                total=total+.85**h*alive*(weight*gain+(1-weight)*mechanical(st,a))
                # Reach is absorbing. Carry its score to the remaining horizon.
                done=(nxt[...,12]>(.9+.05*st[...,13]))
                if h<2:
                    bonus=sum(.85**j for j in range(h+1,3))
                    total=total+alive*done*bonus*(weight*gain+(1-weight)*mechanical(st,a))
                alive=alive*(~done);st=nxt
            if terminal:
                total=total+.6*.85**2*alive*self.value_readout(st)
            return total
        nominal=simulate(torch.ones((1,3),dtype=torch.long,device=s.device)) if hist is not None else None
        def objective(seq):
            score=simulate(ids(seq))
            if hist is None:return score.mean(0)
            return ((score+.12*torch.tanh((score-nominal)/.30))*hist[:,None]).sum(0)
        return int(ids(self.cem.optimize(objective,self.start)[0]))


def policy_scores(models,s,a,hist,task,weight,look,policy,prev,refcmd):
    mscore=scale(mechanical(s[:,None],a))
    if policy=='physics':return mscore
    if policy=='cophy':
        z,_=models.cophy(prev,s,refcmd,s,a)
        cov=geometry_cover(z[...,:2],z[...,2],s[:,None])
        future=s[:,None].expand(-1,len(a[0]),-1).clone()
        future[...,:3]=z;future[...,12]=cov
        score=task_value(s[:,None],future,cov)
    else:
        w=hist.repeat_interleave(3,-1)/3
        one,two=graph_scores(models.graph,s,a,w,task,future=look)
        score=two if look else one
    return (1-weight)*mscore+weight*scale(score)
