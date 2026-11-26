"""Predict local motion before each graph update and integrate its received effect."""
import torch
from torch import nn

from whatif.mechanism.model import mlp, pairmask, pool, tile
from .data import clock
from .model import FACTORS, FEATURE, WIDTH, World


class MotionWorld(World):
    def __init__(self, substeps):
        super().__init__()
        self.substeps=substeps
        del self.delta
        self.endpoint=mlp(2*FACTORS*WIDTH+27,FEATURE,FEATURE)
        self.propose=nn.ModuleList([mlp(WIDTH+8,3,FEATURE) for _ in range(FACTORS)])
        self.response=nn.ModuleList([mlp(2*WIDTH+2,3,FEATURE) for _ in range(FACTORS)])
        for obj in (*self.propose,*self.response):
            nn.init.normal_(obj[-1].weight,std=.018)
            nn.init.zeros_(obj[-1].bias)

    def forward(self, d, removed=None, *, soft=0.):
        # Center the coordinate frame on the moving objects in the shared source state.
        xyz=pool(d['x'][...,:3],d['role'][...,0])
        shift=torch.nn.functional.pad(xyz[...,:2],(0,7))
        centered=dict(d,x=d['x']-shift[:,None],fact=d['fact']-shift[:,None,None],goal=d['goal']-shift)
        out=super().forward(centered, removed, soft=soft)
        out['pose']=out['pose']+shift[:,None,None,None]
        return out

    def rollout(self, d, node, facts, action, reset, keep=None, soft=0.):
        x,role=d['x'],d['role']
        b,k,n=action.shape[:3]
        mask=tile(pairmask(role),k)
        prior=tile(facts['graph'],k)
        omit=torch.ones_like(mask) if keep is None else keep.flatten(0,1)
        ctx=tile(node,k)
        cmd=action.flatten(0,1)
        pose=tile(x,k)+reset.flatten(0,1)
        state=torch.stack([f(torch.cat([ctx,pose[...,i:i+3]],-1))
                           for i,f in zip((0,3,6),self.factor)],-2)
        message=torch.zeros_like(state)
        observed=(d['fact'][:,-1]-d['fact'][:,-2])*(1-d['same'][:,None,None])
        last=tile(observed,k)
        widths=d['widths'].repeat_interleave(self.substeps,dim=1)/self.substeps
        times=clock(widths).unflatten(1,(-1,self.substeps))
        carry=torch.zeros_like(prior)
        outputs=[]
        for window in times.unbind(1):
            scores,magnitudes=[],[]
            for t in window.unbind(1):
                time=tile(t,k)[:,None].expand(-1,n,-1)
                code=self.action(torch.cat([cmd,time],-1)).unflatten(-1,(FACTORS,WIDTH))
                state=state+cmd[...,0,None,None]*code/self.substeps
                local=torch.stack([f(torch.cat([state[:,:,i], p, delta, time],-1))
                                   for i,(f,p,delta) in enumerate(zip(self.propose,pose.split(3,-1),last.split(3,-1)))],-2)
                proposed=pose+local.flatten(-2)
                end=self.endpoint(torch.cat([state.flatten(-2),message.flatten(-2),
                                             pose,last,local.flatten(-2)],-1))
                weights,logit,carry,magnitude=self.interaction(end,proposed,prior,time[:,0],mask,omit,carry,soft)
                message=torch.einsum('bijfg,bifh->bjgh',weights,self.message(state))
                node_time=time[:,:,None].expand(-1,-1,FACTORS,-1)
                state=state+self.recurrent(torch.cat([message,node_time],-1).flatten(0,2),
                                           state.flatten(0,2)).reshape_as(state)
                change=[]
                for i,f in enumerate(self.response):
                    h, m=state[:,:,i],message[:,:,i]
                    change.append(f(torch.cat([h,m,time],-1))-f(torch.cat([h,torch.zeros_like(m),time],-1)))
                next_pose=proposed+torch.cat(change,-1)
                last=next_pose-pose
                pose=next_pose
                scores.append(logit)
                magnitudes.append(magnitude)
            # Window contact indicates an edge opened at some point in that window.
            outputs.append((pose,torch.stack(scores).amax(0),torch.logit(carry.clamp(1.e-5,1-1.e-5)),
                            torch.stack(magnitudes).amax(0),state))
        return [torch.stack(v,1).unflatten(0,(b,k)) for v in zip(*outputs)]
