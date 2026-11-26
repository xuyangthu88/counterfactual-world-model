"""Refine the motion clock and learn signed spatial messages on position-factor links."""
import torch
from torch.nn import functional as F

from whatif.effects.model import FACTORS, FEATURE, WIDTH, relative
from whatif.effects.motion import MotionWorld
from whatif.mechanism.model import mlp, pairmask, pairs, tile


class ShortWorld(MotionWorld):
    def __init__(self):
        super().__init__(1)
        self.vector=mlp(2*FEATURE+3+2,FACTORS*6,80)
        torch.nn.init.zeros_(self.vector[-1].weight)
        torch.nn.init.zeros_(self.vector[-1].bias)

    def forward(self,d,removed=None,*,soft=0.,native=False):
        fine=super().forward(d,removed,soft=soft)
        ends=d['end_idx'][0]
        active=d['window'][:,None,:,:,None,None]
        w=d['window']*d['native_widths'][:,None]
        w=w/w.sum(-1,keepdim=True)
        o=dict(fine)
        for key in ('pose','causal'):
            o[key]=fine[key].index_select(2,ends)
        for key in ('edge','strength'):
            o[key]=fine[key][:,:,None].masked_fill(~active,-torch.inf).amax(3)
        o['reward']=torch.einsum('bst,bkt->bks',w,fine['reward'])
        a,s,l=fine['r_dist']
        o['r_dist']=(torch.einsum('bst,bkth->bksh',w,a),
                     torch.einsum('bst,bkth->bksh',w.square(),s.square()).sqrt(),
                     torch.einsum('bst,bkth->bksh',w,l))
        o['p_dist']=tuple(v.index_select(2,ends) for v in fine['p_dist'])
        if removed is not None:
            o['edge_effect']=fine['edge_effect'].index_select(3,ends)
        if native:
            o['native']=fine
        return o

    def vector_message(self,end,position,proposed,velocity,free,time,weights):
        n=position.size(1)
        x=position[...,:3]
        q=proposed[...,:3]
        speed=velocity[...,:3]
        d=free[:,:,0]

        def endpoints(v):
            return v[:,:,None].expand(-1,-1,n,-1),v[:,None].expand(-1,n,-1,-1)

        directions=torch.stack([relative(x),relative(q),*endpoints(speed),*endpoints(d)],-2)
        directions=directions/(1+directions.norm(dim=-1,keepdim=True))
        pair=torch.cat([pairs(end),relative(q),time[:,None].expand(-1,n,-1,-1)],-1)
        coef=self.vector(pair).unflatten(-1,(FACTORS,6))
        return torch.einsum('bijf,bijfl,bijlc->bjc',weights[...,0],coef,directions)

    def cumulative_displacement(self,state,initial,time):
        return None

    def rollout(self,d,node,facts,action,reset,keep=None,soft=0.):
        x,roles=d['x'],d['role']
        b,k,n=action.shape[:3]
        pose=tile(x,k)+reset.flatten(0,1)
        body=tile(node,k)
        cmd=action.flatten(0,1)
        mask=tile(pairmask(roles),k)
        omit=torch.ones_like(mask) if keep is None else keep.flatten(0,1)
        prior=tile(facts['graph'],k)
        state=torch.stack([f(torch.cat([body,q],-1)) for f,q in zip(self.factor,pose.split(3,-1))],-2)
        initial_state=state
        correction=0.
        message=torch.zeros_like(state)
        observed=(d['fact'][:,-1]-d['fact'][:,-2])*(1-d['same'][:,None,None])
        velocity=tile(observed/d['widths'][:,-1,None,None],k)
        widths=d['native_widths']
        nominal=d['nominal_widths']
        elapsed=widths.cumsum(1)/widths.sum(1,keepdim=True)
        times=torch.stack([torch.log1p(nominal/8),elapsed],-1)
        carry=torch.zeros_like(prior)
        outputs=[]
        for t,dt,parent in zip(times.unbind(1),widths.unbind(1),nominal.unbind(1)):
            time=tile(t,k)[:,None].expand(-1,n,-1)
            dt=tile(dt,k)[:,None,None]
            parent=tile(parent,k)[:,None,None]
            fraction=dt/parent
            last=velocity*parent
            source=self.action(torch.cat([cmd,time],-1)).unflatten(-1,(FACTORS,WIDTH))
            state=state+cmd[...,0,None,None]*source*fraction[...,None]
            free=torch.stack([f(torch.cat([state[:,:,i],p,dp,time],-1))
                              for i,(f,p,dp) in enumerate(zip(self.propose,pose.split(3,-1),last.split(3,-1)))],-2)
            proposed=pose+fraction*free.flatten(-2)
            end=self.endpoint(torch.cat([state.flatten(-2),message.flatten(-2),pose,last,free.flatten(-2)],-1))
            weights,logit,carry,strength=self.interaction(end,proposed,prior,time[:,0],mask,omit,carry,soft)
            message=torch.einsum('bijfg,bifh->bjgh',weights,self.message(state))
            physical=self.vector_message(end,pose,proposed,last,free,time,weights)
            node_time=time[:,:,None].expand(-1,-1,FACTORS,-1)
            step=self.recurrent(torch.cat([message,node_time],-1).flatten(0,2),state.flatten(0,2))
            state=state+fraction[...,None]*step.reshape_as(state)
            change=[]
            for i,f in enumerate(self.response):
                h,m=state[:,:,i],message[:,:,i]
                change.append(f(torch.cat([h,m,time],-1))-f(torch.cat([h,torch.zeros_like(m),time],-1)))
            future=proposed+fraction*(torch.cat(change,-1)+F.pad(physical,(0,6)))
            next_correction=self.cumulative_displacement(state,initial_state,time)
            if next_correction is not None:
                future=future+F.pad(next_correction-correction,(0,6))
                correction=next_correction
            velocity=(future-pose)/dt
            pose=future
            outputs.append((pose,logit,torch.logit(carry.clamp(1.e-5,1-1.e-5)),strength,state))
        return [torch.stack(v,1).unflatten(0,(b,k)) for v in zip(*outputs)]
