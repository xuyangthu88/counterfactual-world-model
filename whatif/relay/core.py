"""Rebuild contact and directed transmission from each branch's evolving endpoints."""
import torch
from torch import nn
from torch.nn import functional as F
from whatif.hitting.core import Core as Abduction
from whatif.mechanism.model import FACTORS,WIDTH,mlp,pairs,tile,pairmask,pool

class Relay(Abduction):
    def __init__(self,fixed=False):
        super().__init__()
        self.fixed=fixed
        self.endpoint=mlp(2*FACTORS*WIDTH+9,32,56)
        self.change=mlp(2*32+3+1+2,1+2*FACTORS**2+3,64)
        self.program_source=mlp(11+7+9,FACTORS*WIDTH,56)
        self.transition=nn.GRUCell(WIDTH+2,WIDTH)
        nn.init.zeros_(self.change[-1].bias)
        nn.init.normal_(self.change[-1].weight,std=.006)

    def forward(self,d):
        x,role,a,goal=d['x'],d['role'],d['a'],d['goal']
        b,k,n=a.shape[:3]
        m,u,fact,kl=self.abduct(d)
        obs=d['f1']-d['f0']
        # Only the named actuator or intervened object receives a candidate command.
        act=torch.cat([a,d['fa'][:,None]],1)
        c=k+1
        cmd=act.flatten(0,1)
        prior=tile(fact,c)
        original=tile(x,c)
        node=tile(self.node(torch.cat([x,role,m,u,obs],-1)),c)
        xy=tile(torch.cat([x[...,:3],obs[...,:3]],-1),c)
        base=self.relation(torch.cat([pairs(torch.cat([node,cmd],-1)),prior[...,None],
                                      xy[:,None]-xy[:,:,None]],-1))
        edges=self.edge(base)
        link=self.transfer(base)
        program=d.get('program',x.new_zeros(b,k,11))
        prog=torch.cat([program,torch.zeros_like(program[:,:1])],1)
        enabled=d.get('feedback',x.new_zeros(b))[:,None].expand(-1,k)
        enabled=torch.cat([enabled,torch.zeros_like(enabled[:,:1])],1).reshape(-1,1,1)
        token=self.program_source(torch.cat([prog.flatten(0,1)[:,None].expand(-1,n,-1),cmd,
                                             tile(goal[:,None]-x,c)],-1))*enabled*cmd[...,0:1]
        state=torch.stack([f(torch.cat([node,original[...,j:j+3]],-1))
                           for j,f in zip((0,3,6),self.factor)],-2)
        state=state+(self.source(cmd)+token).unflatten(-1,(FACTORS,WIDTH))
        inbox=torch.zeros_like(state)
        width=d.get('width')
        if width is None:
            width=torch.where(d['same'][:,None]>0,x.new_full((b,5),6),x.new_tensor([1,1,2,2,2]))
        time=torch.stack([torch.log1p(width/8),width.cumsum(-1)/width.sum(-1,keepdim=True)],-1)
        pose=original
        mask=tile(pairmask(role),c)
        influence=torch.zeros_like(prior)
        histories={key:[] for key in ('position','edge','causal','impact','hidden','sparsity')}
        for t in range(5):
            clock=tile(time[:,t],c)
            if t==3 and 'adaptive' in d:
                # Intervene only after the common macro-9 history. A plan may affect
                # later edges only through its source node and subsequent messages.
                switch=torch.cat([d['adaptive'],torch.zeros_like(d['adaptive'][:,:1])],1).reshape(-1,1,1)
                z=torch.cat([state.flatten(-2),pose-original,tile(goal[:,None],c)-pose,
                             prog.flatten(0,1)[:,None].expand(-1,n,-1)],-1)
                impulse=self.recourse_source(z)*switch*cmd[...,0:1]
                state=state+impulse.unflatten(-1,(FACTORS,WIDTH))
            if not self.fixed or t==0:
                end=self.endpoint(torch.cat([state.flatten(-2),inbox.flatten(-2),pose-original],-1))
                xyz=pose[...,:3]
                edit=self.change(torch.cat([pairs(end),xyz[:,None]-xyz[:,:,None],prior[...,None],
                                            clock[:,None,None].expand(-1,n,n,-1)],-1))
                e=edges+edit[...,:-3]
                direction,magnitude=(link+edit[...,-3:-1]).unbind(-1)
                persistence=edit[...,-1].sigmoid()
                logit=prior+e[...,0]
                pc=logit.sigmoid()
                contact=(pc>=.5).to(pc)+(pc-pc.detach())
                pg,fw=e[...,1:].chunk(2,-1)
                pg=pg.sigmoid()
                factor=(pg>=.5).to(pg)+(pg-pg.detach())
                strength=F.softplus(magnitude)
                conduct=(direction+1.5).sigmoid()
                gate=contact*mask*conduct*(1+strength.tanh())
                weights=gate[...,None,None]*(factor*fw.tanh()).unflatten(-1,(3,3))
            influence=1-(1-pc*conduct)*(1-persistence*influence)
            causal=torch.logit(influence.clamp(1e-6,1-1e-6))
            inbox=torch.einsum('bijfg,bifh->bjgh',weights,state)
            tmsg=clock[:,None,None].expand(-1,n,FACTORS,-1)
            # Retain the pretrained local update and learn a time-conditioned correction.
            h=state.reshape(-1,WIDTH)
            msg=inbox.reshape(-1,WIDTH)
            delta=self.update[min(t,3)](msg,h)+.2*(self.transition(torch.cat([inbox,tmsg],-1).reshape(-1,WIDTH+2),h)-h)
            state=state+delta.reshape_as(state)
            pose=original+torch.cat([f(state[:,:,j]) for j,f in enumerate(self.delta)],-1)
            for key,v in [('position',pose),('edge',logit),('causal',causal),('impact',strength),('hidden',state.flatten(-2))]:
                histories[key].append(v.reshape(b,c,*v.shape[1:]))
            histories['sparsity'].append((gate[...,None]*pg).abs().mean())
        series={k:torch.stack(v,2)[:,:-1] for k,v in histories.items() if k!='sparsity'}
        path=series['position']
        # CoPhy supplies the successor of the factual command at its final physical time.
        anchor=d['same'][:,None,None,None]*(d['f1'][:,None]-histories['position'][-1][:,-1:])
        path=torch.cat([path[:,:,:-1],(path[:,:,-1]+anchor)[:,:,None]],2)
        hidden=series['hidden']
        mover=role[:,None,None,:,0]
        hr=torch.cat([pool(hidden,mover),pool(path,mover)-goal[:,None,None]],-1)
        y=path[:,:,-1]
        out=dict(y=y,r=self.reward(torch.cat([hr[:,:,-1],goal[:,None].expand(-1,k,-1)],-1)),
                    h=hidden[:,:,-1],hr=hr[:,:,-1],track=hr,trajectory=path,edge_time=series['edge'],
                    causal_time=series['causal'],impact_time=series['impact'],edge=series['edge'].amax(2),
                    causal=series['causal'][:,:,-1],impact=series['impact'][:,:,-1],fact=fact,kl=kl,
                    sparse=torch.stack(histories['sparsity']).mean(),context=pool(torch.cat([m,u],-1),role.sum(-1)))
        if d.get('capture',False) or (hasattr(self,'atomic_readout') and not d.get('feedback',x.new_zeros(1)).any()):
            steps=torch.stack(histories['hidden'],3).flatten(-2)
            common=torch.cat([x,obs,role,goal[:,None]-x,m,u],-1)
            packed=torch.cat([steps,common[:,None].expand(-1,c,-1,-1)],-1)
            basepose=histories['position'][-1]
            if d.get('capture',False):
                out.update(packed=packed,basepose=basepose)
            else:
                out.update(self.atomic_readout(packed,basepose,d))
        return out

    def contrast(self,out,i,j):
        if 'aleatoric' not in out:
            return super().contrast(out,i,j)
        def effect(mean,scales):
            tau,load=scales.unbind(-1)
            sd=(tau[:,i].square()+tau[:,j].square()+(load[:,i]-load[:,j]).square()).sqrt()
            return mean[:,i]-mean[:,j],sd
        return effect(out['y'],out['aleatoric']),effect(out['r'],out['reward_noise'])
