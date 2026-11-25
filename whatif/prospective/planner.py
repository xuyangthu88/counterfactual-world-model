"""Keep process draws balanced within every persistent physical condition."""
import torch
from whatif.shared.model import task_value,unpack
from whatif.shared.planning import menus,mechanical,weighted_state,scale
from .filter import correct,gain,predict_grid


def missed_contact(logits,task):
    actuators=logits[...,:1] if task=='pusht' else logits
    return (-actuators).sigmoid().prod(-1)


def enumerate_cw(model,s,a,belief,cfg,contact_cost=0.):
    z,e,peak,rank,w=predict_grid(model.graph,s,a,belief,cfg)
    v=.55*task_value(s[:,None,None],z,peak)+.45*rank
    if contact_cost:v=v-contact_cost*missed_contact(e,'causalworld')
    now=((v+.075*torch.tanh((v-v[:,1:2])/.12))*w[:,None]).sum(-1)
    avg=weighted_state(z,w).reshape(-1,19);candidates=menus(avg,'causalworld')
    j=mechanical(avg[:,None],candidates).argmax(1)
    cmd=candidates[torch.arange(len(j),device=s.device),j][:,None]
    expanded=tuple(x.repeat_interleave(a.shape[1],0) for x in belief)
    y,edges,p,r,second=predict_grid(model.graph,avg,cmd,expanded,cfg)
    follow=.55*task_value(avg[:,None,None],y,p)+.45*r
    if contact_cost:follow=follow-contact_cost*missed_contact(edges,'causalworld')
    later=(follow[:,0]*second).sum(-1).reshape_as(now)
    return (.65*now+.35*later).argmax(1)


def cem(model,s,task,seed,belief=None,correction=None,terminal=False,contact_cost=0.,readout=None,retain_heads=False,event_weight=None,endpoint_ablation=False):
    # Equal 5x3 rollout budget for the graph and the Gaussian ensemble.
    torch.manual_seed(seed)
    eps=torch.randn(3,5,1,16,device=s.device)
    ks=s.new_tensor([-1.,-.5,0.,.5,1.])[:,None].expand(5,3).reshape(15,1)
    latin=torch.stack([torch.randperm(3,device=s.device).float()-1 for _ in range(3)])
    if belief is not None:w=belief.repeat_interleave(3)/3;gg=correction.repeat_interleave(3)[:,None]
    def ids(p):
        return ((p[...,0]+1)*1.999).long().clamp(0,3)*3+((p[...,1]+1)*1.499).long().clamp(0,2)
    def simulate(sequence):
        count=len(sequence)
        st=s.expand(15,count,-1);alive=st.new_ones(15,count);total=0.
        for h in range(3):
            opts=menus(st,task)
            idx=sequence[:,h][None,:,None,None].expand(15,count,1,7)
            a=opts.gather(-2,idx).squeeze(-2)
            if belief is None:
                mu,sd=model.predict(st.reshape(5,3*count,19),a.reshape(5,3*count,7))
                noise=(eps[h][:,None]*s.new_tensor([-1.22474487,0.,1.22474487])[None,:,None,None])
                dy=mu.reshape(5,3,count,16)+sd.reshape(5,3,count,16)*noise
                z,peak=unpack(st,dy.reshape(15,count,16));u=task_value(st,z,peak)
            else:
                draw=latin[h].repeat(5)[:,None].expand(-1,count)
                if event_weight is None:
                    z,edge,peak,rank=model.graph(st,a,ks.expand_as(draw),draw)
                else:
                    z,edge,peak,rank,hit=model.graph.predict_event(st,a,ks.expand_as(draw),draw)
                z=correct(st,a,z,edge,gg.expand_as(draw))
                if endpoint_ablation:peak=z[...,12]
                u=.55*task_value(st,z,peak)+.45*rank
                if contact_cost:u=u-contact_cost*missed_contact(edge,task)
            if readout is not None:
                z,geometric_peak=readout(st,z)
                u=task_value(st,z,peak if retain_heads else geometric_peak)
                if retain_heads and belief is not None:u=.55*u+.45*rank
            instant=.35*u+.65*mechanical(st,a)
            if event_weight is not None:instant=instant+event_weight*hit
            total=total+.85**h*alive*instant
            done=z[...,12]>(.90+.05*st[...,13])
            if event_weight is not None:done=hit
            if h<2:total=total+sum(.85**j for j in range(h+1,3))*alive*done*instant
            alive=alive*(1.-done.float());st=z
        if terminal:total=total+.60*.85**2*alive*model.value_readout(st)
        return total
    nominal=simulate(torch.ones(1,3,device=s.device,dtype=torch.long)) if belief is not None else None
    def score(sequence):
        returns=simulate(ids(sequence))
        if belief is None:return returns.mean(0)
        return ((returns+.12*torch.tanh((returns-nominal)/.30))*w[:,None]).sum(0)
    return int(ids(model.cem.optimize(score,model.start)[0]))
