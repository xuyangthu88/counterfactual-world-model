"""Track a contact in the observed body frame and update a physical response gain."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .surface import ACTIONS,C,DEST,N,OUT,Tee,directions,waypoints
from .surface_data import context,inputs,menu,observation_delta
from .surface_control import setup,scores
from .surface_numpy import ensemble_predict,graph_predict

# The same ordered mechanical prior runs on every episode. Gain and coasting time
# are two properties of the same particle, updated together from the last stroke.
GAIN=np.array([.55,.85,1.25,2.,3.])
COAST=np.array([.015,.05,.15,.35,.60])


def tracked(env,j,drive,budget,stop=True):
    e=env.env;steps=0;hit=False
    env.peak_cover=float(env.obs()[12])
    def advance(target,limit):
        nonlocal steps,hit
        agent=np.asarray(e.agent.position)
        delta=target-agent
        goal=agent+delta*min(1,limit/max(1e-6,np.linalg.norm(delta)))
        _,reward,hit,_=e.step(np.clip(goal,17,495))
        env.peak_cover=max(env.peak_cover,float(reward)*.95)
        steps+=1
    # Refresh the path during approach, since both the body and the pusher can coast.
    for _ in range(14):
        p,R,agent,_=directions(env)
        if np.linalg.norm(agent-DEST[j])<5:break
        for dest in waypoints(p,R,agent,j):
            if np.linalg.norm(np.asarray(e.agent.position)-dest)>4:break
        for _ in range(2):
            if steps>=budget or (stop and hit):return env.obs(),steps,hit
            advance(dest,30.)
    p,R,_,_=directions(env)
    aim=p+R@(DEST[j]+N[j]*(2+drive))
    for _ in range(10):
        if steps>=budget or (stop and hit):break
        advance(aim,max(4.,min(16.,drive*.40)))
        if np.linalg.norm(np.asarray(e.agent.position)-aim)<2:break
    return env.obs(),steps,hit


def choose(env,hist,method,blend=.20,learned_belief=None):
    from . import surface_control
    z=context(env)
    ids,drive=menu(env)
    x=inputs(z,ids,drive)
    # Work in displacement of the object's centre of mass. Velocity is observed.
    twist=np.column_stack((x[:,14:16],x[:,16]/.47))*x[:,17:18]
    vel=np.r_[z[5:7],z[7]]
    pred=twist[:,None]*GAIN[None,:,None]+vel[None,None]*COAST[None,:,None]
    err=x[:,None,2:5]-pred;err[:,:,2]=.47*np.arctan2(np.sin(err[:,:,2]),np.cos(err[:,:,2]))
    phys=-(err**2).sum(-1)
    travel=.013*x[:,18]
    score=(phys+.03*np.tanh((phys-phys[1:2])/.16))@hist-travel
    d=np.sqrt(np.sum(z[2:4]**2)+(.47*z[4])**2)
    if method=='graph':
        est=graph_predict(surface_control.MODEL,x)
        weights=np.full(3,1/3) if learned_belief is None else learned_belief
        v=scores(x,est)@weights
        # Fade the learned long-stroke residual at precision scales.
        weight=blend*(1-np.exp(-d*d))
        score=(1-weight)*score+weight*(v-travel)
    elif method=='pets':
        mu,sd=ensemble_predict(surface_control.MODEL,x)
        offsets=np.vstack((np.eye(4),-np.eye(4)))*2.
        draws=mu[:,:,None]+sd[:,:,None]*offsets
        v=scores(x,draws.transpose(1,0,2,3).reshape(len(x),-1,4)).mean(1)
        weight=blend*(1-np.exp(-d*d));score=(1-weight)*score+weight*(v-travel)
    i=int(np.argmax(score))
    return int(ids[i]),float(drive[i]),pred[i],(est[i,:,:3] if method=='graph' else None)


def episode(job):
    split,seed,method,blend,matched=job
    env=Tee(seed,split)
    try:
        env.stop_on_success=True
        z,_=env.run(ACTIONS[0]);steps=env.steps;hit=z[12]>.95
        # Common broad prior on the effective displacement per unit of push.
        prior=np.array([.28,.30,.24,.12,.06])
        hist=prior.copy();peak=z[12];belief=np.full(3,1/3)
        while steps<280 and not hit:
            j,drive,pred,learned=choose(env,hist,method,blend,belief if matched else None)
            obs,n,hit=tracked(env,j,drive,min(36,280-steps))
            error=pred-observation_delta(z,obs)
            error[:,2]=np.arctan2(np.sin(error[:,2]),np.cos(error[:,2]))
            lp=.75*np.log(hist.clip(1e-6))+.25*np.log(prior)-.5*((error/[.16,.16,.27])**2).sum(1)
            hist=np.exp(lp-lp.max());hist/=hist.sum()
            if matched and learned is not None:
                e=learned-observation_delta(z,obs)
                e[:,2]=np.arctan2(np.sin(e[:,2]),np.cos(e[:,2]))
                lp=.85*np.log(belief.clip(1e-6))-.5*((e/[.13,.13,.25])**2).sum(1)
                belief=np.exp(lp-lp.max());belief/=belief.sum()
            z=obs;steps+=n;peak=max(peak,z[12])
        return dict(scene=seed,protocol=env.protocol,policy=method,success=int(hit),coverage=z[12],peak=peak,steps=steps,weight=blend)
    finally:env.close()


def matched_setup():
    from . import surface_control
    with np.load(OUT/'tracked-models.npz') as z:surface_control.MODEL=dict(z)


if __name__=='__main__':
    import json
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['controlval','control'],default='controlval')
    ap.add_argument('--count',type=int,default=48);ap.add_argument('--workers',type=int,default=24)
    ap.add_argument('--matched',action='store_true')
    a=ap.parse_args()
    label='tracked' if a.matched else 'feedback'
    start=(148000 if a.split=='controlval' else 151000) if a.matched else (125000 if a.split=='controlval' else 141000)
    settings=[('graph',.20),('pets',.20),('physics',0.)]
    if a.matched and a.split=='controlval':
        settings=[(p,w) for p in ('graph','pets') for w in (.12,.35,.70)]+[('physics',0.)]
    if a.matched and a.split=='control':
        with (OUT/'tracked-controlval.csv').open() as f:validation=list(csv.DictReader(f))
        choices={}
        for p in ('graph','pets'):
            def quality(w):
                group=[r for r in validation if r['policy']==p and float(r['weight'])==w]
                return np.mean([.5*(float(r['success'])+float(r['coverage'])) for r in group])
            choices[p]=max((.12,.35,.70),key=quality)
        (OUT/'tracked-selection.json').write_text(json.dumps(choices,indent=2)+'\n')
        settings=[*choices.items(),('physics',0.)]
    with ProcessPoolExecutor(a.workers,initializer=matched_setup if a.matched else setup) as pool:
        rows=list(pool.map(episode,[(a.split,i,m,w,a.matched) for i in range(start,start+a.count) for m,w in settings]))
    with (OUT/f'{label}-{a.split}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    for m,w in settings:
        g=[r for r in rows if r['policy']==m and r['weight']==w]
        print(m,w,sum(r['success'] for r in g),np.mean([r['coverage'] for r in g]),flush=True)
