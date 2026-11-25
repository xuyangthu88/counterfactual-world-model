"""Matched short strokes on different faces, in the body frame of the T."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .surface import ACTIONS, C, DEST, LEVER, N, OUT, P, Tee, candidates, directions, rot, stroke

FACTORS=np.array([.45,.8,1.15])


def menu(env):
    order,drive,_=candidates(env)
    # Remove nearly identical neighboring sites from the menu to retain distinct torques.
    sites=[]
    for j in order:
        if any(np.dot(N[j],N[i])>.99 and abs(LEVER[i]-LEVER[j])<.38 for i in sites):
            continue
        sites.append(j)
        if len(sites)==8:break
    ids=np.repeat(sites,len(FACTORS))
    return ids,drive[ids]*np.tile(FACTORS,len(sites))


def context(env):
    p,R,agent,error=directions(env)
    # All entries below are observable by both controllers.
    body=env.env.block
    vel=np.asarray(body.velocity)@R
    return np.r_[agent/100,error[:2]/100,error[2]/47,
                 vel/100,float(body.angular_velocity), np.asarray(p)/256-1,
                 np.cos(body.angle),np.sin(body.angle)]


def inputs(c,j,drive):
    c=np.asarray(c); j=np.asarray(j); drive=np.asarray(drive)
    agent=c[:2]*100
    return np.concatenate([np.broadcast_to(c,(len(j),len(c))),
           P[j]/100,N[j],LEVER[j,None],drive[:,None]/100,
           np.linalg.norm(DEST[j]-agent,axis=1)[:,None]/100],axis=1).astype('float32')


def observation_delta(before,after):
    a,b=before[2],after[2]
    R=rot(a); S=rot(b)
    old=before[:2]+R@C/100;new=after[:2]+S@C/100
    dyaw=np.arctan2(np.sin(b-a),np.cos(b-a))
    return np.r_[(new-old)@R,dyaw]


def sample(job):
    seed,split,tracked=job
    if tracked:
        from .feedback import tracked as execute
    else:
        execute=stroke
    rng=np.random.default_rng(seed)
    # Include ordinary starts and late-stage alignments in the training distribution.
    init=('neartrain' if split=='train' else 'nearval') if seed%3==0 else split
    def restart():
        e=Tee(seed,init)
        e.run(ACTIONS[0])
        return e
    env=restart()
    prefix=[]
    for _ in range(rng.integers(0,10)):
        ids,lengths=menu(env)
        i=int(rng.choice([1,1,1,3,4,7]))
        j,g=int(ids[i]),float(lengths[i])
        execute(env,j,g,35,stop=False);prefix.append((j,g))
    c=context(env);start=env.obs()
    ids,lengths=menu(env)
    x=inputs(c,ids,lengths);damp=env.physics
    env.close()
    obs=[];hits=[];steps=[]
    for j,g in zip(ids,lengths):
        env=restart()
        for h,l in prefix:execute(env,h,l,35,stop=False)
        env.edge[:]=False
        end,n,_=execute(env,int(j),g,35,stop=False)
        cover=max(env.peak_cover,end[12]) if tracked else end[12]
        obs.append(np.r_[observation_delta(start,end),cover])
        hits.append(bool(env.edge[0]));steps.append(n)
        env.close()
    return x,np.asarray(obs,'float32'),np.asarray(hits),damp,np.asarray(steps),seed


if __name__=='__main__':
    from pathlib import Path
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='train');ap.add_argument('--count',type=int,default=460);ap.add_argument('--workers',type=int,default=30)
    ap.add_argument('--tracked',action='store_true')
    a=ap.parse_args(); start={'train':110000,'val':120000}[a.split]+(60000 if a.tracked else 0)
    with ProcessPoolExecutor(a.workers) as pool:
        rows=list(pool.map(sample,[(seed,a.split,a.tracked) for seed in range(start,start+a.count)]))
    out=dict(zip(['x','y','edge','physics','steps','seed'],map(np.asarray,zip(*rows))))
    np.savez_compressed(Path('data/crossbench')/(('tracked-' if a.tracked else 'surface-')+a.split+'.npz'),**out)
    print(out['x'].shape,flush=True)
