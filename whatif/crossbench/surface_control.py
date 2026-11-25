"""Compare short contact decisions under identical native actions and step budgets."""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .surface import ACTIONS, OUT, Tee, stroke
from .surface_data import context, inputs, menu, observation_delta
from .surface_numpy import ensemble_predict, graph_predict

MODEL=None


def setup():
    global MODEL
    with np.load(OUT/'surface-models.npz') as z:MODEL=dict(z)


def scores(x,response):
    delta=response[...,:3]
    gap=x[:,None,2:5]-delta
    gap[...,2]=.47*np.arctan2(np.sin(gap[...,2]),np.cos(gap[...,2]))
    cost=(gap**2).sum(-1)
    cov=response[...,3].clip(0,1)
    return -cost+.22*cov+.10/(1+np.exp(-(cov-.92)/.05))


def decision(env,method,weight,posterior):
    j,drive=menu(env)
    x=inputs(context(env),j,drive)
    geom=np.column_stack((x[:,14:16],x[:,16]/.47))*x[:,17:18]
    gap=x[:,2:5]-geom
    gap[:,2]=.47*np.arctan2(np.sin(gap[:,2]),np.cos(gap[:,2]))
    prior=-(gap**2).sum(1)
    predictions=None
    if method=='graph':
        predictions=graph_predict(MODEL,x)
        utility=scores(x,predictions)
        # Pair the chosen contact and the nominal contact under each physical draw.
        val=(utility+.07*np.tanh((utility-utility[1:2])/.12))@posterior
    elif method=='pets':
        mu,sd=ensemble_predict(MODEL,x)
        # Deterministic cubature of the five Gaussian ensemble components.
        offsets=np.vstack((np.eye(4),-np.eye(4)))*2.
        draws=mu[:,:,None]+sd[:,:,None]*offsets
        val=scores(x,draws.transpose(1,0,2,3).reshape(len(x),-1,4)).mean(1)
    else:
        val=prior
    score=(1-weight)*prior+weight*val-.015*x[:,18]
    i=int(np.argmax(score))
    return int(j[i]),float(drive[i]),None if predictions is None else predictions[i,:,:3]


def play(job):
    seed,split,method,weight=job
    env=Tee(seed,split)
    try:
        env.stop_on_success=True
        z,_=env.run(ACTIONS[0])
        used=env.steps;hit=z[12]>.95;peak=z[12]
        hist=np.full(3,1/3)
        pushes=1
        while used<280 and not hit:
            j,drive,pred=decision(env,method,weight,hist)
            obs,n,hit=stroke(env,j,drive,min(35,280-used))
            peak=max(peak,obs[12])
            if pred is not None:
                err=pred-observation_delta(z,obs)
                err[:,2]=np.arctan2(np.sin(err[:,2]),np.cos(err[:,2]))
                logp=.85*np.log(hist.clip(1e-6))-.5*((err/np.array([.13,.13,.25]))**2).sum(1)
                hist=np.exp(logp-logp.max());hist/=hist.sum()
            z=obs;used+=n;pushes+=1
        return dict(scene=seed,protocol=env.protocol,policy=method,weight=weight,success=int(hit),
                    coverage=z[12],peak=peak,steps=used,pushes=pushes)
    finally:env.close()


def run(args):
    if args.split=='controlval':
        policies=[('geometry',0.)]+[(m,w) for m in ('graph','pets') for w in (.35,.7,1.)]
        start=125000
    else:
        with (OUT/'surface-controlval.csv').open() as f:rows=list(csv.DictReader(f))
        chosen={}
        for m in ('graph','pets'):
            options=sorted({float(r['weight']) for r in rows if r['policy']==m})
            def score(w):
                g=[r for r in rows if r['policy']==m and float(r['weight'])==w]
                return np.mean([.5*(int(r['success'])+float(r['coverage'])) for r in g])
            chosen[m]=max(options,key=score)
        (OUT/'surface-selection.json').write_text(json.dumps(chosen,indent=2)+'\n')
        policies=[('geometry',0.),*chosen.items()]
        start=131000
    jobs=[(i,args.split,m,w) for i in range(start,start+args.count) for m,w in policies]
    with ProcessPoolExecutor(args.workers,initializer=setup) as pool: rows=list(pool.map(play,jobs))
    with (OUT/f'surface-{args.split}.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');wr.writeheader();wr.writerows(rows)
    for m,w in policies:
        g=[r for r in rows if (r['policy'],r['weight'])==(m,w)]
        print(m,w,sum(r['success'] for r in g),round(np.mean([r['coverage'] for r in g]),4),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['controlval','control'],default='controlval')
    ap.add_argument('--count',type=int,default=48);ap.add_argument('--workers',type=int,default=28)
    run(ap.parse_args())
