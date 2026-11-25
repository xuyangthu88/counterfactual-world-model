"""Short, directed contact proposals for the native PushT actuator."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from .tasks import ACTIONS, Tee

OUT=Path('results/crossbench')
C=np.array([0.,45.])
L=47.
POINTS=np.array([[-60.,0],[60.,0],[60.,30],[15.,30],[15.,120],[-15.,120],[-15.,30],[-60.,30]])
POLY=Polygon(POINTS)
CLEAR=POLY.buffer(13.5,join_style=2)
CORNERS=np.array(POLY.buffer(19.,join_style=2).exterior.coords[:-1])


def rot(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])


def visible(a,b):
    return not LineString([a,b]).crosses(CLEAR) and not CLEAR.contains(Point((a+b)/2))


def static_routes():
    n=len(CORNERS)
    dist=np.full((n,n),np.inf)
    for i in range(n):
        for j in range(n):
            if i==j or visible(CORNERS[i],CORNERS[j]):
                dist[i,j]=np.linalg.norm(CORNERS[i]-CORNERS[j])
    for i in range(n):
        dist=np.minimum(dist,dist[:,i:i+1]+dist[i:i+1,:])
    sites=[]
    for a,b in zip(POINTS,np.roll(POINTS,-1,axis=0)):
        tangent=(b-a)/np.linalg.norm(b-a)
        outward=tangent[[1,0]]*np.array([1.,-1.])
        for t in (.12,.32,.5,.68,.88):
            pt=a+t*(b-a)
            dest=pt+17.*outward
            if POLY.distance(Point(dest))<15.5:continue
            approach=np.array([np.linalg.norm(v-dest) if visible(v,dest) else np.inf for v in CORNERS])
            via=(dist+approach[None]).min(1)
            sites.append((pt,-outward,dest,via))
    return sites


SITES=static_routes()
P,N,DEST,VIA=map(np.array,zip(*SITES))
LEVER=((P[:,0]-C[0])*N[:,1]-(P[:,1]-C[1])*N[:,0])/L
TWIST=np.column_stack((N,LEVER))


def directions(env):
    e=env.env
    p=np.asarray(e.block.position)
    R=rot(e.block.angle)
    agent=(np.asarray(e.agent.position)-p)@R
    target=e.goal_pose
    goal_com=target[:2]+rot(target[2])@C
    goal_xy=(goal_com-p)@R-C
    dyaw=np.arctan2(np.sin(target[2]-e.block.angle),np.cos(target[2]-e.block.angle))
    return p,R,agent,np.r_[goal_xy,L*dyaw]


def candidates(env):
    p,R,agent,error=directions(env)
    # A unilateral normal impulse induces a planar translation and a torque about the body centre.
    drive=(TWIST@error/(TWIST**2).sum(1)).clip(2.,35.)
    gain=(2*(TWIST@error)*drive-(TWIST**2).sum(1)*drive**2)
    dists=np.linalg.norm(DEST-agent,axis=1)
    world=DEST@R.T+p
    permitted=((world>=17)&(world<=495)).all(1)
    score=np.where(permitted,gain-1.5*dists,-1e8)
    return np.argsort(-score),drive,(p,R,agent,error)


def waypoints(p,R,agent,j):
    dest=DEST[j]
    if visible(agent,dest):
        return [dest@R.T+p]
    choices=[(np.linalg.norm(v-agent)+VIA[j,i],i) for i,v in enumerate(CORNERS)
             if visible(agent,v) and np.all((v@R.T+p>=17)&(v@R.T+p<=495))]
    if not choices:return [dest@R.T+p]
    _,i=min(choices)
    route=[CORNERS[i]]
    while not visible(CORNERS[i],dest):
        viable=[k for k,v in enumerate(CORNERS) if VIA[j,k]<VIA[j,i]-1e-3 and visible(CORNERS[i],v)]
        if not viable:break
        i=min(viable,key=lambda k:np.linalg.norm(CORNERS[i]-CORNERS[k])+VIA[j,k])
        route.append(CORNERS[i])
    return [v@R.T+p for v in [*route,dest]]


def stroke(env,j,drive,budget,stop=True):
    e=env.env
    p,R,agent,_=directions(env)
    steps=0
    hit=False
    def advance(target):
        nonlocal steps,hit
        _,_,done,_=e.step(np.clip(target,17.,495.))
        steps+=1
        hit=bool(done)
    # Reposition in free space before pressing the chosen face.
    for dest in waypoints(p,R,agent,j):
        for _ in range(10):
            if steps>=budget or (stop and hit):return env.obs(),steps,hit
            if np.linalg.norm(np.asarray(e.agent.position)-dest)<3.5:break
            advance(dest)
    # Re-read the body pose after approach; the pusher may already have touched an edge.
    p,R,_,_=directions(env)
    aim=p+R@(DEST[j]+N[j]*(2.+drive*1.10))
    for _ in range(8):
        if steps>=budget or (stop and hit):break
        advance(aim)
        if np.linalg.norm(np.asarray(e.agent.position)-aim)<2.:break
    return env.obs(),steps,hit


def test_case(args):
    split,seed,budget=args
    env=Tee(seed,split)
    try:
        env.stop_on_success=True
        z,_=env.run(ACTIONS[0]);used=env.steps
        hist=[float(z[12])]; hit=z[12]>.95
        while used<budget and not hit:
            ids,drive,_=candidates(env)
            j=int(ids[0])
            z,n,hit=stroke(env,j,drive[j],budget-used)
            used+=n;hist.append(float(z[12]))
        return seed,int(hit),float(z[12]),max(hist),used
    finally:
        env.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--start',type=int,default=99000)
    ap.add_argument('--count',type=int,default=32)
    ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--split',default='controlval')
    ap.add_argument('--output',default='surface-pilot.csv')
    args=ap.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        results=list(pool.map(test_case,[(args.split,i,280) for i in range(args.start,args.start+args.count)]))
    with (OUT/args.output).open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['scene','success','coverage','peak','steps']);w.writerows(results)
    print(np.array(results).mean(0))
