"""Shared, observable contact description; no simulator import is needed for planning."""
from pathlib import Path
import numpy as np
from whatif.crossbench.tasks import ACTIONS

DATA=Path('data/shared')
OUT=Path('results/shared')
START={'train':221000,'val':231000,'test':241000,'controlval':251000,'transfer_val':255000,'control':261000}
# Body-local candidate sites, boundary of the two rectangles in PushT.
corners=np.array([[-60.,0],[60.,0],[60.,30],[15.,30],[15.,120],[-15.,120],[-15.,30],[-60.,30]])
sites=[]; natives=[]
for i,(a,b) in enumerate(zip(corners,np.roll(corners,-1,axis=0))):
    v=(b-a)/np.linalg.norm(b-a)
    outward=v[[1,0]]*np.array([1,-1])
    for t in (.16,.50,.84):
        pt=a+(b-a)*t
        dest=pt+outward*17.
        # Outside both inflated rectangles, also at reentrant corners.
        def rect_distance(x,lo,hi):
            return np.linalg.norm(np.maximum(np.maximum(lo-x,x-hi),0.))
        if min(rect_distance(dest,[-60,0],[60,30]),rect_distance(dest,[-15,30],[15,120]))<15.5:continue
        sites.append(np.r_[(pt-[0.,45.])/100,-outward])
SITES=np.array(sites)
L=.47
FACTORS=np.array([.35,.80,1.35])
N_TEE=12
N_ACTIONS=12


def rotate(x,theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.stack((c*x[...,0]-s*x[...,1],s*x[...,0]+c*x[...,1]),axis=-1)


def wrap(a):
    return np.arctan2(np.sin(a),np.cos(a))


def menu(state):
    """Returns [source-relative contact xy, normal xy, stroke, torque arm, distance], in body units."""
    s=np.asarray(state)
    if s[13]<.5:
        goal=s[9:11]-s[:2]
        phi=np.arctan2(goal[1],goal[0])+ACTIONS[:,0]-s[2]
        n=np.stack((np.cos(phi),np.sin(phi)),axis=1)
        side=np.stack((-n[:,1],n[:,0]),axis=1)
        r=-n+ACTIONS[:,1,None]*side
        tips=rotate(s[3:9].reshape(3,2)-s[:2],-s[2])
        dist=np.linalg.norm(r[:,None]-tips,axis=-1).min(1)
        arm=r[:,0]*n[:,1]-r[:,1]*n[:,0]
        return np.column_stack((r,n,ACTIONS[:,2],arm,dist)).astype('float32')
    r,n=SITES[:,:2],SITES[:,2:]
    arm=r[:,0]*n[:,1]-r[:,1]*n[:,0]
    h=np.column_stack((n,arm/L))
    delta=rotate(s[9:11]-s[:2],-s[2])+rotate(np.array([0.,.45]),s[11]-s[2])-[0.,.45]
    e=np.r_[delta,L*wrap(s[11]-s[2])]
    agent=rotate(s[3:5]-s[:2],-s[2])-[0.,.45]
    dest=r-.17*n
    travel=np.linalg.norm(dest-agent,axis=1)
    drive=np.clip((h@e)/(h*h).sum(1),.018,.36)
    gain=2*(h@e)*drive-(h*h).sum(1)*drive**2
    world=s[:2]+rotate(dest+[0.,.45],s[2])
    score=gain-.013*travel-10*(np.abs(world)>2.39).any(1)
    # Four different torque/contact choices, each with a precision and two longer strokes.
    chosen=[]
    for _ in range(4):
        j=score.argmax();chosen.append(j)
        score[np.logical_and(n@n[j]>.99,np.abs(arm-arm[j])<.22)]=-1e4
    j=np.repeat(chosen,3)
    return np.column_stack((r[j],n[j],drive[j]*np.tile(FACTORS,4),arm[j],travel[j])).astype('float32')
