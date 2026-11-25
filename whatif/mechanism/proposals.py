"""Object-frame mixture actor. NumPy inference is also used by simulator workers."""
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MODES=6
WEIGHTS=ROOT/'results/mechanism/demonstrations/proposal.npz'


def local_view(state,role,goal):
    target=np.average(state,axis=0,weights=role[:,0])
    ids=np.flatnonzero(role[:,1])
    ex,ey=target[3:6],target[6:9]
    rotation=np.stack([ex,ey,np.cross(ex,ey)],axis=1)
    own=(state[ids,:3]-target[:3])@rotation
    dest=(goal[:3]-target[:3])@rotation
    orientation=goal[3:].reshape(2,3)@rotation
    # Each actor scores against the same scene, expressed in the movable body's frame.
    cond=np.r_[dest,orientation.ravel(),target[:3],ex,ey,len(ids)/3.]
    feat=np.concatenate([own,np.broadcast_to(cond,(len(ids),len(cond)))],axis=-1)
    return ids,rotation,feat.astype(np.float32)


class Proposal:
    def __init__(self):
        self.weights=dict(np.load(WEIGHTS))

    def candidates(self,state,role,goal,rng,k):
        ids,rotation,z=local_view(state,role,goal)
        w=self.weights
        z=(z-w['center'])/w['spread']
        for j in range(4):
            z=z@w['w'+str(j)].T+w['b'+str(j)]
            if j<3:
                z=z/(1+np.exp(-np.clip(z,-40,40)))
        z=z.reshape(len(ids),MODES,7)
        logits=z[:,:,0].ravel()
        logp=logits-np.logaddexp.reduce(logits)
        means=z[:,:,1:].reshape(-1,2,3)
        order=np.argsort(-logp)
        choose=[order[0]]
        if k>1:
            choose+=list(order[1:min(4,k,len(order))])
            choose+=list(rng.choice(len(logp),k-len(choose),p=np.exp(logp)/np.exp(logp).sum()))
        commands=np.zeros((k,len(role),7),np.float32)
        prior=np.empty(k,np.float32)
        for i,m in enumerate(choose):
            shift=means[m].copy()
            if i>=min(4,k,len(order)):
                shift+=rng.normal(0,.075,(2,3))
            node=ids[m//MODES]
            commands[i,node]=np.r_[1,(shift@rotation.T).ravel()]
            prior[i]=logp[m]-logp.max()
        return commands,prior
