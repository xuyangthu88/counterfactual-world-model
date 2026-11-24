"""Branched pushes with one to three movable cubes on CausalWorld's native TriFinger stage."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic

import numpy as np
import pybullet as pb
from causal_world.configs.world_constants import WorldConstants
from .collect import ACTIONS, DISTURBANCE, FRICTION, MASS, PROBE
from .simulator import PushingScene, Push

# Record object states at push onset and at five subsequent phase boundaries.
TICKS = (0, 24, 27, 30, 33, 42)
SPLITS = {"train": (30000, 600), "val": (31000, 96), "two": (32000, 144),
          "three": (33000, 192), "physical": (34000, 144), "frame0": (35000, 128),
          "frameplus": (35000, 128), "frameminus": (35000, 128)}
FRAME = {"frameplus": 2*np.pi/3, "frameminus": -2*np.pi/3}


# Four real first pushes on the same held-out three-cube layouts, with a fresh
# second-push disturbance shared across all counterfactual second actions.
FIRST_ACTION = {"probe32": 4, "probe41": PROBE, "probe54": 1, "probe69": 6}
SPLITS.update({split: (36000, 192) for split in FIRST_ACTION})


class Chain(PushingScene):
    def __init__(self, rng, n, physical, seed, frame=0.):
        p = ([rng.choice([.045,.105]),rng.choice([.33,.69])] if physical else
             [rng.choice(MASS),rng.choice(FRICTION)])
        xy = rng.uniform([.063,-.034],[.079,-.015])
        yaw = rng.uniform(-.09,.09)
        positions = [xy]
        for _ in range(n-1):
            positions.append(positions[-1] + [-rng.uniform(.071,.085),rng.uniform(-.017,.017)])
        self.target = positions[-1] + rng.uniform([-.046,-.010],[-.023,.010])
        rotation = np.array([[np.cos(frame),-np.sin(frame)],[np.sin(frame),np.cos(frame)]])
        if frame:
            positions = [rotation @ q for q in positions]
            self.target = rotation @ self.target
        super().__init__(positions[0], yaw+frame, *p, self.target, seed)
        self.active = 0 if frame>0 else (2 if frame<0 else 1)
        self.push_angle = frame
        self.particle = p
        self.bodies = [self.body]
        for i,pos in enumerate(positions[1:],1):
            key = f"passive_cube_{i}"
            self.stage.add_rigid_general_object(
                name=key, shape="cube", initial_position=np.r_[pos,.0325],
                initial_orientation=np.array(pb.getQuaternionFromEuler([0,0,rng.uniform(-.09,.09)+frame])),
                size=np.full(3,.065),mass=float(p[0]))
            self.bodies.append(self.stage.get_rigid_objects()[key].get_block_ids()[0])
        for _ in range(5):
            self.env.step(self.neutral)

    def nodes(self):
        result = []
        for body in self.bodies:
            pos, rot = pb.getBasePositionAndOrientation(body, physicsClientId=self.client)
            v, omega = pb.getBaseVelocity(body, physicsClientId=self.client)
            result.append([pos[0],pos[1],pb.getEulerFromQuaternion(rot)[2],v[0],v[1],omega[2]])
        return np.array(result)

    def branch(self, action):
        states = [self.nodes()]
        n = len(self.bodies)
        # row 0 is robot; rows 1..n are cube senders. Columns are cube receivers.
        contacts = np.zeros((len(TICKS)-1,n+1,n),bool)
        lookup = {body:i for i,body in enumerate(self.bodies)}
        tick = 0
        phase = 0
        def record():
            nonlocal tick,phase
            tick += 1
            for c in pb.getContactPoints(physicsClientId=self.client):
                if c[9] <= .001:
                    continue
                a,b = c[1:3]
                if a in lookup and b in lookup:
                    i,j = lookup[a],lookup[b]
                    if i != j:
                        contacts[phase,i+1,j] = contacts[phase,j+1,i] = True
                elif a == WorldConstants.ROBOT_ID and b in lookup:
                    contacts[phase,0,lookup[b]] = True
                elif b == WorldConstants.ROBOT_ID and a in lookup:
                    contacts[phase,0,lookup[a]] = True
            if tick == TICKS[phase+1]:
                states.append(self.nodes())
                phase += 1
        self.push(action,after_step=record)
        return states, contacts


def episode(job):
    split,seed = job
    rng = np.random.default_rng(seed)
    n = 3 if split=="three" or split.startswith("frame") or split in FIRST_ACTION else (1 if split=="train" and seed%5==0 else 2)
    probe = FIRST_ACTION.get(split, PROBE)
    return collect_layout(seed, rng, n, split=="physical", FRAME.get(split,0.), probe)


def collect_layout(seed, rng, n, physical, frame, probe):
    scene = Chain(rng, n, physical, seed, frame)
    noise = rng.choice(DISTURBANCE,2)
    try:
        # The same action command refers to the source cube at the start of each push.
        states=np.zeros((2,len(ACTIONS),len(TICKS),3,6),np.float32)
        edges=np.zeros((2,len(ACTIONS),len(TICKS)-1,4,3),bool)
        fingertips=[]
        for t in range(2):
            fingertips.append(scene.robot.get_latest_full_state()["end_effector_positions"].reshape(3,3)[scene.active].copy())
            with scene.snapshot() as restore:
                for i,(offset,stroke) in enumerate(ACTIONS):
                    restore()
                    st,ed=scene.branch(Push(offset,stroke+noise[t]))
                    states[t,i,:,:n]=np.asarray(st)
                    edges[t,i,:,:n+1,:n]=ed
            if t==0:
                offset,stroke=ACTIONS[probe]
                scene.push(Push(offset,stroke+noise[t]))
                if not np.allclose(scene.nodes(),states[0,probe,-1,:n],atol=2e-5):
                    raise RuntimeError(f"factual branch did not reproduce in scene {seed}")
        return seed,n,states,edges,scene.particle,noise,scene.target,fingertips
    finally:
        scene.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--splits",nargs="+",choices=SPLITS,default=list(SPLITS))
    ap.add_argument("--workers",type=int,default=12)
    ap.add_argument("--out",type=Path,default=Path("data/chains"))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for split in args.splits:
        start,count=SPLITS[split]
        tic=monotonic()
        with ProcessPoolExecutor(args.workers) as pool:
            rows=list(pool.map(episode,[(split,i) for i in range(start,start+count)]))
        names=("seed","count","states","edges","physics","noise","goal","tips")
        np.savez_compressed(args.out/(split+".npz"),actions=ACTIONS,ticks=TICKS,probe=FIRST_ACTION.get(split,PROBE),frame=FRAME.get(split,0.),
                            **dict(zip(names,map(np.asarray,zip(*rows)))))
        print(split,count,round(monotonic()-tic,1),flush=True)


if __name__=="__main__":
    main()
