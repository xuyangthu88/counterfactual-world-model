"""Paired interventions with disjoint layout seeds. Run using the simulator environment."""
import argparse
import ast
import multiprocessing as mp
import os
from pathlib import Path
import random

import numpy as np
from .native import A, D, N, Simulator, menu, pose, roles

ROOT = Path(__file__).resolve().parents[2]
SPLITS = {'train': (123000,384), 'val': (223000,72), 'test': (323000,128)}
BRANCHES = 4
NATIVE = None


def cophy_api():
    global NATIVE
    if NATIVE is None:
        import pybullet as pb
        import pybullet_data
        upstream = ROOT/'work/stage7/upstream/cophy'
        source = upstream/'data_generation/create_blocktwer_examples.py'
        os.chdir(upstream)
        # Import just these upstream definitions: the standalone script parses argv on import.
        names = {'generate_random_tower','move_one_block','set_up_env','set_up_tower','get_current_states'}
        nodes = [n for n in ast.parse(source.read_text()).body
                 if isinstance(n,ast.FunctionDef) and n.name in names]
        NATIVE = dict(pb=pb, np=np, pybullet_data=pybullet_data, choice=random.choice,
                      OFF_MIN_X=-1., OFF_MAX_X=1.)
        exec(compile(ast.Module(nodes,type_ignores=[]),str(source),'exec'), NATIVE)
        pb.connect(pb.DIRECT)
    return NATIVE


def tower(seed, deletion=False, trace=False, *, height=None, trajectory_pairs=False):
    """Replay matched block interventions; trace records five physical windows and exact deletion effects."""
    api = cophy_api()
    pb = api['pb']
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)
    height = 3+seed%2 if height is None else height
    trace = trace or trajectory_pairs
    poses, angles = api['generate_random_tower'](height)
    masses = rng.choice([1.,10.],height)
    frictions = rng.choice([.5,1.],height)
    gravity = [0.,0.,-10.]
    gravity[rng.integers(2)] = rng.choice([-.5,0.,.5])
    def simulate(p, remove=None):
        api['set_up_env'](.2,5,gravity)
        colors = (['red','green','blue','yellow'] * ((height+3)//4))[:height]
        ids = api['set_up_tower'](p,angles,colors,masses,frictions)
        if remove is not None:
            i,j = remove
            pb.setCollisionFilterPair(ids[i],ids[j],-1,-1,0)
        def obs():
            x = np.zeros((N,D),np.float32)
            for i,v in enumerate(api['get_current_states'](ids)):
                x[i] = pose(v[:3],v[3:7])
            return x
        start = obs()
        edge = np.zeros((N,N),np.float32)
        lookup = {body:i for i,body in enumerate(ids)}
        snapshots,windows=[],[]
        for tick in range(30):
            pb.stepSimulation()
            for contact in pb.getContactPoints():
                i,j = lookup.get(contact[1]),lookup.get(contact[2])
                if i is not None and j is not None and contact[9] > .001:
                    edge[i,j] = edge[j,i] = 1
            if trace and (tick+1)%6==0:
                snapshots.append(obs())
                windows.append(edge.copy())
                edge.fill(0)
        if trace:
            return start,np.stack(snapshots),np.stack(windows)
        return start, obs(), edge
    x,f1,fe = simulate(poses)
    fa = np.zeros((N,A),np.float32)
    actions, ys, edges, effects, deltas = [], [], [], [], []
    pairs = list(zip(*np.tril_indices(N, -1)))
    for _ in range(BRANCHES):
        j,axis = rng.integers(height),rng.integers(2)
        delta = rng.choice([-1,1])*rng.choice(np.arange(.1,2.01,.05))
        act = fa.copy()
        act[j,0] = 1
        act[j,1+axis] = act[j,4+axis] = delta
        alt = api['move_one_block'](poses,j,'xy'[axis],delta)
        _,y,e = simulate(alt)
        if deletion or trace:
            impact = np.zeros((5,N,N) if trace else (N,N),np.float32)
            if trajectory_pairs:
                # [disabled unordered pair, time, receiver, coordinate]
                signed = np.zeros((len(pairs),5,N,3),np.float32)
            for p,(src,dst) in enumerate(pairs):
                if src >= height:
                    break
                _,missing,_ = simulate(alt,(src,dst))
                difference = y[...,:3]-missing[...,:3]
                changes = np.linalg.norm(difference,axis=-1)
                impact[...,src,dst],impact[...,dst,src] = changes[...,dst],changes[...,src]
                if trajectory_pairs:
                    signed[p] = difference
            if trajectory_pairs:
                deltas.append(signed)
            effects.append(impact)
        actions.append(act)
        ys.append(y)
        edges.append(e)
    if trace:
        row = dict(seed=seed,f0=x,fa=fa,f1=f1[-1],fe=fe.max(0),x=x,role=roles(list(range(height)),[],[]),
                   goal=np.zeros(D),a=actions,traj=ys,contact=edges,impact=effects)
        if trajectory_pairs:
            row.update(fact=np.concatenate([x[None],f1]), off=deltas)
        return row
    if deletion:
        return seed,np.stack(effects)
    return dict(f0=x,fa=fa,f1=f1,fe=fe,x=x,role=roles(list(range(height)),[],[]),
                goal=np.zeros(D),a=actions,y=ys,e=edges,r=np.zeros((BRANCHES,3)),seed=seed)


def pushing(domain,seed,split,fine=False,proposal=False):
    rng = np.random.default_rng(seed)
    if proposal:
        from .proposals import Proposal
        proposer = Proposal()
    def get_actions(state,role,goal,count):
        if proposal:
            return proposer.candidates(state,role,goal,rng,count)[0]
        return menu(state,role,rng,count,fine)
    commands = []
    ys, edges, rewards = [], [], []
    for i in range(BRANCHES):
        sim = Simulator(domain,seed,split)
        try:
            if i == 0:
                # Sample varied factual histories using exactly the controller's proposal law.
                for _ in range(1+seed%(12 if proposal else 4)):
                    f0, goal = sim.observe()
                    if proposal:
                        candidates=get_actions(f0,sim.role,goal,4)
                        fa=candidates[rng.choice(4,p=[.7,.1,.1,.1])]
                    else:
                        fa=get_actions(f0,sim.role,goal,1)[0]
                    commands.append(fa)
                    f1, fe, _ = sim.execute(fa,fine=fine,landmarks=proposal)
                x = f1
                role = sim.role
                actions = get_actions(x,role,goal,24 if proposal else BRANCHES)
                if proposal:
                    chosen=np.r_[0,rng.choice(np.arange(1,24),3,replace=False)]
                    actions=actions[chosen]
            else:
                for command in commands:
                    replay,_,_ = sim.execute(command,fine=fine,landmarks=proposal)
                if not np.allclose(replay,x,atol=1.e-5):
                    raise RuntimeError(f'{domain} factual replay diverged at {seed}')
            y,e,r = sim.execute(actions[i],fine=fine,landmarks=proposal)
            ys.append(y)
            edges.append(e)
            rewards.append(r)
        finally:
            sim.close()
    return dict(f0=f0,fa=fa,f1=f1,fe=fe,x=x,role=role,goal=goal,a=actions,
                y=ys,e=edges,r=rewards,seed=seed)


def sample(item):
    domain,seed,split,fine,proposal = item
    return tower(seed) if domain == 'cophy' else pushing(domain,seed,split,fine,proposal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('domain',choices=['cophy','causalworld','pusht'])
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--count',type=int)
    parser.add_argument('--fine',action='store_true')
    parser.add_argument('--proposal',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output=ROOT/'data/mechanism'
        if args.fine or args.proposal:
            args.output=args.output/('demonstrations' if args.proposal else 'fine')
    args.output.mkdir(parents=True,exist_ok=True)
    splits = {'train':(523000,1536), 'val':(623000,144),'test':(723000,160)} if args.fine else SPLITS
    if args.proposal:
        splits={'train':(923000,1536),'val':(1023000,144),'test':(1123000,160)}
    for split,(base,n) in splits.items():
        n = args.count or n
        with mp.Pool(args.workers,maxtasksperchild=32) as pool:
            rows = list(pool.imap(sample,[(args.domain,base+i,split,args.fine,args.proposal) for i in range(n)]))
        dest = args.output/f'{args.domain}-{split}.npz'
        np.savez_compressed(dest,**{k:np.asarray([row[k] for row in rows],
                           dtype=np.int64 if k=='seed' else np.float32) for k in rows[0]})
        print(dest,n,flush=True)


if __name__ == '__main__':
    main()
