"""Native paired interventions on a feedback policy; no simulator queries by the selector."""
import argparse
import json
import multiprocessing as mp

import numpy as np

from whatif.mechanism.native import Simulator
from whatif.mechanism.proposals import Proposal, ROOT
from whatif.mechanism.collect import tower
from whatif.paired.data import save

DATA = ROOT / 'data/feedback'
OUT = ROOT / 'results/feedback'
DOMAINS = ('pusht', 'causalworld')
SPLITS = {'train': (3000000, 640), 'val': (3100000, 112),
          'test': (3200000, 160), 'control': (3300000, 192)}
HORIZONS = (1, 4, 12, 35)
ACTOR = None


def command(x, role, goal, weights):
    global ACTOR
    if ACTOR is None:
        ACTOR = Proposal()
    a = ACTOR.candidates(x, role, goal, None, 1)[0][0]
    if weights is None:
        return a
    target = np.average(x, axis=0, weights=role[:, 0])
    u, v = target[3:6], target[6:9]
    rotation = np.stack([u, v, np.cross(u, v)])
    node = a[:, 0].argmax()
    d = a[node, 1:].reshape(2, 3)
    # Learned output-layer gains, shared body-frame translation and relative stroke.
    shift = weights[2:5] @ rotation
    residual = (weights[5] * (d[1] - d[0])).copy()
    gain = weights[:2].copy()
    if len(weights) > 6:
        diff = (goal-target).reshape(3, 3)
        gain += weights[6:8] * np.tanh(np.linalg.norm(diff))
        shift += weights[8:11] @ np.tanh(diff @ rotation.T) @ rotation
    d += gain[:, None] * d + shift
    d[1] += residual
    return a


def episode(item):
    domain, seed, split, weights, mode, *extra = item
    prefix = extra[0] if extra else 1
    sim = Simulator(domain, int(seed), split)
    try:
        f0, goal = sim.observe()
        role = sim.role
        fa = command(f0, role, goal, None)
        x, fe, initial = sim.execute(fa, landmarks=True, stop=True)
        peak, last = initial[2], initial[1]
        history = [(f0,fa,x)]
        for tick in range(2,prefix+1):
            if not sim.success:
                f0 = x
                fa = command(x,role,goal,None)
                x,fe,initial = sim.execute(fa,landmarks=True,stop=True)
                peak,last = max(peak,initial[2]),initial[1]
            if tick in (6,12,18):
                history.append((f0,fa,x))
        facts = dict(f0=f0, fa=fa, f1=x, fe=fe, x=x, role=role, goal=goal, seed=seed)
        if prefix > 1:
            facts.update(zip(('hx','ha','hy'),map(np.stack,zip(*history))))
            facts['alive'] = float(not sim.success)
        if mode == 'observe':
            facts['alive'] = float(not sim.success)
            return facts
        a = command(x, role, goal, weights)
        obs, edge = x, np.zeros_like(fe)
        first_y, first_e, first_r = x, edge.copy(), initial
        snapshots, contacts, scores, rewards = [], [], [], []
        native = initial[0]
        remaining = 36-prefix
        for t in range(1, remaining+1):
            if not sim.success:
                act = a if t == 1 else command(obs, role, goal, weights)
                obs, e, r = sim.execute(act, landmarks=True, stop=True)
                native = r[0]
                edge = np.maximum(edge, e)
                peak, last = max(peak, r[2]), r[1]
                if t == 1:
                    first_y, first_e, first_r = obs, e, r
            if t in (*HORIZONS[:3],remaining):
                rewards.append(native)
                snapshots.append(obs)
                contacts.append(edge)
                scores.append([sim.success, peak, last])
        outcome = np.asarray(scores, np.float32)
        if mode == 'score':
            return outcome[-1]
        return dict(**facts, a=a, y=first_y, e=first_e, r=first_r,
                    future=np.stack(snapshots), contact=np.stack(contacts), native=np.asarray(rewards), outcome=outcome)
    finally:
        sim.close()


def design(pool):
    rng = np.random.default_rng(271)
    center = np.zeros(6)
    std = np.array([.35, .45, .17, .17, .12, .35])
    gens, archive = [], []
    for gen in range(3):
        noises = rng.normal(size=(6, 6))
        params = np.clip(center + np.r_[noises, -noises] * std, -1., 1.)
        seeds = range(2900000 + gen * 1000, 2900016 + gen * 1000)
        items = [(dom, s, 'train', w, 'score') for w in params for dom in DOMAINS for s in seeds]
        y = np.stack(pool.map(episode, items, chunksize=2)).reshape(len(params), 2, len(seeds), 3)
        score = (y @ [1., .5, .25]).mean(2)
        order = np.argsort(-score.mean(1))
        elite = params[order[:4]]
        center = elite.mean(0)
        std = np.maximum(elite.std(0), [.09,.09,.045,.045,.035,.09])
        gens.append(dict(weights=params.tolist(), score=score.tolist(), elite=order[:4].tolist()))
        archive.extend([params[order[0]], params[score[:,0].argmax()], params[score[:,1].argmax()]])
        print('generation', gen + 1, score[order[0]], flush=True)
    bank = [np.zeros(6), center]
    for w in reversed(archive):
        if not any(np.array_equal(w, v) for v in bank):
            bank.append(w)
        if len(bank) == 7:
            break
    (OUT / 'bank.json').write_text(json.dumps(dict(bank=np.stack(bank).tolist(), generations=gens), indent=2))


def panel(pool, split, expanded=False):
    tag = 'expanded-' if expanded else ''
    bank = np.array(json.loads((OUT / (tag+'bank.json')).read_text())['bank'])
    base, count = SPLITS[split]
    if expanded and split != 'control':
        base += 600000
        if split == 'train':
            count = 1408
    for domain in DOMAINS:
        if split == 'control':
            items = [(domain, s, split, None, 'observe') for s in range(base, base+count)]
            save(DATA / f'{domain}-observe.npz', pool.map(episode, items))
            continue
        items = [(domain, s, split, w, 'panel') for s in range(base, base+count) for w in bank]
        rows = pool.map(episode, items, chunksize=2)
        panel_rows = []
        for i in range(0, len(rows), len(bank)):
            branch = rows[i:i+len(bank)]
            row = branch[0].copy()
            for k in ('a','y','e','r','future','contact','native','outcome'):
                row[k] = np.stack([b[k] for b in branch])
            if any(not np.allclose(b['x'], row['x'], atol=1e-5) for b in branch):
                raise ValueError('unmatched factual states')
            panel_rows.append(row)
        save(DATA / f'{tag}{domain}-{split}.npz', panel_rows)
        print(domain, split, count, flush=True)


def control(pool):
    plan = json.loads((OUT / 'choices.json').read_text())
    bank = np.array(json.loads((OUT / 'expanded-bank.json').read_text())['bank'])
    result = {}
    for domain, choices in plan.items():
        base, count = SPLITS['control']
        # Identical (seed, policy) is one deterministic closed-loop run shared by the arms.
        keys = sorted({(base+i, v) for arm in choices.values() for i,v in enumerate(arm)})
        items = [(domain, seed, 'control', bank[v], 'score') for seed,v in keys]
        values = dict(zip(keys, pool.map(episode, items, chunksize=2)))
        result[domain] = {arm: [values[(base+i,v)].tolist() for i,v in enumerate(selected)]
                          for arm,selected in choices.items()}
        print(domain, 'actual distinct controlled episodes', len(keys), flush=True)
    (OUT / 'episodes.json').write_text(json.dumps(result))


def cophy(pool):
    from functools import partial
    base = 3400000
    seeds = list(range(base, base+160))
    save(DATA / 'cophy-test.npz', pool.map(tower, seeds))
    out = pool.map(partial(tower,deletion=True), seeds, chunksize=2)
    np.savez_compressed(DATA/'excision-test.npz',seed=[k for k,v in out],impact=[v for k,v in out])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('job', choices=['bank','cophy','collect','expand','panels','observe','execute'])
    p.add_argument('--workers', type=int, default=48)
    args = p.parse_args()
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    with mp.Pool(args.workers, maxtasksperchild=80) as pool:
        if args.job == 'bank':
            design(pool)
        elif args.job == 'cophy':
            cophy(pool)
        elif args.job == 'observe':
            panel(pool,'control',True)
        elif args.job == 'expand':
            from .search import expand
            expand(pool)
        elif args.job in ('collect','panels'):
            for split in SPLITS:
                panel(pool, split, args.job == 'panels')
        else:
            control(pool)


if __name__ == '__main__':
    main()
