"""Collect same-state open-loop potential outcomes and optional edge interventions."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from whatif.mechanism.collect import tower
from whatif.mechanism.native import Simulator, menu

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/effects'
OUT = ROOT / 'results/effects'
PROTOCOL = json.loads(Path(__file__).with_name('protocol.json').read_text())
DOMAINS = ('cophy', 'causalworld', 'pusht')
PUSH_WINDOWS = (1, 2, 4, 6, 8)


def pushing(domain, seed, env_split):
    rng = np.random.default_rng(seed)
    prefix = []
    branches = []

    def record(sim, action):
        ticks = []
        sim.execute(action, landmarks=True, on_tick=lambda y, e, r: ticks.append((y, e, r)))
        state, edge, reward = zip(*ticks)
        starts = (0, *PUSH_WINDOWS[:-1])
        return (
            np.stack([state[end-1] for end in PUSH_WINDOWS]),
            np.stack([np.maximum.reduce(edge[start:end]) for start, end in zip(starts, PUSH_WINDOWS)]),
            np.array([np.mean(reward[start:end]) for start, end in zip(starts, PUSH_WINDOWS)], np.float32),
        )

    for branch in range(PROTOCOL['branches']):
        sim = Simulator(domain, seed, env_split)
        try:
            if branch == 0:
                # Vary the initial state using the same open-loop proposal law as the queries.
                for _ in range(1 + seed % 5):
                    f0, goal = sim.observe()
                    prefix.append(menu(f0, sim.role, rng, 1, fine=True)[0])
                    observed, factual_edge, _ = record(sim, prefix[-1])
                x, goal = sim.observe()
                role = sim.role.copy()
                actions = menu(x, role, rng, PROTOCOL['branches'], fine=True)
            else:
                for action in prefix:
                    sim.execute(action, landmarks=True)
                if not np.allclose(sim.observe()[0], x, atol=1.e-5):
                    raise RuntimeError(f'{domain} {seed}: factual replay changed')
            branches.append(record(sim, actions[branch]))
        finally:
            sim.close()
    traj, contact, rewards = map(np.stack, zip(*branches))
    return dict(
        seed=seed, x=x, a=actions, role=role, goal=goal, fa=prefix[-1],
        fact=np.concatenate([f0[None], observed]), fe=factual_edge.max(0),
        traj=traj, contact=contact, rewards=rewards,
    )


def sample(item):
    domain, seed, split = item
    if domain == 'cophy':
        return tower(seed, height=5 if split == 'transfer' else None, trajectory_pairs=True)
    return pushing(domain, seed, 'test' if split == 'transfer' else 'val')


def items(domain, split):
    spec = PROTOCOL['splits'][split]
    n = spec[domain]
    base = spec['base']
    if split == 'transfer' and domain == 'causalworld':
        seeds = (base + 4*(i//2) + 2 + (i % 2) for i in range(n))
    elif split == 'transfer' and domain == 'pusht':
        seeds = (base + 2*i + 1 for i in range(n))
    else:
        seeds = range(base, base+n)
    return [(domain, seed, split) for seed in seeds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('domain', choices=DOMAINS)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--split', choices=PROTOCOL['splits'])
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    with mp.Pool(args.workers, maxtasksperchild=32) as pool:
        for split in ([args.split] if args.split else PROTOCOL['splits']):
            rows = pool.map(sample, items(args.domain, split), chunksize=2)
            fields = ['seed','x','a','role','goal','fa','fact','fe','traj','contact',
                      *(['off'] if args.domain == 'cophy' else ['rewards'])]
            path = DATA / f'{args.domain}-{split}.npz'
            np.savez_compressed(path, **{key: np.asarray([row[key] for row in rows]) for key in fields})
            print(path.relative_to(ROOT), len(rows), flush=True)


if __name__ == '__main__':
    main()
