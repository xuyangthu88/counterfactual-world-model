"""Fork identical native states and record the hand-off between action phases."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from whatif.shared.actions import menu
from whatif.factual.collect import wall_case

DATA = Path("data/precision")
OUT = Path("results/precision")
START = {"train": 611000, "val": 621000, "test": 631000, "controlval": 641000, "control": 651000}
SIZES = {
    "pusht": (1000, 192, 256),
    "causalworld": (360, 128, 192),
    "near": (420, 160, 160),
    "wall": (0, 128, 160),
    "p9": (0, 0, 128),
}
OFFSET = {"pusht": 0, "causalworld": 0, "near": 2000, "wall": 4000, "p9": 6000}


def collect(job):
    from whatif.shared.sim import Scene
    from whatif.shared.returns import physics

    task, split, seed = job
    if task == "wall":
        d = wall_case(split, seed, True)
        return {k: v for k, v in d.items() if k not in ("nullpose", "source")}
    cw = task in ("causalworld", "p9")
    sim = "causalworld" if cw else "pusht"
    mode = (
        "p9"
        if task == "p9"
        else (
            "train" if split == "train" else "transfer_val" if not cw and split == "val" else "test"
        )
    )
    rng = np.random.default_rng(seed)

    def reset():
        env = Scene(sim, seed, mode, near=task == "near")
        if task == "near":
            # Scene near resets inherit the train damping list; override for held-out cohorts.
            damping = (
                (0.0, 0.08, 0.55)[seed % 3]
                if split == "train"
                else (0.0 if seed % 2 == 0 else (0.18 if split == "val" else 0.25))
            )
            env.env.env.space.damping = damping
            env.env.mass = damping
            env.env.protocol = int(damping > 0)
            env.env.physics = int(np.digitize(damping, [0.04, 0.3]))
        return env

    env = reset()
    prefix = []
    for _ in range(int(rng.integers(0, 3 if cw or task == "near" else 6))):
        x = env.state()
        ix = int(np.argsort(-physics(x, menu(x)))[rng.choice([0, 0, 1, 3])])
        prefix.append(ix)
        env.push(ix)
    prev = env.state()
    b = int(rng.choice([0, 1, 2, 4, 7]))
    fact = menu(prev)[b]
    ub, ua = rng.choice([-0.04, 0.0, 0.04], 2)
    s = env.push(b, noise=ub)[0]
    group = env.env.protocol
    param = env.env.physics
    env.close()
    actions = menu(s)
    ends = []
    ph = []
    edges = []
    peaks = []
    phase_edges = []
    ticks = []
    peak_history = []
    ref = int(rng.integers(12))
    future_idx = rng.integers(0, 12, 2)
    for j in range(12):
        env = reset()
        for ix in prefix:
            env.push(ix)
        check = env.push(b, noise=ub)[0]
        if j == 0:
            np.testing.assert_allclose(check, s, atol=6e-4, rtol=1e-5)
        flux = []
        z, e, _, p = env.push(j, noise=ua, phases=flux)
        assert len(flux) == 3
        ends.append(z)
        edges.append(e)
        peaks.append(p)
        ph.append([v[0] for v in flux])
        phase_edges.append([v[1] for v in flux])
        ticks.append([v[2] for v in flux])
        peak_history.append([v[3] for v in flux])
        if j == ref:
            future = []
            for ix in future_idx:
                future.append(env.push(int(ix))[0])
        env.close()
    return dict(
        seed=seed,
        state=s,
        previous=prev,
        fact=fact,
        cmd=actions,
        end=ends,
        ref=ref,
        physics=param,
        protocol=group,
        phase=ph,
        phase_edges=phase_edges,
        phase_ticks=ticks,
        phase_peak=peak_history,
        edge=edges,
        peak=peaks,
        future=future,
        future_idx=future_idx,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=SIZES)
    ap.add_argument("split", choices=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    n = SIZES[a.task][["train", "val", "test"].index(a.split)]
    assert n
    start = START[a.split] + OFFSET[a.task]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(collect, [(a.task, a.split, i) for i in range(start, start + n)]))
    np.savez_compressed(
        DATA / f"{a.task}-{a.split}.npz", **{k: [r[k] for r in rows] for k in rows[0]}
    )
    print(a.task, a.split, n, flush=True)
