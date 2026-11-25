"""Counterfactual completion returns under a shared native feedback continuation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from whatif.shared.actions import menu
from whatif.shared.returns import physics
from .collect import DATA

SIZES = {"pusht": (1800, 256), "causalworld": (640, 160)}
START = {"train": 781000, "val": 791000}


def sample(job):
    from whatif.shared.sim import Scene

    task, split, seed = job
    cw = task == "causalworld"
    tau = 0.9 if cw else 0.95
    limit = 420 if cw else 280
    rng = np.random.default_rng(seed)
    # One in four PushT training resets is from the established fine-alignment curriculum.
    stagger = not cw and seed >= 801000
    near = not cw and ((seed // 2 if stagger else seed) % 4 == 1)
    depth = (seed % 3 + 1 + (seed // 3) % 2) % 3 if stagger and split == "train" else seed % 3
    mode = "train" if split == "train" else "test" if cw else "transfer_val"

    def reset(layout):
        env = Scene(task, layout, mode, near=near)
        if near and split == "val":
            value = 0.18 if layout % 2 else 0.0
            env.env.env.space.damping = value
            env.env.mass = value
            env.env.protocol = int(value > 0)
        return env

    for attempt in range(96):
        layout = seed + 1000000 * attempt
        env = reset(layout)
        pref = []
        used = 0
        state = env.state()
        target = [1, int(rng.integers(2, 5)), int(rng.integers(5, 11 if not cw else 9))][depth]
        for h in range(target):
            cmds = menu(state)
            order = np.argsort(-physics(state, cmds))
            idx = int(1 if h == 0 and attempt == 0 else order[rng.choice([0, 0, 0, 1, 2, 4, 7])])
            z, _, ticks, _ = env.push(idx, budget=min(42, limit - used), stop=True)
            if z[12] > tau or limit - used - ticks < (42 if cw else 3):
                break
            previous = state
            fact = cmds[idx]
            state = z
            pref.append(idx)
            used += ticks
        env.close()
        if pref:
            break
    else:
        raise RuntimeError(f"no eligible factual anchor for {task} {seed}")
    remaining = limit - used
    actions = menu(state)
    ends = []
    peaks = []
    durations = []
    hits = []
    coverage = []
    end_step = []
    edge = []
    action_peak = []
    for j in range(12):
        env = reset(layout)
        for idx in pref:
            anchor = env.push(idx, stop=True)[0]
        np.testing.assert_allclose(anchor, state, atol=0.0006, rtol=1e-5)
        z, e, ticks, pk = env.push(j, budget=min(42, remaining), stop=True)
        first = z.copy()
        peak = max(float(z[12]), pk)
        steps = ticks
        ends.append(first)
        edge.append(e)
        action_peak.append(peak)
        end_step.append(ticks)
        while z[12] <= tau and steps < remaining:
            cmd = menu(z)
            idx = int(physics(z, cmd).argmax())
            z, _, ticks, pk = env.push(idx, budget=min(42, remaining - steps), stop=True)
            peak = max(peak, float(z[12]), pk)
            steps += ticks
        if steps > remaining:
            raise RuntimeError("native budget exceeded")
        hits.append(int(z[12] > tau))
        coverage.append(float(z[12]))
        peaks.append(peak)
        durations.append(steps)
        protocol = env.env.protocol
        env.close()
    return dict(
        seed=seed,
        layout=layout,
        near=near,
        protocol=protocol,
        previous=previous,
        fact=fact,
        state=state,
        cmd=actions,
        ref=int(rng.integers(12)),
        end=ends,
        edge=edge,
        action_peak=action_peak,
        action_steps=end_step,
        remaining=remaining,
        success=hits,
        coverage=coverage,
        steps=durations,
        peak=peaks,
    )


def main(task, split, workers):
    n = SIZES[task][split == "val"]
    start = START[split]
    ids = range(start, start + n)
    if task == "pusht":
        ids = (
            [*range(781000, 781600), *range(801000, 802200)]
            if split == "train"
            else range(806000, 806256)
        )
    with ProcessPoolExecutor(workers) as pool:
        rows = []
        for row in pool.map(sample, [(task, split, i) for i in ids]):
            rows.append(row)
            if len(rows) % 100 == 0:
                print(task, split, len(rows), "/", n, flush=True)
    path = DATA / f"return-{task}-{split}.npz"
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **{k: [r[k] for r in rows] for k in rows[0]})
    path.with_suffix(".tmp").replace(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=SIZES)
    p.add_argument("split", choices=START)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    main(a.task, a.split, a.workers)
