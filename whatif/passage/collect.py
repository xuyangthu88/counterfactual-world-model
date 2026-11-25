"""Independent native branches and exact within-action coverage, including precision finishes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import numpy as np
from whatif.precision.collect import collect

DATA = Path("data/passage")
OUT = Path("results/passage")
START = {"val": 721000, "test": 731000}
OFFSET = {"pusht": 0, "causalworld": 0, "near": 2000, "wall": 4000, "p9": 6000, "finish": 8000}
COUNTS = {
    "pusht": (192, 320),
    "causalworld": (160, 240),
    "near": (160, 192),
    "wall": (160, 192),
    "p9": (0, 128),
    "finish": (256, 384),
}


def finishing_scene(task, split, seed):
    from whatif.shared.sim import Scene

    # A separate, goal-near diagnostic. The primary control reset always uses the official law.
    env = Scene("pusht", seed, "transfer_val" if split == "val" else "test")
    e = env.env.env
    rng = np.random.default_rng(seed)
    theta = e.goal_pose[2] + rng.uniform(-0.16, 0.16)
    direction = rng.uniform(-np.pi, np.pi)
    radius = rng.uniform(2.0, 11.0)
    pos = e.goal_pose[:2] + radius * np.array([np.cos(direction), np.sin(direction)])
    phi = rng.uniform(-np.pi, np.pi)
    side = np.array([np.cos(phi), np.sin(phi)])
    agent = pos + np.array([-np.sin(theta), np.cos(theta)]) * 45 + side * rng.uniform(100, 150)
    e.block.angular_velocity = 0
    e.block.velocity = (0, 0)
    e.agent.velocity = (0, 0)
    e.block.angle = theta
    e._set_state(np.r_[agent, pos, theta])
    return env


def main(task, split, workers):
    label = "near" if task == "finish" else task
    n = COUNTS[task][split == "test"]
    if not n:
        return
    fn = partial(
        collect,
        record_coverage=task != "wall",
        reset_scene=finishing_scene if task == "finish" else None,
    )
    start = START[split] + OFFSET[task]
    with ProcessPoolExecutor(workers) as jobs:
        rows = list(jobs.map(fn, [(label, split, seed) for seed in range(start, start + n)]))
    path = DATA / f"{task}-{split}.npz"
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **{k: [r[k] for r in rows] for k in rows[0]})
    path.with_suffix(".tmp").replace(path)
    print(task, split, n, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=COUNTS)
    p.add_argument("split", choices=START)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    main(a.task, a.split, a.workers)
