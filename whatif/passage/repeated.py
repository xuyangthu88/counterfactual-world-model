"""Repeat each paired intervention under all three native stroke-length disturbances."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from whatif.precision.collect import collect
from .collect import DATA, finishing_scene

SIZES = {"pusht": (48, 96), "causalworld": (48, 96), "finish": (64, 128)}
OFF = {"pusht": 0, "causalworld": 1000, "finish": 2000}


def run(job):
    task, split, seed = job
    rows = [
        collect(
            ("near" if task == "finish" else task, split, seed),
            record_coverage=True,
            reset_scene=finishing_scene if task == "finish" else None,
            future_noise=u,
            record_followups=False,
        )
        for u in (-0.04, 0.0, 0.04)
    ]
    base = rows[0]
    out = {
        k: base[k]
        for k in ("previous", "fact", "state", "cmd", "ref", "seed", "physics", "protocol")
    }
    for row in rows[1:]:
        for key in ("previous", "state", "cmd", "fact", "ref"):
            np.testing.assert_allclose(row[key], base[key], atol=0.0006, rtol=1e-5)
    for k in ("end", "edge", "peak", "first_passage"):
        out[k] = np.stack([r[k] for r in rows], axis=1)
    return out


def main(task, split, workers):
    n = SIZES[task][split == "test"]
    start = (762000 if split == "val" else 772000) + OFF[task]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(run, [(task, split, i) for i in range(start, start + n)]))
    path = DATA / f"repeat-{task}-{split}.npz"
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **{k: [r[k] for r in rows] for k in rows[0]})
    path.with_suffix(".tmp").replace(path)
    print(task, split, n, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=SIZES)
    p.add_argument("split", choices=["val", "test"])
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    main(a.task, a.split, a.workers)
