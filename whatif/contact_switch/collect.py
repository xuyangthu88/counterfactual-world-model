"""Unseen paired-action panels. Control and prediction use separate reset ranges."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from whatif.prospective.collect import collect

DATA = Path("data/contact_switch")
OUT = Path("results/contact_switch")
START = {"val": 411000, "test": 421000, "controlval": 431000, "control": 441000}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["causalworld", "pusht"])
    ap.add_argument("split", choices=["val", "test"])
    ap.add_argument("count", type=int)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    # Include both intervention spaces in CausalWorld validation on fresh layouts.
    protocol = "val" if a.task == "pusht" and a.split == "val" else "test"
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(
            pool.map(
                collect,
                [(a.task, protocol, i) for i in range(START[a.split], START[a.split] + a.count)],
            )
        )
    np.savez_compressed(
        DATA / f"{a.task}-{a.split}.npz", **{k: [r[k] for r in rows] for k in rows[0]}
    )
