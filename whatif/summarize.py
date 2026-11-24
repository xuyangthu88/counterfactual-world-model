"""Compare controllers on matching scenes; resample scenes after averaging training seeds."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()
    with (args.out/"per_scene.csv").open() as f:
        rows = list(csv.DictReader(f))
    metrics = ["end_mm", "success", "overlap", "miss_mm", "miss_deg", "offset_mm",
               "offset_deg", "miss_es", "offset_es", "oracle_mm", "start_mm"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]+"/"+row["mode"]].append(row)
    summary = {"split": {"train":400, "validation":80, "test":160}, "methods":{}}
    matrices = {}
    for name, group in groups.items():
        group.sort(key=lambda r: (int(r["seed"]), int(r["scene"])))
        matrix = np.array([[float(r[k]) for k in metrics] for r in group])
        scenes = sorted({int(r["scene"]) for r in group})
        seeds = sorted({int(r["seed"]) for r in group})
        assert len(group) == len(seeds)*len(scenes)
        matrix = matrix.reshape(len(seeds), len(scenes), len(metrics))
        matrices[name] = matrix.mean(axis=0)
        per_fit = matrix.mean(axis=1)
        summary["methods"][name] = {
            metric: {"mean": float(per_fit[:, j].mean()),
                     "seed_sd": float(per_fit[:, j].std(ddof=1)) if len(seeds)>1 else None,
                     "per_seed": per_fit[:, j].tolist()} for j, metric in enumerate(metrics)}
    rng = np.random.default_rng(20261124)
    resamples = rng.integers(len(scenes), size=(5000,len(scenes)))
    contrasts = {}
    for a,b in [("graph/reflect","graph/prior"),("graph/reflect","dense/reflect"),
                ("graph/reflect","graph/reflect_mean")]:
        delta = matrices[a]-matrices[b]
        bs = delta[resamples].mean(axis=1)
        contrasts[a+" minus "+b] = {k:[float(delta[:,j].mean()), *np.quantile(bs[:,j],[.025,.975]).tolist()]
                                     for j,k in enumerate(metrics)}
    summary["paired_scene_contrasts"] = contrasts
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
