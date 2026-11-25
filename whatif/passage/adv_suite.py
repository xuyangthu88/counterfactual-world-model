"""Run and freeze the common long-horizon return comparison."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .model import OUT
from .advantage import POLICIES


def modes(method):
    return (["base"] if method == "graph" else ["raw"]) + list(POLICIES)


def freeze():
    config = {}
    hashes = {}
    for m in ["graph", "pets", "cophy"]:
        p = OUT / f"advantage-{m}.pt"
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for task in ["pusht", "causalworld"]:
        config[task] = {}
        for model in ["graph", "pets", "cophy"]:
            summary = {}
            for mode in modes(model):
                with (OUT / f"return-controlval-{task}-{model}-{mode}.csv").open() as f:
                    rows = list(csv.DictReader(f))
                assert [int(r["seed"]) for r in rows] == list(range(811000, 811064))
                summary[mode] = [
                    sum(int(r["success"]) for r in rows),
                    sum(float(r["coverage"]) for r in rows) / len(rows),
                    sum(int(r["steps"]) for r in rows) / len(rows),
                ]
            selection = max(summary, key=lambda k: (summary[k][0], summary[k][1], -summary[k][2]))
            learned = max(POLICIES, key=lambda k: (summary[k][0], summary[k][1], -summary[k][2]))
            config[task][model] = {
                "selection": selection,
                "learned": learned,
                "validation": summary,
            }
    config["hashes"] = hashes
    config["test_seeds"] = {
        "pusht": [821000, 821159],
        "causalworld": [821000, 821159],
        "p9": [827000, 827079],
    }
    (OUT / "return-freeze.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)


def jobs_for(phase):
    if phase == "test":
        jobs = [
            (t, m, mode)
            for t in ["pusht", "causalworld", "p9"]
            for m, mode in [
                ("graph", "selected"),
                ("graph", "base"),
                ("pets", "raw"),
                ("cophy", "raw"),
                ("pets", "adapt"),
                ("cophy", "adapt"),
            ]
        ]
        config = json.loads((OUT / "return-freeze.json").read_text())
        for t in ("pusht", "causalworld", "p9"):
            for m in ("graph", "pets", "cophy"):
                if config["causalworld" if t == "p9" else t][m]["selection"] not in POLICIES:
                    jobs.append((t, m, "learned"))
        return jobs
    return [
        (t, m, mode)
        for t in ["pusht", "causalworld"]
        for m in ["graph", "pets", "cophy"]
        for mode in modes(m)
        if (phase == "reference" and mode in ("base", "raw"))
        or (phase == "mix" and mode in POLICIES)
    ]


def run(opts):
    if opts.phase == "freeze":
        return freeze()
    if opts.phase == "test" and not (OUT / "return-freeze.json").exists():
        raise ValueError("Freeze from validation before test.")
    split = "control" if opts.phase == "test" else "controlval"
    logdir = Path("work/stage13")
    logdir.mkdir(parents=True, exist_ok=True)

    def launch(item):
        task, method, mode = item
        count = 64 if split == "controlval" else 80 if task == "p9" else 160
        gpu = (
            opts.baseline_gpu
            if method == "pets" or (method == "cophy" and mode != "raw")
            else opts.graph_gpu
        )
        command = [
            sys.executable,
            "-m",
            "whatif.passage.adv_control",
            split,
            task,
            method,
            mode,
            "--count",
            str(count),
            "--workers",
            str(opts.workers),
        ]
        with (logdir / f"return-{split}-{task}-{method}-{mode}.log").open("w") as f:
            subprocess.run(
                command,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return item

    with ThreadPoolExecutor(opts.parallel) as pool:
        for job in pool.map(launch, jobs_for(opts.phase)):
            print(*job, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["reference", "mix", "freeze", "test"])
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--graph-gpu", type=int, default=0)
    p.add_argument("--baseline-gpu", type=int, default=5)
    run(p.parse_args())
