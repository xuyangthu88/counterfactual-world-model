"""Use the learned full-episode action return to rerank complete graph-planner scores."""

import argparse
import hashlib
import json
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
import torch
from whatif.shared.planning import scale
from whatif.precision.control import run
from .adv_control import main as prior_runner
from .advantage import features, load_head, models, scores
from .history import average_evidence
from .planning import beam
from .collect import OUT
from pathlib import Path

SEEDS = {"controlval": 831000, "control": 841000}
WEIGHT = 0.30


def freeze():
    names = [
        "graph.pt",
        "advantage-graph.pt",
        "advantage-pets.pt",
        "advantage-cophy.pt",
        "freeze.json",
        "return-freeze.json",
    ]
    record = dict(
        primary="graph-joint",
        rule="0.70 scaled full two-step paired graph return + 0.30 scaled full-episode return; fixed before validation and test",
        validation=[831000, 831063],
        test=[841000, 841159],
        P9=[847000, 847079],
        comparisons=[
            "original beam policy",
            "previous-history averaged",
            "fixed PETS CEM",
            "fixed CoPhyNet",
            "validation-selected same-data baselines",
        ],
        sha256={key: hashlib.sha256((OUT / key).read_bytes()).hexdigest() for key in names},
    )
    (OUT / "joint-freeze.json").write_text(json.dumps(record, indent=2) + "\n")


def execute(args):
    if args.mode not in ("joint", "average"):
        return prior_runner(args, seeds=SEEDS, prefix="joint-")
    task = "causalworld" if args.task == "p9" else args.task
    rule = json.loads((OUT / "freeze.json").read_text())[task]["policy"]
    assert rule in ("guided", "beam")
    m = models("graph", args.device)
    head = load_head("graph", args.device)
    means = json.loads((OUT / "history-reference.json").read_text())["means"]

    def select(m, s, a, ctx, task, mass, step, seeds, remaining, prev, last):
        rem = s.new_tensor(remaining)
        decisions = []
        for j in range(0, len(s), 3):
            ix = slice(j, j + 3)
            short = beam(m, s[ix], a[ix], ctx[ix], task, mass[ix], rule, return_scores=True)
            xx = features(m, "graph", s[ix], a[ix], ctx[ix], prev[ix], last[ix], rem[ix], task)
            long = scores(head, xx)
            decisions.append(((1 - WEIGHT) * scale(short) + WEIGHT * scale(long)).argmax(-1))
        return torch.cat(decisions)

    transform = (lambda ctx, t: average_evidence(ctx, t, means)) if args.mode == "average" else None
    args.variant = args.mode
    return run(
        args,
        models=m,
        selector=select,
        output=OUT,
        start_seeds=SEEDS,
        tag_prefix="joint-",
        transform_context=transform,
    )


def suite(split):
    freeze = json.loads((OUT / "return-freeze.json").read_text())
    jobs = []
    for t in ("pusht", "causalworld", "p9") if split == "control" else ("pusht", "causalworld"):
        jobs.extend([(t, "graph", "joint"), (t, "graph", "base")])
        if split == "control":
            jobs.extend([(t, "graph", "average"), (t, "pets", "raw"), (t, "cophy", "raw")])
            for m in ("pets", "cophy"):
                if freeze["causalworld" if t == "p9" else t][m]["selection"] != "raw":
                    jobs.append((t, m, "adapt"))

    def launch(job):
        t, m, mode = job
        n = (80 if t == "p9" else 160) if split == "control" else 64
        cmd = [sys.executable, "-m", "whatif.passage.joint", split, t, m, mode, "--count", str(n)]
        gpu = (
            os.environ.get("BASELINE_GPU", "5") if m == "pets" else os.environ.get("GRAPH_GPU", "0")
        )
        with Path(f"work/stage13/joint-{split}-{t}-{m}-{mode}.log").open("w") as f:
            subprocess.run(
                cmd,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(*job, flush=True)

    with ThreadPoolExecutor(int(os.environ.get("PARALLEL_JOBS", "7"))) as pool:
        list(pool.map(launch, jobs))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("split", choices=[*SEEDS, "freeze"])
    p.add_argument("task", nargs="?")
    p.add_argument("method", nargs="?")
    p.add_argument("mode", nargs="?")
    p.add_argument("--count", type=int)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--sim-python", default=os.environ.get("SIM_PY", "/tmp/whatif-push-venv/bin/python")
    )
    a = p.parse_args()
    if a.split == "freeze":
        freeze()
    elif a.task:
        execute(a)
    else:
        suite(a.split)
