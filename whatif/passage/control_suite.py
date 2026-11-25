"""Execute Stage13 validation or frozen comparisons in real simulators."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from .model import OUT


def run(args):
    if args.split == "control":
        if not (OUT / "freeze.json").is_file():
            raise ValueError("Select the validation configuration first.")
        jobs = [
            (t, m, policy)
            for t in ("pusht", "causalworld", "p9")
            for m, policy in (
                ("graph", "selected"),
                ("pets", "raw"),
                ("cophy", "raw"),
                ("graph", "previous"),
                ("graph", "averagefact"),
                ("graph", "noevent"),
            )
        ]
    else:
        jobs = [
            (t, m, policy)
            for t in ("pusht", "causalworld")
            for m, policy in (
                ("graph", "beam"),
                ("graph", "guided"),
                ("graph", "cem"),
                ("pets", "raw"),
                ("cophy", "raw"),
            )
        ]
        jobs.append(("pusht", "graph", "timed"))
    logdir = Path("work/stage13")
    logdir.mkdir(parents=True, exist_ok=True)

    def launch(job):
        t, m, policy = job
        n = 64 if args.split == "controlval" else 80 if t == "p9" else 160
        cmd = [
            sys.executable,
            "-m",
            "whatif.passage.control",
            args.split,
            t,
            m,
            policy,
            "--count",
            str(n),
            "--workers",
            str(args.native_workers),
        ]
        gpu = args.baseline_gpu if m == "pets" or policy == "previous" else args.graph_gpu
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        with (logdir / f"{args.split}-{t}-{m}-{policy}.log").open("w") as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        return job

    with ThreadPoolExecutor(args.parallel) as pool:
        for job in pool.map(launch, jobs):
            print(*job, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("split", choices=["controlval", "control"])
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--graph-gpu", type=int, default=0)
    p.add_argument("--baseline-gpu", type=int, default=5)
    p.add_argument("--native-workers", type=int, default=6)
    run(p.parse_args())
