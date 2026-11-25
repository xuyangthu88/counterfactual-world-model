"""Run each preregistered native comparison once per task/planner."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .collect import OUT


def main(args):
    config = json.loads((OUT / "freeze.json").read_text())
    log = Path("work/stage12")
    jobs = []
    for task in ("pusht", "causalworld", "p9"):
        for name in ("graph", "pets", "cophy", "physics", "uniform", "static"):
            jobs.append((task, name, "selected" if name != "physics" else "raw"))
    jobs.append(("pusht", "graph", "stage11"))
    for name in ("graph", "pets", "cophy"):
        for mode in ("raw", "endonly"):
            if config["pusht"][name] != mode:
                jobs.append(("pusht", name, mode))

    def run(x):
        task, name, mode = x
        gpu = args.graph_gpu if name in ("graph", "static", "uniform") else args.baseline_gpu
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        command = [
            sys.executable,
            "-m",
            "whatif.precision.control",
            "control",
            task,
            name,
            mode,
            "--count",
            "80" if task == "p9" else "160",
        ]
        with (log / f"test-{task}-{name}-{mode}.log").open("w") as f:
            subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(task, name, mode, flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(run, jobs))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graph-gpu", type=int, default=0)
    p.add_argument("--baseline-gpu", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
