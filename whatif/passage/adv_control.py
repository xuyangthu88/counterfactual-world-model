"""Native long-horizon value control on an independently seeded second cohort."""

import argparse
import json
import os
import torch
from whatif.shared.planning import mechanical, scale
from whatif.precision.control import run
from .model import OUT
from .planning import choose
from .advantage import features, load_head, models, scores, POLICIES

START = {"controlval": 811000, "control": 821000}


def main(args, *, seeds=START, prefix="return-"):
    method, reported = args.method, args.mode
    task = "causalworld" if args.task == "p9" else args.task
    policy = reported
    if reported in ("selected", "adapt", "learned"):
        freeze = json.loads((OUT / "return-freeze.json").read_text())
        policy = freeze[task][method]["learned" if reported == "learned" else "selection"]
    args.variant = "raw"
    common = dict(output=OUT, start_seeds=seeds, tag_prefix=prefix, output_variant=reported)
    if policy == "raw":
        return run(args, **common)
    m = models(method, args.device)
    head = load_head(method, args.device) if policy in POLICIES else None

    def select(model, s, a, ctx, task, mass, step, seeds, remaining, prev, last):
        if policy == "base":
            return choose(model, s, a, ctx, task, mass, "selected", step, seeds)
        lam = POLICIES[policy]
        ticks = torch.as_tensor(remaining, device=s.device, dtype=s.dtype)
        xx = features(model, method, s, a, ctx, prev, last, ticks, task)
        q = scores(head, xx)
        mech = mechanical(s[:, None], a)
        return (lam * scale(q) + (1 - lam) * scale(mech)).argmax(-1)

    run(args, models=m, selector=select, **common)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("split", choices=["controlval", "control"])
    p.add_argument("task", choices=["causalworld", "pusht", "p9"])
    p.add_argument("method", choices=["graph", "pets", "cophy"])
    p.add_argument("mode")
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument(
        "--sim-python", default=os.environ.get("SIM_PY", "/tmp/whatif-push-venv/bin/python")
    )
    main(p.parse_args())
