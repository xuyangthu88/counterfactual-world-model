"""Native closed-loop policies on Stage 13 layout IDs."""

import argparse
import json
from pathlib import Path
from whatif.precision.control import run

OUT = Path("results/passage")
START = {"controlval": 741000, "control": 751000}


def main(args):
    if args.mode in ("raw", "previous"):
        args.variant = "exact" if args.mode == "previous" and args.task == "pusht" else "raw"
        run(args, output=OUT, start_seeds=START)
        return
    from .planning import choose
    from .model import install
    from whatif.factual.model import AdaptedModels

    m = AdaptedModels(args.device)
    install(m, args.device)
    args.variant = args.mode

    def select(model, s, a, ctx, task, mass, step, seeds, remaining, previous, last):
        return choose(model, s, a, ctx, task, mass, args.mode, step, seeds)

    transform = None
    if args.mode == "averagefact":
        means = json.loads((OUT / "history-reference.json").read_text())["means"]

        def transform(ctx, task):
            ctx = ctx.clone()
            ctx[..., :46] = ctx.new_tensor(means[task])
            return ctx

    run(args, output=OUT, start_seeds=START, models=m, selector=select, transform_context=transform)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["controlval", "control"])
    ap.add_argument("task", choices=["pusht", "causalworld", "p9"])
    ap.add_argument("method", choices=["graph", "pets", "cophy"])
    ap.add_argument("mode")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sim-python", default="/tmp/whatif-push-venv/bin/python")
    main(ap.parse_args())
