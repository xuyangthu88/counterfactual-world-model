"""Rank one intervention with observed branch returns and fixed continuation."""

import csv
import json
from pathlib import Path
import numpy as np
import torch
from whatif.shared.planning import mechanical, scale
from .advantage import models, prepare, load_head, scores, POLICIES, return_value
from .model import OUT


@torch.no_grad()
def run():
    device = "cuda"
    table = {}
    per_layout = []
    for task in ("causalworld", "pusht"):
        with np.load(f"data/passage/return-{task}-val.npz") as f:
            s = torch.as_tensor(f["state"], device=device).float()
            a = torch.as_tensor(f["cmd"], device=device).float()
            seeds = f["seed"]
        mech = mechanical(s[:, None], a)
        tab = {}
        for m in ("graph", "pets", "cophy"):
            model = models(m, device)
            x, y = prepare(model, m, task, "val", device)
            q = scores(load_head(m, device), x)
            truth = return_value(y)
            ids = torch.arange(len(y), device=device)
            cases = {
                "head": q,
                **{k: lam * scale(q) + (1 - lam) * scale(mech) for k, lam in POLICIES.items()},
            }
            if m == "graph":
                cases.update(physics=mech, oracle=truth)
            for mode, value in cases.items():
                choice = value.argmax(1)
                chosen = y[ids, choice]
                gap = truth.max(1).values - truth[ids, choice]
                label = f"{m}/{mode}" if mode not in ("physics", "oracle") else mode
                tab[label] = dict(
                    n=len(s),
                    success=int(chosen[:, 0].sum()),
                    coverage=float(chosen[:, 1].mean()),
                    fraction_budget_used=float(chosen[:, 2].mean()),
                    regret=float(gap.mean()),
                )
                per_layout.extend(
                    (task, label, int(seed), int(act), int(hit), float(g))
                    for seed, act, hit, g in zip(seeds, choice.cpu(), chosen[:, 0].cpu(), gap.cpu())
                )
        table[task] = tab
    (OUT / "return-ranking-validation.json").write_text(json.dumps(table, indent=2) + "\n")
    with (OUT / "return-ranking-validation.csv").open("w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["task", "policy", "seed", "action", "success", "regret"])
        out.writerows(per_layout)


if __name__ == "__main__":
    torch.set_num_threads(2)
    run()
