"""Domain averages of completed-stroke features estimated on the Stage 12 training set."""

import json
import torch
from whatif.factual.model import evidence
from whatif.precision.evaluate import panel
from .collect import OUT


def main():
    means = {}
    for task in ("causalworld", "pusht"):
        d = panel(task, "train", "cpu")
        x = evidence(d["previous"], d["fact"], d["state"])
        means[task] = x[:, :46].mean(0).tolist()
    (OUT / "history-reference.json").write_text(json.dumps({"means": means}, indent=2) + "\n")


def average_evidence(context, task, means):
    new = context.clone()
    key = "causalworld" if task in ("causalworld", "p9") else "pusht"
    new[..., :46] = context.new_tensor(means[key])
    return new


if __name__ == "__main__":
    with torch.no_grad():
        main()
