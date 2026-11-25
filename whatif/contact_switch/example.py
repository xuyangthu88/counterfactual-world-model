"""Illustrate the first held-out pusher-contact change recovered by retraining."""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from whatif.shared.actions import corners, rotate
from whatif.shared.model import SharedGraph, grid
from .collect import OUT
from .evaluate import load_state, graph_belief


@torch.no_grad()
def main():
    d = load_state("pusht", "test", "cpu")
    threshold = json.loads((OUT / "thresholds.json").read_text())["pusht"]
    rows = torch.arange(len(d["ref"]))
    ref = d["ref"]
    actual = d["edge"][..., 0].bool()
    changed = actual != actual[rows, ref, None]
    snapshots = {}
    for name, path in [("old", "results/shared/graph.pt"), ("ours", OUT / "graph.pt")]:
        model = SharedGraph()
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        w = graph_belief(model, d).repeat_interleave(3, -1) / 3
        z, e, _, _ = grid(model, d["state"], d["cmd"])
        p = (e[..., 0].sigmoid() * w[:, None]).sum(2)
        active = p >= threshold[name]
        flips = active != active[rows, ref, None]
        changed &= (flips & (active == actual)) if name == "ours" else ~flips
        snapshots[name] = (z, w, p)
    if not changed.any():
        return
    i, a = changed.nonzero()[0].tolist()
    b = int(ref[i])
    s = d["state"][i].numpy()
    z, w, p = snapshots["ours"]
    weights = w[i].numpy()
    ends = d["end"][i, :, :3].numpy()
    actions = d["cmd"][i].numpy()
    center = 256 + 100 * s[:2]
    colors = ["#5a5a99", "#197461"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), layout="constrained")
    ax = axes[0]

    def boundary(xy, angle, **opts):
        outline = xy + rotate(np.vstack([corners, corners[:1]]), angle)
        ax.plot(*outline.T, **opts)

    boundary(center, s[2], color=".35", lw=2, label="Initial T")
    ax.scatter(*(256 + 100 * s[3:5]), marker="o", s=75, color="#bb7827", label="Initial pusher")
    for idx, c, desc in zip([b, a], colors, ["Reference", "Alternative"]):
        boundary(256 + 100 * ends[idx, :2], ends[idx, 2], color=c, lw=1.1, ls="--")
        start = center + rotate((actions[idx, :2] + [0, 0.45]) * 100, s[2])
        direction = rotate(actions[idx, 2:4], s[2]) * max(6.0, 100 * actions[idx, 4])
        ax.annotate(
            "", xy=start + direction, xytext=start, arrowprops=dict(color=c, arrowstyle="->", lw=2)
        )
        ax.scatter(*start, color=c, s=25, label=f'{desc} ({"hit" if actual[i,idx] else "no hit"})')
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("Simulator x (px)")
    ax.set_ylabel("Simulator y (px)")
    ax.legend(fontsize=8)
    ax.set_title("Same start, two real strokes")
    delta = (z[i, a, :, :2] - z[i, b, :, :2]).numpy() * 100
    obs = (ends[a, :2] - ends[b, :2]) * 100
    ax = axes[1]
    ax.scatter(
        *delta.T, s=15 + 370 * weights, alpha=0.48, color=colors[1], label="Paired particles"
    )
    ax.scatter(*(delta.T @ weights), s=62, marker="D", color=colors[1], label="Posterior mean")
    ax.scatter(*obs, marker="*", s=145, color="#ad5b24", label="Simulator")
    ax.axhline(0, color=".7", lw=0.6)
    ax.axvline(0, color=".7", lw=0.6)
    ax.set_xlabel("Alternative minus reference: x (px)")
    ax.set_ylabel("y (px)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax.set_title("Distribution of the action effect")
    fig.savefig("reports/stages/assets/10-example.png", dpi=175)
    plt.close(fig)
    output = {
        "seed": int(d["seed"][i]),
        "ref_index": b,
        "alt_index": a,
        "pusher_observed": [int(actual[i, b]), int(actual[i, a])],
        "probabilities": {
            n: [float(ps[i, b]), float(ps[i, a])] for n, (_, _, ps) in snapshots.items()
        },
        "delta_xy_px": obs.tolist(),
        "posterior_mean_xy_px": (delta.T @ weights).tolist(),
    }
    (OUT / "example.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
