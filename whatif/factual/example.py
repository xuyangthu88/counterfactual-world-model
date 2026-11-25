"""First native diagnostic test layout containing free, pusher-only and wall-contact outcomes."""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from whatif.shared.actions import corners, rotate
from whatif.shared.model import grid
from .collect import OUT, DATA
from .model import FactualGraph, evidence


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    with np.load(DATA / "wall_crossed-test.npz") as f:
        d = dict(f)
    for i, edges in enumerate(d["edge"].astype(bool)):
        groups = [
            ~edges.any(-1),
            edges[:, 0] & ~edges[:, 1:].any(-1),
            edges[:, 0] & edges[:, 1:].any(-1),
        ]
        if all(np.any(g) for g in groups):
            break
    else:
        raise ValueError("No layout with all three observed outcomes")
    indices = [np.flatnonzero(g)[0] for g in groups]
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    state = d["state"][i]
    xy = state[:2] * 100 + 256
    angle = state[2]
    model = FactualGraph()
    model.load_state_dict(torch.load(OUT / "graph.pt", map_location="cpu", weights_only=True))
    w = (
        model.belief(
            evidence(t(d["previous"][i : i + 1]), t(d["fact"][i : i + 1]), t(state[None]))
        ).repeat_interleave(3, -1)
        / 3
    )
    z, edges, _, _ = grid(model, t(state[None]), t(d["cmd"][i : i + 1]))
    prob = (edges.sigmoid() * w[:, None, :, None]).sum(2)[0].numpy()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), layout="constrained")
    shapes = []
    for j in indices:
        end = d["end"][i, j]
        shapes.append(end[:2] * 100 + 256 + rotate(corners, end[2]))
    initial = xy + rotate(corners, angle)
    pusher = state[3:5] * 100 + 256
    pts = np.concatenate([initial, pusher[None], *shapes])
    lo = np.maximum(0, pts.min(0) - 26)
    hi = np.minimum(511, pts.max(0) + 26)
    labels = ["Stand off", "Pusher only", "Pusher and wall"]
    records = []
    for ax, j, shape, label in zip(axes, indices, shapes, labels):
        for q, c, style in [(initial, "#888888", "--"), (shape, "#147869", "-")]:
            ax.plot(*np.vstack([q, q[:1]]).T, color=c, ls=style, lw=1.8)
        ax.scatter(*pusher, s=50, c="#ae6f2e")
        a = d["cmd"][i, j]
        site = xy + rotate((a[:2] + [0, 0.45]) * 100, angle)
        direc = rotate(a[2:4], angle) * 100 * a[4]
        ax.scatter(*site, color="#55538d", s=22)
        if np.linalg.norm(direc) > 1:
            ax.annotate(
                "",
                xy=site + direc,
                xytext=site,
                arrowprops=dict(arrowstyle="->", color="#55538d", lw=2),
            )
        for border in [5, 506]:
            ax.axvline(border, c="#384343", lw=3)
            ax.axhline(border, c="#384343", lw=3)
        ax.set(
            xlim=(lo[0], hi[0]),
            ylim=(hi[1], lo[1]),
            aspect="equal",
            title=f"{label} · action {j}",
            xlabel="x (px)",
        )
        ax.spines[["right", "top"]].set_visible(False)
        records.append(
            dict(action=int(j), actual=d["edge"][i, j].astype(int).tolist(), prob=prob[j].tolist())
        )
    axes[0].set_ylabel("y (px)")
    fig.savefig("reports/stages/assets/11-wall-example.png", dpi=165)
    plt.close(fig)
    (OUT / "example.json").write_text(
        json.dumps({"seed": int(d["seed"][i]), "examples": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
