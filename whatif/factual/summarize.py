import csv
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .collect import OUT, DATA, START


def unify(prefix, target, steps=False):
    files = [
        p for p in sorted(OUT.glob(prefix + "-*.csv")) if p.name.endswith("-steps.csv") == steps
    ]
    rows = []
    for p in files or [OUT / target]:
        with p.open() as f:
            rows.extend(csv.DictReader(f))
    if files:
        with (OUT / target).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return rows


def paired(a):
    draw = np.random.default_rng(11).integers(len(a), size=(6000, len(a)))
    return dict(
        mean=a.mean(0).tolist(),
        ci95=np.quantile(a[draw].mean(1), [0.025, 0.975], axis=0).T.tolist(),
    )


def main():
    unify("controlval", "validation.csv")
    unify("controlval", "validation-trajectory.csv", True)
    rows = unify("control", "control.csv")
    trace = unify("control", "trajectory.csv", True)
    endpoint = {(r["task"], r["policy"], r["seed"]): r for r in trace}
    for r in rows:
        if int(r["steps"]):
            last = endpoint[r["task"], r["policy"], r["seed"]]
            assert (
                r["steps"] == last["steps"]
                and abs(float(r["coverage"]) - float(last["coverage"])) < 1e-6
            )
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.5), layout="constrained")
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    pred = json.loads((OUT / "prediction-test.json").read_text())
    out = {}
    names = {
        "physics": "Physics",
        "ours": "Learned",
        "uniform": "Uniform",
        "legacy": "Old posterior",
        "pets": "PETS",
        "cophy": "CoPhyNet",
        "masked": "Hidden rho",
        "fixed": "Fixed coast",
        "source_only": "Source only",
        "ncii": "NCII",
    }
    colors = {
        "physics": "#899292",
        "ours": "#117568",
        "uniform": "#77a6a4",
        "legacy": "#557b93",
        "pets": "#b77733",
        "cophy": "#7666a3",
        "masked": "#9eb7c4",
        "fixed": "#b0b3b8",
        "ncii": "#7666a3",
        "source_only": "#899292",
    }
    with np.load(OUT / "paired-test.npz") as raw:
        for t in ("causalworld", "pusht", "wall"):
            factor = 65 if t == "causalworld" else 100
            fc = {}
            for n in ("uniform", "legacy", "scaled", "static", "shuffled", "pets", "cophy"):
                fc[n] = {
                    metric: paired(
                        factor * (raw[t + "_ours_" + metric] - raw[t + "_" + n + "_" + metric])
                    )
                    for metric in ["xy", "energy"]
                }
            out[t] = dict(forecast_pairs=fc)
            if t == "wall":
                continue
            samples = {}
            group = {}
            ids = None
            cap, goal = (420, 0.9) if t == "causalworld" else (280, 0.95)
            for method in sorted({r["policy"] for r in rows if r["task"] == t}):
                subset = sorted(
                    [r for r in rows if (r["task"], r["policy"]) == (t, method)],
                    key=lambda r: int(r["seed"]),
                )
                assert [int(r["seed"]) for r in subset] == list(
                    range(START["control"], START["control"] + 160)
                )
                x = np.array(
                    [[int(r["success"]), float(r["coverage"]), int(r["steps"])] for r in subset]
                )
                samples[method] = x
                assert np.all(x[:, 2] <= cap) and np.array_equal(x[:, 0], x[:, 1] > goal)
                protocols = np.array([int(r["protocol"]) for r in subset])
                if ids is not None:
                    np.testing.assert_array_equal(protocols, ids)
                ids = protocols

                def describe(z):
                    return dict(
                        n=len(z),
                        success=int(z[:, 0].sum()),
                        coverage=z[:, 1].mean(),
                        steps=z[:, 2].mean(),
                    )

                group[method] = dict(
                    **describe(x), strata={str(j): describe(x[ids == j]) for j in np.unique(ids)}
                )
            pairs = {}
            for n, x in samples.items():
                if n == "ours":
                    continue
                delta = samples["ours"] - x
                pairs[n] = paired(delta)
                if t == "pusht":
                    pairs[n]["new_damping"] = paired(delta[ids == 1])
            out[t].update(control=group, control_pairs=pairs)
            if t == "pusht":
                out[t]["components"] = {
                    a + " minus " + b: paired(samples[a] - samples[b])
                    for a, b in [("ours", "uniform"), ("uniform", "masked"), ("masked", "fixed")]
                }
            panels = (
                [(axes[0], None, "CausalWorld")]
                if t == "causalworld"
                else [(axes[1], 0, "PushT · default damping"), (axes[2], 1, "PushT · damping 0.25")]
            )
            for ax, g, title in panels:
                order = ["physics", "cophy", "pets", "legacy", "uniform", "ours"]
                for j, n in enumerate(order):
                    x = samples[n] if g is None else samples[n][ids == g]
                    y = 100 * x[:, 0].mean()
                    ax.bar(j, y, color=colors[n], width=0.70)
                    ax.text(j, y + 2, f"{int(x[:,0].sum())}/{len(x)}", ha="center", fontsize=8)
                ax.set_xticks(range(len(order)), [names[n] for n in order], rotation=27, ha="right")
                ax.set_ylim(0, 111)
                ax.set_title(title)
        crossed = json.loads((OUT / "prediction-crossed-test.json").read_text())["wall_crossed"]
        with np.load(OUT / "paired-crossed-test.npz") as cp:
            out["wall_crossed"] = {
                "forecast_pairs": {
                    n: {
                        metric: paired(
                            100
                            * (
                                cp["wall_crossed_ours_" + metric]
                                - cp["wall_crossed_" + n + "_" + metric]
                            ),
                        )
                        for metric in ["xy", "energy"]
                    }
                    for n in ("uniform", "legacy", "scaled", "static", "shuffled", "pets", "cophy")
                }
            }
        graph = crossed
        order = ["source_only", "ncii", "ours"]
        x = np.arange(3)
        for k, (key, label, col) in enumerate(
            [
                ("balanced_error", "BMR (lower)", "#b47548"),
                ("switch_precision", "Switch precision", "#50929e"),
                ("switch_recall", "Switch recall", "#366454"),
            ]
        ):
            y = [100 * graph[n]["edges"][key] for n in order]
            ax2[0].bar(x + (k - 1) * 0.24, y, 0.24, color=col, label=label)
        ax2[0].set_xticks(x, [names[n] for n in order])
        ax2[0].set_ylim(0, 110)
        ax2[0].set_ylabel("Percent")
        ax2[0].set_title("PushT · executable wall interventions")
        ax2[0].legend(fontsize=8)
        methods = ["fixed", "masked", "uniform", "ours", "legacy", "pets"]
        for k, (grp, label) in enumerate([("0", "Native damping"), ("1", "Damping 0.25")]):
            y = [100 * out["pusht"]["control"][n]["strata"][grp]["success"] / 80 for n in methods]
            ax2[1].bar(
                np.arange(len(y)) + (k - 0.5) * 0.37,
                y,
                0.37,
                color=("#80a7a5", "#216b60")[k],
                label=label,
            )
        ax2[1].set_xticks(range(len(methods)), [names[n] for n in methods], rotation=22, ha="right")
        ax2[1].set_ylabel("Native success (%)")
        ax2[1].set_ylim(0, 105)
        ax2[1].set_title("PushT · feedback and posterior")
        ax2[1].legend(fontsize=8)
    axes[0].set_ylabel("Native success (%)")
    for ax in [*axes, *ax2]:
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig("reports/stages/assets/11-control.png", dpi=165)
    fig2.savefig("reports/stages/assets/11-mechanism.png", dpi=165)
    plt.close(fig)
    plt.close(fig2)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    manifest = {}
    for p in sorted(DATA.glob("*.npz")):
        with np.load(p) as d:
            manifest[p.name] = dict(
                n=len(d["seed"]),
                axis1=(
                    "consecutive strokes" if p.stem.endswith("history") else "alternative actions"
                ),
                actions=d["cmd"].shape[1],
                seeds=[int(d["seed"].min()), int(d["seed"].max())],
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
