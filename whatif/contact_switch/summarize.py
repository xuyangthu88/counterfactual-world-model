import csv
import hashlib
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .collect import DATA, OUT, START


def combine(prefix, dest, trace=False):
    files = [
        p for p in sorted(OUT.glob(prefix + "-*.csv")) if p.name.endswith("-steps.csv") == trace
    ]
    records = []
    for path in files or [OUT / dest]:
        with path.open() as f:
            records.extend(csv.DictReader(f))
    if files:
        with (OUT / dest).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    return records


def main():
    combine("controlval", "validation.csv")
    rows = combine("control", "control.csv")
    trace = combine("control", "trajectory.csv", True)
    last = {}
    for row in trace:
        key = (row["task"], row["policy"], row["seed"])
        if key not in last or int(row["decision"]) > int(last[key]["decision"]):
            last[key] = row
    for row in rows:
        key = (row["task"], row["policy"], row["seed"])
        if int(row["steps"]):
            endpoint = last[key]
            assert (
                row["steps"] == endpoint["steps"]
                and abs(float(row["coverage"]) - float(endpoint["coverage"])) < 1e-6
            )
    pred = json.loads((OUT / "prediction-test.json").read_text())
    rng = np.random.default_rng(11)
    result = {}
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.2), layout="constrained")
    colors = {
        "ours": "#0f756b",
        "old": "#729e9d",
        "pets": "#af6f2a",
        "cophy": "#746aa2",
        "physics": "#82878b",
        "uniform": "#8aaac4",
        "ncii": "#785c9b",
    }
    labels = {
        "ours": "WhatIf",
        "old": "Previous graph",
        "pets": "PETS",
        "cophy": "CoPhyNet",
        "physics": "Physics",
        "uniform": "Uniform",
        "ncii": "NCII",
    }

    def paired(diff):
        ix = rng.integers(len(diff), size=(6000, len(diff)))
        return {
            "mean": np.mean(diff, axis=0).tolist(),
            "ci95": np.quantile(diff[ix].mean(1), [0.025, 0.975], axis=0).T.tolist(),
        }

    with np.load(OUT / "paired-test.npz") as errors:
        for k, t in enumerate(["causalworld", "pusht"]):
            scores = {}
            by = {}
            groupids = None
            cap, goal = (420, 0.9) if t == "causalworld" else (280, 0.95)
            for method in sorted({r["policy"] for r in rows if r["task"] == t}):
                group = sorted(
                    [r for r in rows if (r["task"], r["policy"]) == (t, method)],
                    key=lambda r: int(r["seed"]),
                )
                seeds = [int(r["seed"]) for r in group]
                assert seeds == list(range(START["control"], START["control"] + 160))
                x = np.array(
                    [[int(r["success"]), float(r["coverage"]), int(r["steps"])] for r in group]
                )
                assert (x[:, 2] <= cap).all() and np.array_equal(x[:, 0], x[:, 1] > goal)
                ids = np.array([int(r["protocol"]) for r in group])
                if groupids is not None:
                    np.testing.assert_array_equal(ids, groupids)
                groupids = ids

                def describe(a):
                    return dict(
                        n=len(a),
                        success=int(a[:, 0].sum()),
                        coverage=float(a[:, 1].mean()),
                        steps=float(a[:, 2].mean()),
                    )

                scores[method] = x
                by[method] = {
                    **describe(x),
                    "strata": {str(j): describe(x[ids == j]) for j in np.unique(ids)},
                }
            control = {}
            units = 65 if t == "causalworld" else 100
            for n, x in scores.items():
                if n == "ours":
                    continue
                diff = scores["ours"] - x
                control[n] = paired(diff)
                if t == "pusht":
                    control[n]["new_damping"] = paired(diff[groupids == 1])
            position = {
                n: paired(units * (errors[t + "_ours"] - errors[t + "_" + n]))
                for n in ["old", "uniform", "pets", "cophy"]
            }
            # Resample layouts, keeping the correlated sibling actions and sources together.
            new_counts = errors[t + "_ours_switch"]
            edge_ci = {}
            ix = rng.integers(len(new_counts), size=(6000, len(new_counts)))

            def switch_rates(counts):
                tp, predicted, true = counts.T
                return np.stack(
                    (tp / true.clip(min=1), 2 * tp / (predicted + true).clip(min=1)), axis=-1
                )

            for n in ("old", "ncii"):
                old_counts = errors[t + "_" + n + "_switch"]
                diffs = switch_rates(new_counts[ix].sum(1)) - switch_rates(old_counts[ix].sum(1))
                point = switch_rates(new_counts.sum(0)) - switch_rates(old_counts.sum(0))
                edge_ci[n] = dict(
                    metrics=["recall", "f1"],
                    mean=point.tolist(),
                    ci95=np.quantile(diffs, [0.025, 0.975], axis=0).T.tolist(),
                )
            result[t] = dict(
                control=by, paired_control=control, paired_effect_xy=position, paired_switch=edge_ci
            )
            axis = ax[0, k]
            names = ["ours", "old", "ncii"]
            for i, n in enumerate(names):
                m = pred[t][n]["edges"]
                for off, metric, alpha in [
                    (-0.18, "switch_precision", 0.50),
                    (0.18, "switch_recall", 1.0),
                ]:
                    value = 100 * m[metric]
                    axis.bar(i + off, value, width=0.34, color=colors[n], alpha=alpha)
                    axis.text(i + off, value + 1.4, f"{value:.0f}", ha="center", fontsize=9)
            axis.set_xticks(range(3), [labels[n] for n in names])
            axis.set_ylim(0, 100)
            title = "CausalWorld" if t == "causalworld" else "PushT"
            axis.set_title(title + " · contact changes")
            axis.set_ylabel("Precision (light) / recall (dark), %")
            axis = ax[1, k]
            names = ["physics", "cophy", "pets", "old", "ours"]
            sets = (
                [(None, "All")]
                if t == "causalworld"
                else [(0, "Native damping"), (1, "Damping 0.25")]
            )
            for j, n in enumerate(names):
                for h, (group, _) in enumerate(sets):
                    a = scores[n] if group is None else scores[n][groupids == group]
                    xx = j if len(sets) == 1 else j + (h - 0.5) * 0.40
                    avg = 100 * a[:, 0].mean()
                    axis.bar(
                        xx,
                        avg,
                        color=colors[n],
                        alpha=1.0 if len(sets) == 1 or h == 1 else 0.48,
                        width=0.64 if len(sets) == 1 else 0.38,
                    )
                    axis.text(xx, avg + 1.2, f"{int(a[:,0].sum())}", ha="center", fontsize=8)
            axis.set_xticks(range(len(names)), [labels[n] for n in names], rotation=19, ha="right")
            axis.set_title(title + " · native closed-loop success")
            axis.set_ylim(0, 110)
            axis.set_ylabel("Success (%)")
            axis.set_xlabel(
                "Counts out of 160"
                if t == "causalworld"
                else "Counts out of 80: native (light), damping 0.25 (dark)"
            )
    for axis in ax.ravel():
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
        axis.spines[["right", "top"]].set_visible(False)
    fig.savefig("reports/stages/assets/10-contact-and-control.png", dpi=165)
    plt.close(fig)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {}
    for p in sorted(DATA.glob("*.npz")):
        with np.load(p) as d:
            manifest[p.name] = dict(
                n=len(d["seed"]),
                seeds=[int(d["seed"].min()), int(d["seed"].max())],
                keys=sorted(d.files),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
