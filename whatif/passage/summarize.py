"""Summarize paired native trials and plot the three questions in the Stage 13 report."""

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .collect import DATA, OUT

ASSETS = Path("reports/stages/assets")
MODELS = {
    "whatif": "WhatIf",
    "pets": "PETS",
    "cophy": "CoPhyNet",
    "stage12": "Stage 12",
    "ncii": "NCII",
    "source": "仅接触源",
}
COL = {
    "whatif": "#19766b",
    "pets": "#bf743e",
    "cophy": "#7963a7",
    "stage12": "#80929c",
    "ncii": "#587da3",
    "source": "#a5a5a0",
}
TASKS = {
    "causalworld": "CausalWorld",
    "pusht": "普通 PushT",
    "near": "近目标",
    "finish": "精调起点",
    "wall": "近墙",
    "p9": "P9",
}


def read(name):
    return json.loads((OUT / f"{name}.json").read_text())


def save(name, d):
    (OUT / f"{name}.json").write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")


def interval(d):
    x = np.asarray(d, dtype=float)
    resample = np.random.default_rng(11).integers(len(x), size=(6000, len(x)))
    return [float(x.mean()), *np.quantile(x[resample].mean(1), [0.025, 0.975]).tolist()]


def outcome(rows):
    return {
        "n": len(rows),
        "success": sum(int(i["success"]) for i in rows),
        "coverage": float(np.mean([float(i["coverage"]) for i in rows])),
        "steps": float(np.mean([int(i["steps"]) for i in rows])),
    }


def controls(folder, pattern, start, name):
    summary = {}
    cis = {}
    for task in ("pusht", "causalworld", "p9"):
        table = {}
        prefix = pattern.format(task=task)
        for path in sorted(folder.glob(prefix + "*.csv")):
            if path.stem.endswith("-steps"):
                continue
            with path.open() as f:
                rows = list(csv.DictReader(f))
            if not rows or "seed" not in rows[0]:
                continue
            first = start + (6000 if task == "p9" else 0)
            n = 80 if task == "p9" else 160
            ids = [int(r["seed"]) for r in rows]
            if ids[0] != first:
                continue
            assert ids == list(range(first, first + n)), (path, ids)
            for r in rows:
                assert 0 <= int(r["steps"]) <= (280 if task == "pusht" else 420)
                assert int(r["success"]) == (
                    float(r["coverage"]) > (0.95 if task == "pusht" else 0.90)
                )
            trace = path.with_name(path.stem + "-steps.csv")
            with trace.open() as f:
                ends = {int(r["seed"]): r for r in csv.DictReader(f)}
            for r in rows:
                if int(r["steps"]):
                    last = ends[int(r["seed"])]
                    assert last["coverage"] == r["coverage"] and last["steps"] == r["steps"]
            table[path.stem[len(prefix) :]] = rows
        assert table, (folder, task, pattern)
        group_values = sorted({r["protocol"] for r in next(iter(table.values()))})
        summary[task] = {}
        for key, rows in table.items():
            summary[task][key] = {
                "all": outcome(rows),
                **{v: outcome([r for r in rows if r["protocol"] == v]) for v in group_values},
            }
        cis[task] = {}
        for k, rows in table.items():
            for j, base in table.items():
                if k == j:
                    continue
                cis[task][k + " minus " + j] = {}
                for v in ["all", *group_values]:
                    delta = [
                        100 * (int(a["success"]) - int(b["success"]))
                        for a, b in zip(rows, base)
                        if v == "all" or a["protocol"] == v
                    ]
                    cis[task][k + " minus " + j][v] = interval(delta)
    save(name + "-summary", summary)
    save(name + "-paired", cis)
    return summary


def records():
    files = {}
    for path in sorted(DATA.glob("*.npz")):
        with np.load(path) as f:
            item = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "seeds": [int(f["seed"].min()), int(f["seed"].max())],
                "fields": {k: list(f[k].shape) for k in f.files},
            }
            if "remaining" in f:
                y = f["success"]
                item.update(
                    reachable=int(y.any(1).sum()),
                    action_dependent=int((y.min(1) != y.max(1)).sum()),
                    average_remaining=float(f["remaining"].mean()),
                    remaining_range=[int(f["remaining"].min()), int(f["remaining"].max())],
                )
            files[path.name] = item
    (DATA / "manifest.json").write_text(json.dumps(files, indent=2) + "\n")
    with np.load(OUT / "paired-test.npz") as f:
        comp = {}
        for t in TASKS:
            comp[t] = {}
            for measure in ("xy", "brier"):
                key = t + "-whatif-" + measure
                if key not in f:
                    continue
                comp[t][measure] = {
                    k: interval(f[key] - f[t + "-" + k + "-" + measure])
                    for k in ("stage12", "pets", "cophy", "mean_fact")
                }
        save("paired-prediction", comp)


def style():
    font = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )


def export(fig, name):
    fig.savefig(ASSETS / f"13-{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def graph_and_pose():
    style()
    z = read("prediction-test")
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.25), layout="constrained")
    for i, (tasks, unit, mult) in enumerate(
        [(["causalworld", "p9"], "mm", 65), (["pusht", "finish", "wall"], "px", 100)]
    ):
        x = np.arange(len(tasks))
        for j, k in enumerate(["whatif", "pets", "cophy"]):
            y = [z[t][k]["effect"]["xy"] * mult for t in tasks]
            ax[i].bar(x + (j - 1) * 0.23, y, 0.22, color=COL[k], label=MODELS[k])
        ax[i].set_xticks(x, [TASKS[t] for t in tasks])
        ax[i].set_ylabel(f"配对动作效果误差（{unit}）")
        if i == 0:
            ax[i].legend(frameon=False, fontsize=8)
    x = np.arange(3)
    ts = ["causalworld", "pusht", "wall"]
    for j, k in enumerate(("whatif", "ncii", "source")):
        ax[2].bar(
            x + (j - 1) * 0.23,
            [z[t][k]["edges"]["switch_f1"] for t in ts],
            0.22,
            color=COL[k],
            label=MODELS[k],
        )
    ax[2].set_xticks(x, [TASKS[t] for t in ts])
    ax[2].set_ylabel("换动作引起接触切换的 F1")
    ax[2].legend(frameon=False, fontsize=8)
    export(fig, "world-model")


def passage_figure():
    style()
    ex = read("crossing-example")
    z = read("prediction-test")
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.4), layout="constrained")
    for j, color in enumerate(["#19766b", "#bb7850"]):
        length = ex["phase_ticks"][j][-1]
        vals = [ex["start"], *ex["coverage_path"][j][:length]]
        ax[0].plot(
            np.arange(length + 1),
            vals,
            color=color,
            linewidth=2.4,
            label=f"动作 {ex['actions'][j]}",
        )
        ax[0].scatter([length], [vals[-1]], color=color, s=28)
    tick = ex["first_passage"][0]
    ax[0].axhline(0.95, color="#888", ls="--", lw=1)
    ax[0].scatter(
        [tick], [ex["coverage_path"][0][tick - 1]], color="#19766b", marker="*", s=90, zorder=6
    )
    ax[0].set_title(f"同一起点的两次真实推动 · 布局 {ex['seed']}")
    ax[0].set_xlabel("原生步")
    ax[0].set_ylabel("T 形目标面积覆盖率")
    ax[0].legend(frameon=False)
    ts = ["causalworld", "pusht", "finish"]
    x = np.arange(3)
    for j, k in enumerate(["stage12", "whatif", "pets", "cophy"]):
        pick = [z[t][k]["crossing"] for t in ts]
        ax[1].bar(
            x + (j - 1.5) * 0.19,
            [d["selected_hits"] / d["attainable"] * 100 for d in pick],
            0.18,
            color=COL[k],
            label=MODELS[k],
        )
    ax[1].set_xticks(x, [TASKS[t] for t in ts])
    ax[1].set_ylim(0, 108)
    ax[1].set_ylabel("有可行动作时，一推选中成功动作（%）")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=8)
    export(fig, "crossing")


def control_figure(first, second, final):
    style()
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 3.5), layout="constrained")
    for a, table, labels, heading in [
        (
            ax[0],
            first,
            {"whatif": "graph-selected", "pets": "pets-raw", "cophy": "cophy-raw"},
            "途中命中规划",
        ),
        (
            ax[1],
            second,
            {"whatif": "graph-selected", "pets": "pets-raw", "cophy": "cophy-raw"},
            "整局回报与机械分 · 验证选择",
        ),
        (
            ax[2],
            final,
            {"whatif": "graph-joint", "pets": "pets-raw", "cophy": "cophy-raw"},
            "整局回报与图规划 · 固定组合",
        ),
    ]:
        groups = [
            ("pusht", "all", "PushT"),
            ("pusht", "1", "新阻尼"),
            ("causalworld", "all", "CausalWorld"),
            ("p9", "all", "P9"),
        ]
        x = np.arange(len(groups))
        for j, k in enumerate(["whatif", "pets", "cophy"]):
            y = [table[t][labels[k]][g] for t, g, _ in groups]
            a.bar(
                x + (j - 1) * 0.24,
                [i["success"] * 100 / i["n"] for i in y],
                0.23,
                color=COL[k],
                label=MODELS[k],
            )
        a.set_xticks(x, [v[-1] for v in groups])
        a.set_ylim(0, 102)
        a.set_title(heading)
        a.set_ylabel("原生任务成功率（%）")
    ax[1].legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    export(fig, "control")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    records()
    graph_and_pose()
    passage_figure()
    first = controls(OUT, "control-{task}-", 751000, "control")
    second = controls(OUT, "return-control-{task}-", 821000, "return-control")
    final = controls(OUT, "joint-control-{task}-", 841000, "joint-control")
    control_figure(first, second, final)


if __name__ == "__main__":
    main()
