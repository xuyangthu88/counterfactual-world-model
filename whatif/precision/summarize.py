"""Aggregate paired native outcomes and produce the three report figures."""

import csv
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from .collect import OUT, DATA

ASSETS = Path("reports/stages/assets")
TASKS = ["causalworld", "p9", "pusht", "near", "wall"]
TITLES = {
    "causalworld": "CausalWorld",
    "p9": "CausalWorld P9",
    "pusht": "PushT",
    "near": "近目标",
    "wall": "近墙",
}
NAMES = {
    "graph": "WhatIf",
    "pets": "PETS",
    "cophy": "CoPhyNet",
    "ncii": "NCII",
    "source_only": "仅接触源",
}
COL = {
    "graph": "#146f68",
    "pets": "#bb6c30",
    "cophy": "#79659e",
    "ncii": "#6c81a9",
    "source_only": "#a4a7a4",
}


def paired(values):
    v = np.array(values)
    ix = np.random.default_rng(11).integers(len(v), size=(6000, len(v)))
    return [float(v.mean()), *np.quantile(v[ix].mean(1), [0.025, 0.975]).tolist()]


def read(path):
    return json.loads((OUT / path).read_text())


def outcomes(rows):
    return dict(
        n=len(rows),
        success=int(sum(int(i["success"]) for i in rows)),
        coverage=float(np.mean([float(i["coverage"]) for i in rows])),
        steps=float(np.mean([int(i["steps"]) for i in rows])),
    )


def manifest():
    tab = {}
    for path in sorted(DATA.glob("*.npz")):
        with np.load(path) as f:
            line = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape": {k: list(f[k].shape) for k in f.files},
            }
            if "seed" in f:
                line["seeds"] = [int(f["seed"].min()), int(f["seed"].max())]
            tab[path.name] = line
    (DATA / "manifest.json").write_text(json.dumps(tab, indent=2) + "\n")


def control():
    res = {}
    ci = {}
    cfg = read("freeze.json")
    for task in ("pusht", "causalworld", "p9"):
        rows_by_key = {}
        for path in sorted(OUT.glob(f"control-{task}-*.csv")):
            if path.stem.endswith("-steps"):
                continue
            method, mode = path.stem.split("-")[2:]
            rows = list(csv.DictReader(path.open()))
            traces = list(csv.DictReader((OUT / f"{path.stem}-steps.csv").open()))
            final = {int(i["seed"]): i for i in traces}
            size = 80 if task == "p9" else 160
            first = 657000 if task == "p9" else 651000
            assert [int(i["seed"]) for i in rows] == list(range(first, first + size))
            for item in rows:
                i = int(item["seed"])
                k = int(item["steps"])
                assert 0 <= k <= (280 if task == "pusht" else 420)
                if k:
                    assert k == int(final[i]["steps"]) and float(final[i]["coverage"]) == float(
                        item["coverage"]
                    )
                assert (float(item["coverage"]) > (0.95 if task == "pusht" else 0.9)) == bool(
                    int(item["success"])
                )
            rows_by_key[f"{method}-{mode}"] = rows
        assert all(
            k in rows_by_key
            for k in (
                "graph-selected",
                "pets-selected",
                "cophy-selected",
                "static-selected",
                "uniform-selected",
                "physics-raw",
            )
        )
        if task == "pusht":
            for name in ("graph", "pets", "cophy"):
                for mode in ("raw", "endonly"):
                    if cfg[task][name] == mode:
                        rows_by_key[name + "-" + mode] = rows_by_key[name + "-selected"]
        res[task] = {}
        ci[task] = {}

        def detail(rows):
            return {
                "all": outcomes(rows),
                **{
                    str(p): outcomes([r for r in rows if int(r["protocol"]) == p])
                    for p in sorted({int(i["protocol"]) for i in rows})
                },
            }

        for key, rows in rows_by_key.items():
            res[task][key] = detail(rows)

        def compare(k1, k2):
            r1, r2 = rows_by_key[k1], rows_by_key[k2]
            return {
                scope: paired(
                    [
                        100 * (int(i["success"]) - int(j["success"]))
                        for i, j in zip(r1, r2)
                        if scope == "all" or str(i["protocol"]) == scope
                    ]
                )
                for scope in res[task][k1]
            }

        for key in rows_by_key:
            if key != "graph-selected":
                ci[task]["graph-selected_minus_" + key] = compare("graph-selected", key)
        if task == "pusht":
            for name in ("graph", "pets", "cophy"):
                for mode in ("endonly", "selected"):
                    ci[task][f"{name}-{mode}_minus_raw"] = compare(name + "-" + mode, name + "-raw")
    (OUT / "control-summary.json").write_text(json.dumps(res, indent=2) + "\n")
    (OUT / "paired-control.json").write_text(json.dumps(ci, indent=2) + "\n")
    return res


def crossing_counts():
    summary = {}
    for task in ("pusht", "near"):
        for split in ("val", "test"):
            with np.load(DATA / f"{task}-{split}.npz") as panel:
                initial = panel["state"][:, None, 12]
                hit = (initial <= 0.95) & (panel["peak"] > 0.95)
                terminal = panel["end"][..., 12]
                phase_max = panel["phase"][..., 12].max(2)
                summary[f"{task}-{split}"] = {
                    "observed_crossings": int(hit.sum()),
                    "missed_by_true_end": int((hit & (terminal <= 0.95)).sum()),
                    "missed_by_true_handoffs": int((hit & (phase_max <= 0.95)).sum()),
                }
    (OUT / "native-crossings.json").write_text(json.dumps(summary, indent=2) + "\n")


def figures():
    fpath = Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf"
    if fpath.exists():
        font_manager.fontManager.addfont(str(fpath))
    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )
    old = read("frozen-val.json")
    new = read("refined-val.json")
    test = read("refined-test.json")
    res = control()
    crossing_counts()
    manifest()

    def finish(fig, name):
        fig.savefig(ASSETS / f"12-{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), layout="constrained")
    ax = axes[0]
    x = np.arange(3)
    for j, metric in enumerate(["learned", "exact"]):
        y = [100 * old["pusht"][m]["readout"][metric]["regret"] for m in ("graph", "pets", "cophy")]
        bars = ax.bar(
            x + (j - 0.5) * 0.36,
            y,
            0.33,
            label=["原读出", "精确面积"][j],
            color=["#a5afc0", "#246f83"][j],
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.set_xticks(x, ["WhatIf", "PETS", "CoPhyNet"])
    ax.set_ylim(0, 6.3)
    ax.set_ylabel("最优与所选动作的真实覆盖差（百分点）")
    ax.set_title("同一预测位姿，改善任务读出")
    ax.legend(frameon=False)
    ax = axes[1]
    x = np.arange(1, 4)
    for tab, typ, label, style in [
        (old, "open_px", "Stage 11 连推", "--"),
        (new, "open_px", "分段模型连推", "-"),
        (new, "teacher_px", "分段模型，给定真实起点", ":"),
    ]:
        y = [r[typ] for r in tab["pusht"]["graph"]["rollout"]]
        ax.plot(x, y, style, marker="o", linewidth=2, label=label)
    ax.set_xticks(x)
    ax.set_xlabel("预测到第几推")
    ax.set_ylabel("位置误差（px）")
    ax.set_title("按预测位姿重选接触点时，误差累积")
    ax.legend(frameon=False)
    finish(fig, "diagnosis")
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.5), layout="constrained")
    for j, (tasks, unit) in enumerate(
        [(["causalworld", "p9"], "mm"), (["pusht", "near", "wall"], "px")]
    ):
        ax = axes[j]
        x = np.arange(len(tasks))
        for q, m in enumerate(["graph", "pets", "cophy"]):
            y = [test[t][m]["effect"]["xy"] * (65 if unit == "mm" else 100) for t in tasks]
            ax.bar(x + (q - 1) * 0.24, y, 0.23, label=NAMES[m], color=COL[m])
        ax.set_xticks(x, [TITLES[t] if t != "p9" else "P9" for t in tasks])
        ax.set_ylabel(f"动作效果差的位置误差（{unit}）")
        ax.legend(frameon=False, fontsize=8) if j == 0 else None
    ax = axes[2]
    x = np.arange(3)
    ts = ["causalworld", "pusht", "wall"]
    for q, m in enumerate(["graph", "ncii", "source_only"]):
        y = [test[t][m]["edges"]["switch_f1"] for t in ts]
        ax.bar(x + (q - 1) * 0.24, y, 0.23, label=NAMES[m], color=COL[m])
    ax.set_xticks(x, [TITLES[t] for t in ts])
    ax.set_ylabel("动作换边 F1")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, "prediction")
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), layout="constrained")
    groups = [
        ("pusht", "all", "PushT 全部"),
        ("pusht", "1", "PushT 新阻尼 0.25"),
        ("causalworld", "all", "CausalWorld"),
        ("p9", "all", "P9"),
    ]
    ax = axes[0]
    x = np.arange(4)
    for q, m in enumerate(["graph", "pets", "cophy"]):
        y = [
            100 * res[t][m + "-selected"][g]["success"] / res[t][m + "-selected"][g]["n"]
            for t, g, _ in groups
        ]
        ax.bar(x + (q - 1) * 0.24, y, 0.23, color=COL[m], label=NAMES[m])
    ax.set_xticks(x, ["PushT", "新阻尼", "CW", "P9"])
    ax.set_ylabel("原生任务成功率（%）")
    ax.set_ylim(0, 101)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    x = np.arange(3)
    for q, mode in enumerate(["raw", "endonly", "selected"]):
        y = [res["pusht"][m + "-" + mode]["all"]["success"] for m in ["graph", "pets", "cophy"]]
        bars = ax.bar(
            x + (q - 1) * 0.25,
            y,
            0.24,
            color=["#a5afc0", "#6c92a4", "#146f68"][q],
            label=["原读出", "仅换终点面积", "验证选定"][q],
        )
        ax.bar_label(bars, fontsize=8, padding=1)
    ax.set_ylim(0, 160)
    ax.set_xticks(x, ["WhatIf", "PETS", "CoPhyNet"])
    ax.set_ylabel("相同 160 个 PushT 布局的成功数")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    x = np.arange(3)
    for q, m in enumerate(["uniform", "static", "graph"]):
        y = [
            100 * res[t][m + "-selected"]["all"]["success"] / res[t][m + "-selected"]["all"]["n"]
            for t in ["pusht", "causalworld", "p9"]
        ]
        ax.bar(
            x + (q - 1) * 0.24,
            y,
            0.23,
            color=["#c8c5ba", "#84ada8", "#146f68"][q],
            label=["均匀", "固定平均", "逐事实"][q],
        )
    ax.set_xticks(x, ["PushT", "CW", "P9"])
    ax.set_ylabel("相同模型和执行器的成功率（%）")
    ax.set_ylim(0, 101)
    ax.legend(frameon=False, fontsize=8)
    finish(fig, "control")
    with np.load(OUT / "refined-test.npz") as ar, np.load(OUT / "frozen-test.npz") as prev:
        details = {}
        for t in TASKS:
            unit = 65 if t in ("p9", "causalworld") else 100
            details[t] = {
                k: paired(unit * (ar[t + "-graph-effect"] - a))
                for k, a in [
                    ("old", prev[t + "-graph-effect"]),
                    ("pets", ar[t + "-pets-effect"]),
                    ("cophy", ar[t + "-cophy-effect"]),
                ]
            }
        (OUT / "paired-prediction.json").write_text(json.dumps(details, indent=2) + "\n")


if __name__ == "__main__":
    figures()
