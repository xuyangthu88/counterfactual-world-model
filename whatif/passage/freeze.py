"""Select the native planner on held-out validation layouts before any control test."""

import csv
import hashlib
import json
from pathlib import Path
from .model import OUT


def run():
    configs = {}
    for task in ("pusht", "causalworld"):
        modes = ["beam", "guided", "cem"] + (["timed"] if task == "pusht" else [])
        scores = {}
        for mode in modes:
            with (OUT / f"controlval-{task}-graph-{mode}.csv").open() as f:
                rows = list(csv.DictReader(f))
            assert [int(r["seed"]) for r in rows] == list(range(741000, 741064))
            scores[mode] = dict(
                success=sum(int(r["success"]) for r in rows),
                coverage=sum(float(r["coverage"]) for r in rows) / len(rows),
                steps=sum(int(r["steps"]) for r in rows) / len(rows),
            )
        best = max(
            modes,
            key=lambda mode: (
                scores[mode]["success"],
                scores[mode]["coverage"],
                -scores[mode]["steps"],
            ),
        )
        configs[task] = dict(policy=best, validation=scores)
    configs["p9"] = {"policy": configs["causalworld"]["policy"], "transfer": "causalworld"}
    configs["hashes"] = {
        str(file): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in (
            OUT / "graph.pt",
            OUT / "duration.pt",
            Path("results/precision/graph.pt"),
            Path("results/precision/pets.pt"),
            Path("results/precision/cophy.pt"),
        )
    }
    configs["controls"] = {
        "pusht": [751000, 751159],
        "causalworld": [751000, 751159],
        "p9": [757000, 757079],
        "primary": ["selected", "pets/raw", "cophy/raw"],
        "comparators": ["previous", "averagefact", "noevent"],
    }
    (OUT / "freeze.json").write_text(json.dumps(configs, indent=2) + "\n")
    print(json.dumps(configs, indent=2))


if __name__ == "__main__":
    run()
