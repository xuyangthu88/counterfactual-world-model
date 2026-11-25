"""Lock planner choices and record checkpoint/validation provenance before test."""

import csv
import hashlib
import json
from .collect import OUT


def main():
    config = {
        "causalworld": dict(graph="raw", pets="raw", cophy="raw"),
        "pusht": {},
        "validation": {},
    }
    for method in ("graph", "pets", "cophy"):
        scores = {}
        candidates = (
            ("raw", "endonly", "exact")
            if method == "cophy"
            else ("raw", "endonly", "exact", "local")
        )
        for mode in candidates:
            file = OUT / f"controlval-pusht-{method}-{mode}.csv"
            with file.open() as f:
                rows = list(csv.DictReader(f))
            assert [int(x["seed"]) for x in rows] == list(range(641000, 641064))
            scores[mode] = [
                sum(int(i["success"]) for i in rows),
                sum(float(i["coverage"]) for i in rows) / len(rows),
                sum(int(i["steps"]) for i in rows) / len(rows),
            ]
        config["validation"][method] = scores
        config["pusht"][method] = max(
            scores, key=lambda k: (scores[k][0], scores[k][1], -scores[k][2])
        )
    config["checkpoints"] = {
        key: hashlib.sha256((OUT / f"{key}.pt").read_bytes()).hexdigest()
        for key in ("graph", "pets", "cophy")
    }
    config["primary_seeds"] = [651000, 651159]
    config["p9_seeds"] = [657000, 657079]
    config["test_comparisons"] = {
        "pusht": "three selected policies, physics, old graph and planner, per-model raw/endonly, graph uniform/static",
        "causalworld": "three selected policies, physics, graph static/uniform",
        "p9": "three transferred policies, physics, graph static/uniform",
    }
    (OUT / "freeze.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
