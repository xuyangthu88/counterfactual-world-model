import csv
import json
from .collect import OUT


def rows():
    paths = [p for p in sorted(OUT.glob("controlval-*.csv")) if "-steps" not in p.name]
    for p in paths or [OUT / "validation.csv"]:
        with p.open() as f:
            yield from csv.DictReader(f)


if __name__ == "__main__":
    records = list(rows())
    score = {}
    for task in ("pusht", "causalworld"):
        score[task] = {}
        for name in ("direct", "cost", "old"):
            group = [r for r in records if r["policy"] == name and r["task"] == task]
            assert len(group) == len({x["seed"] for x in group}) == 80
            score[task][name] = sum(
                0.60 * int(x["success"])
                + 0.35 * float(x["coverage"])
                - 0.05 * int(x["steps"]) / (280 if task == "pusht" else 420)
                for x in group
            ) / len(group)
    choice = max(["direct", "cost"], key=lambda n: sum(v[n] for v in score.values()))
    (OUT / "freeze.json").write_text(
        json.dumps(
            dict(choice=choice, contact_cost=0.25 if choice == "cost" else 0.0, validation=score),
            indent=2,
        )
        + "\n"
    )
    print(choice, score)
