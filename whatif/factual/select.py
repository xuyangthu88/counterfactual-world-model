import argparse
import csv
import json
import shutil
from .collect import OUT


def freeze(which):
    if which == "graph":
        scores = {
            n: min(x[1] for x in json.loads((OUT / f"{n}-fit.json").read_text()))
            for n in ("g12", "g32")
        }
        choice = min(scores, key=scores.get)
        shutil.copyfile(OUT / f"{choice}.pt", OUT / "graph.pt")
        (OUT / "graph-choice.json").write_text(
            json.dumps(dict(choice=choice, scores=scores), indent=2) + "\n"
        )
        return
    rows = []
    files = [p for p in OUT.glob("controlval-*.csv") if not p.stem.endswith("-steps")]
    for p in files or [OUT / "validation.csv"]:
        with p.open() as f:
            rows.extend(csv.DictReader(f))
    result = {}
    for t in ("causalworld", "pusht"):
        vals = {}
        for name in ("cem", "enumerate", "root", "pets", "pets_plain", "cophy", "cophy_high"):
            rr = [r for r in rows if (r["task"], r["policy"]) == (t, name)]
            assert len(rr) == len({r["seed"] for r in rr}) == 64
            vals[name] = sum(
                0.60 * int(r["success"])
                + 0.35 * float(r["coverage"])
                - 0.05 * int(r["steps"]) / (420 if t == "causalworld" else 280)
                for r in rr
            ) / len(rr)
        result[t] = dict(
            ours=max(("cem", "enumerate", "root"), key=vals.get),
            terminal=(t == "causalworld") == (vals["pets"] >= vals["pets_plain"]),
            cophy=0.25 if vals["cophy"] >= vals["cophy_high"] else 0.7,
            validation=vals,
        )
    (OUT / "freeze.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["graph", "control"])
    freeze(ap.parse_args().stage)
