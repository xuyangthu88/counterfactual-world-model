"""Independent same-state intervention tests, native crossings and repeated-response distributions."""

import argparse
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from whatif.factual.model import AdaptedModels, evidence, cases
from whatif.precision.model import install as install_phase
from whatif.precision.evaluate import particles
from whatif.precision.geometry import overlap
from whatif.crossbench.evaluate import metrics
from whatif.contact_switch.evaluate import edge_metrics
from whatif.shared.evaluate import contrast
from .collect import DATA, OUT, COUNTS
from .history import average_evidence

UNIT = {"pusht": 100, "near": 100, "finish": 100, "wall": 100, "causalworld": 65, "p9": 65}


def load_panel(name, split, device):
    with np.load(DATA / f"{name}-{split}.npz") as f:
        row = {k: torch.as_tensor(f[k], device=device) for k in f.files}
    return {k: v.float() if v.is_floating_point() else v for k, v in row.items()}


def forecasts(old, new, d, task):
    from .model import passage_grid

    s = d["state"]
    a = d["cmd"]
    previous, edge_old = particles(old, d)
    ctx = evidence(d["previous"], d["fact"], s)
    z, e, peak, rank, event = passage_grid(new.graph, s, a, ctx)
    prob = new.graph.belief(ctx).repeat_interleave(3, -1) / 3
    mean_ctx = average_evidence(ctx, task, new.history_means)
    zm, _, pm, _, hm = passage_grid(new.graph, s, a, mean_ctx)
    avg = new.graph.belief(mean_ctx).repeat_interleave(3, -1) / 3
    result = {"whatif": (z, prob, peak, event), "mean_fact": (zm, avg, pm, hm)}
    tau = 0.90 if task in ("causalworld", "p9") else 0.95
    for k, (x, w, pk) in previous.items():
        if k == "graph" and task not in ("causalworld", "p9"):
            # Stage12 validation-selected planner uses maximum of current and predicted terminal area.
            pk = torch.maximum(s[:, None, None, 12], overlap(x[..., :3], s[:, None, None, 9:12]))
        result["stage12" if k == "graph" else k] = (x, w, pk, (pk > tau).float())
    return result, (e.sigmoid() * prob[:, None, :, None]).sum(2), edge_old


def crossing(prob, s, first, tau):
    # Exclude episodes already successful before intervention. Inspect both pooled
    # probability error and whether the highest-ranked action really crosses.
    valid = s[:, 12] <= tau
    pp = prob[valid].cpu().numpy()
    yy = (first[valid] <= 42).cpu().numpy()
    active = yy.any(1)
    if len(pp) == 0:
        return {}
    choice = pp.argmax(1)
    pick = yy[np.arange(len(yy)), choice]
    return dict(
        layouts=len(pp),
        trials=int(yy.size),
        hits=int(yy.sum()),
        attainable=int(active.sum()),
        selected_hits=int(pick.sum()),
        brier=float(np.mean((pp - yy) ** 2)),
        ap=float(average_precision_score(yy.ravel(), pp.ravel())) if yy.any() else None,
    )


def estimate_pose(z, w, d):
    s = d["state"]
    y = d["end"]
    r = d["ref"]
    i = torch.arange(len(s), device=s.device)
    n = y.shape[1] - 1
    other = torch.arange(n + 1, device=s.device)[None] != r[:, None]
    delta = contrast(z, z[i, r, None])[other].reshape(len(s), n, -1, 3)
    truth = contrast(y, y[i, r, None])[other].reshape(len(s), n, 3)
    effect, per = metrics(delta, truth, w)
    pose = metrics(z[..., :3], y[..., :3], w)[0]
    return dict(pose=pose, effect=effect), per


def repeat(name, split, old, new, device):
    d = load_panel("repeat-" + name, split, device)
    s = d["state"]
    y = d["end"]
    r = d["ref"]
    inputs = dict(d)
    inputs["end"] = y[:, :, 1]
    preds, _, _ = forecasts(old, new, inputs, name)
    i = torch.arange(len(s), device=device)
    mask = torch.arange(y.shape[1], device=device)[None] != r[:, None]
    truth = contrast(y, y[i, r, None])[..., :2]
    b, n = mask.shape
    n -= 1
    truth = truth[mask].view(b, n, 3, 2)
    unit = UNIT[name]
    real_pass = (d["first_passage"] <= 42).float().mean(2)
    tau = 0.90 if name == "causalworld" else 0.95
    eligible = s[:, 12] <= tau
    positive = real_pass.max(1).values > 0
    result = {}
    for k, (z, w, peak, p_hit) in preds.items():
        dx = contrast(z, z[i, r, None])[..., :2][mask].reshape(b, n, -1, 2)
        dist = torch.cdist(dx, truth, compute_mode="donot_use_mm_for_euclid_dist")
        within = torch.cdist(dx, dx, compute_mode="donot_use_mm_for_euclid_dist")
        yy = torch.cdist(truth, truth, compute_mode="donot_use_mm_for_euclid_dist")
        es = (dist.mean(-1) * w[:, None]).sum(-1) - 0.5 * (
            within * w[:, None, :, None] * w[:, None, None, :]
        ).sum((-2, -1))
        energy_distance = (2 * es - yy.mean((-2, -1))).clamp_min(0)
        mu = (dx * w[:, None, :, None]).sum(2)
        dispersion = ((dx - mu[:, :, None]).norm(dim=-1) * w[:, None]).sum(2)
        empirical_disp = (truth - truth.mean(2, keepdim=True)).norm(dim=-1).mean(2)
        # Inferred physical uncertainty and execution noise have different meanings.
        # Separate the conditional second moments for the five equally stratified noise triples.
        if dx.shape[2] == 15:
            cube = dx.reshape(b, n, 5, 3, 2)
            mu_k = cube.mean(3)
            pk = w.reshape(b, 5, 3).sum(2)
            proc = ((cube - mu_k.unsqueeze(3)).square().sum(-1).mean(3) * pk[:, None]).sum(2)
            latent = ((mu_k - mu[:, :, None]).square().sum(-1) * pk[:, None]).sum(2)
        else:
            proc = torch.zeros_like(es)
            latent = torch.zeros_like(es)
        actual_variance = (truth - truth.mean(2, keepdim=True)).square().sum(-1).mean(2)
        prob = (p_hit * w[:, None]).sum(2)
        best = prob.argmax(1)
        chosen = real_pass[i, best]
        # A positive gap means another executable candidate had a higher real hit frequency.
        regret = real_pass.max(1).values - chosen
        result[k] = dict(
            energy_score=float(es.mean()) * unit,
            energy_distance=float(energy_distance.mean()) * unit,
            mean_radius_pred=float(dispersion.mean()) * unit,
            mean_radius_observed=float(empirical_disp.mean()) * unit,
            rms_within=float(proc.mean().sqrt()) * unit,
            rms_between=float(latent.mean().sqrt()) * unit,
            rms_repeated=float(actual_variance.mean().sqrt()) * unit,
            event_probability_mse=float((prob[eligible] - real_pass[eligible]).square().mean()),
            attainable=int((eligible & positive).sum()),
            selected_hit_probability=(
                float(chosen[eligible & positive].mean()) if (eligible & positive).any() else None
            ),
            hit_frequency_regret=(
                float(regret[eligible & positive].mean()) if (eligible & positive).any() else None
            ),
        )
    return result


def main(split, device):
    from .model import install

    torch.set_num_threads(2)
    old = AdaptedModels(device)
    install_phase(old, device)
    new = AdaptedModels(device)
    install(new, device)
    new.history_means = json.loads((OUT / "history-reference.json").read_text())["means"]
    thresholds_path = OUT / "edge-cutoffs.json"
    thresholds = {} if split == "val" else json.loads(thresholds_path.read_text())
    res = {}
    paired = {}
    for task, counts in COUNTS.items():
        if not counts[split == "test"]:
            continue
        d = load_panel(task, split, device)
        s = d["state"]
        y = d["end"]
        a = d["cmd"]
        preds, edge, other_edge = forecasts(old, new, d, task)
        tau = 0.90 if task in ("causalworld", "p9") else 0.95
        valid = s[:, 12] <= tau
        r = d["ref"]
        out = {}
        paired[task + "-seed"] = d["seed"].cpu().numpy()
        for key, (z, w, pk, event) in preds.items():
            out[key], per = estimate_pose(z, w, d)
            paired[task + "-" + key + "-xy"] = per.cpu().numpy() * UNIT[task]
            p = (event * w[:, None]).sum(2)
            pk = (pk * w[:, None]).sum(2)
            if "first_passage" in d:
                actual = (d["first_passage"] <= 42).float()
                truth = d["peak"]
                assert torch.all((truth > tau)[valid] == actual[valid].bool())
                out[key]["crossing"] = crossing(p, s, d["first_passage"], tau)
                out[key]["crossing"]["peak_mae"] = float((pk[valid] - truth[valid]).abs().mean())
                paired[task + "-" + key + "-brier"] = (
                    (p[valid] - actual[valid]).square().mean(1).cpu().numpy()
                )
            out[key]["paired_peak_mae"] = float(
                ((pk - pk.gather(1, r[:, None])) - (d["peak"] - d["peak"].gather(1, r[:, None])))
                .abs()
                .sum()
                / (len(s) * (a.shape[1] - 1))
            )
        st = s[:, None].expand_as(y)
        train_name = (
            task if task in ("wall", "causalworld") else "causalworld" if task == "p9" else "pusht"
        )
        base = cases(train_name, "train", "cpu")["edge"].float().mean((0, 1)).to(device)
        probs = {
            "whatif": edge,
            "stage12": other_edge,
            "ncii": old.ncii.edges(st, a).sigmoid(),
            "source": base.expand_as(edge),
        }
        er = d["edge"].cpu().numpy().astype(bool)
        ref = r.cpu().numpy()
        if split == "val":
            thresholds[task] = {}
        for key, pr in probs.items():
            v = pr.cpu().numpy()
            if split == "val":

                def crit(cutoff):
                    score = edge_metrics(er, v, ref, cutoff)
                    return score["balanced_error"] + 1 - score["switch_f1"]

                thresholds[task][key] = float(min(np.linspace(0.025, 0.975, 39), key=crit))
            cut = thresholds["causalworld" if task == "p9" else task][key]
            out.setdefault(key, {})["edges"] = edge_metrics(er, v, ref, cut, breakdown=True)
        if task == "finish" and split == "test":
            # First eligible seed in collection order. Model predictions do not enter selection.
            transient = (d["first_passage"] <= 42) & (y[..., 12] <= tau) & valid[:, None]
            miss = d["first_passage"] > 42
            ids = torch.where(transient.any(1) & miss.any(1))[0]
            if len(ids):
                j = int(ids[0])
                ia = int(torch.where(transient[j])[0][0])
                ib = int(torch.where(miss[j])[0][0])
                example = {
                    "seed": int(d["seed"][j]),
                    "actions": [ia, ib],
                    "start": float(s[j, 12]),
                    "prediction": {},
                }
                for field in (
                    "coverage_path",
                    "first_passage",
                    "phase_ticks",
                    "phase_peak",
                    "edge",
                    "peak",
                ):
                    example[field] = d[field][j, [ia, ib]].cpu().tolist()
                example["target_delta"] = contrast(y[j, ia], y[j, ib]).cpu().tolist()
                for name, (z, w, _, phit) in preds.items():
                    delta = contrast(z[j, ia], z[j, ib])
                    est = (delta * w[j, :, None]).sum(0)
                    example["prediction"][name] = {
                        "p_hit": (phit[j, [ia, ib]] * w[j]).sum(-1).cpu().tolist(),
                        "delta": est.cpu().tolist(),
                    }
                (OUT / "crossing-example.json").write_text(json.dumps(example, indent=2) + "\n")
        out["layouts"] = len(s)
        res[task] = out
    if split == "val":
        thresholds_path.write_text(json.dumps(thresholds, indent=2) + "\n")
    (OUT / f"prediction-{split}.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(OUT / f"paired-{split}.npz", **paired)
    repeated = {t: repeat(t, split, old, new, device) for t in ("pusht", "causalworld", "finish")}
    (OUT / f"repeated-{split}.json").write_text(
        json.dumps(repeated, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("split", choices=["val", "test"])
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    with torch.no_grad():
        main(a.split, a.device)
