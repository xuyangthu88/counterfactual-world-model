"""Matched interventions, open-loop errors, and validation of the native task readout."""

import argparse
import json
import numpy as np
import torch
from whatif.contact_switch.evaluate import edge_metrics
from whatif.crossbench.models import geometry_cover, wrap
from whatif.crossbench.evaluate import metrics
from whatif.shared.evaluate import contrast
from whatif.shared.model import grid, unpack
from whatif.shared.planning import menus
from whatif.factual.model import AdaptedModels, evidence, cases
from .geometry import overlap
from .collect import DATA, OUT


def panel(task, split, device):
    with np.load(DATA / f"{task}-{split}.npz") as f:
        row = {k: torch.as_tensor(f[k], device=device) for k in f.files}
    return {k: v.float() if v.is_floating_point() else v for k, v in row.items()}


def particles(model, d):
    s, a, end = d["state"], d["cmd"], d["end"]
    st = s[:, None].expand_as(end)
    b, n = a.shape[:2]
    ctx = evidence(d["previous"], d["fact"], s)
    z, edge, peak, _ = grid(model.graph, s, a)
    w = model.graph.belief(ctx).repeat_interleave(3, -1) / 3
    co, _ = model.cophy(d["previous"], s, d["fact"], s, a)
    c = st.clone()
    c[..., :3] = co
    c[..., 12] = geometry_cover(co[..., :2], co[..., 2], st)
    mu, sd = model.predict(
        st.reshape(-1, 19), a.reshape(-1, 7), ctx[:, None].expand(-1, n, -1).reshape(-1, 52)
    )
    g = torch.Generator(device=s.device).manual_seed(11)
    eps = (
        torch.randn(5, 1, 1, 1, 16, generator=g, device=s.device)
        * s.new_tensor([-1.224745, 0.0, 1.224745])[None, None, None, :, None]
    )
    dy = mu.reshape(5, b, n, 1, 16) + sd.reshape(5, b, n, 1, 16) * eps
    p, pk = unpack(
        st[:, :, None].expand(-1, -1, 15, -1), dy.permute(1, 2, 0, 3, 4).reshape(b, n, 15, 16)
    )
    return {
        "graph": (z, w, peak),
        "pets": (p, torch.full_like(w, 1 / 15), pk),
        "cophy": (c[:, :, None], s.new_ones(b, 1), c[..., 12, None]),
    }, (edge.sigmoid() * w[:, None, :, None]).sum(2)


def rollout(model, d, particles):
    s = d["state"]
    i = torch.arange(len(s), device=s.device)
    k = s.new_tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).repeat_interleave(3).expand(len(s), -1)
    st = particles[0][i, d["ref"]]
    w = particles[1]
    y = d["end"][i, d["ref"]]
    err = (
        lambda a, b: float((((a[..., :2] * w[..., None]).sum(1) - b[:, :2]).norm(dim=-1)).mean())
        * 100
    )
    result = [dict(open_px=err(st, y), teacher_px=err(st, y))]
    for h in range(2):
        u = s.new_tensor([-1.0, 0.0, 1.0]).roll(h + 1).repeat(5).expand_as(k)

        def transition(x):
            actions = menus(x, "pusht")
            idx = d["future_idx"][:, h, None, None, None].expand(len(x), 15, 1, 7)
            return model.graph(x, actions.gather(-2, idx).squeeze(-2), k, u)[0]

        forced = transition(y[:, None].expand_as(st))
        st = transition(st)
        y = d["future"][:, h]
        result.append(dict(open_px=err(st, y), teacher_px=err(forced, y)))
    return result


def readouts(z, w, peak, d):
    s = d["state"]
    truth = d["end"][..., 12]
    tp = d["peak"]

    def measure(c, target):
        pred = (c * w[:, None]).sum(-1)
        pick = pred.argmax(1, keepdim=True)
        return dict(
            mae=float((pred - target).abs().mean()),
            regret=float((target.amax(1) - target.gather(1, pick)[:, 0]).mean()),
        )

    g = overlap(z[..., :3], s[:, None, None, 9:12])
    return dict(
        learned=measure(z[..., 12], truth),
        exact=measure(g, truth),
        learned_peak=measure(peak, tp),
        endpoint_as_peak=measure(g, tp),
    )


def main(args):
    split = args.split
    torch.set_num_threads(2)
    m = AdaptedModels(args.device)
    if args.refined:
        from .model import install

        install(m, args.device)
    summary = {}
    arrays = {}
    mean_file = OUT / "posterior-mean.json"
    means = {} if split == "val" else json.loads(mean_file.read_text())
    if split == "val":
        with np.load(DATA / "geometry-ref.npz") as f:
            p = torch.from_numpy(f["pose"])
            g = torch.from_numpy(f["goal"])
            y = f["expected"]
            variants = {
                "cpu64": overlap(p, g),
                "gpu32": overlap(p.to(args.device).float(), g.to(args.device).float()),
            }
            summary["reference"] = {
                k: {
                    "n": len(y),
                    "max": float(np.max(abs(v.cpu().numpy() - y))),
                    "mae": float(np.mean(abs(v.cpu().numpy() - y))),
                    "threshold_mismatches": int(np.sum((v.cpu().numpy() > 0.95) != (y > 0.95))),
                }
                for k, v in variants.items()
            }
    threshold_file = OUT / ("thresholds.json" if args.refined else "frozen-thresholds.json")
    th = {} if split == "val" else json.loads(threshold_file.read_text())
    for task in ["pusht", "near", "causalworld", "wall"] + (["p9"] if split == "test" else []):
        d = panel(task, split, args.device)
        s = d["state"]
        r = d["ref"]
        i = torch.arange(len(s), device=s.device)
        pred, edge = particles(m, d)
        res = {}
        y = d["end"]
        n = y.shape[1] - 1
        mask = torch.arange(n + 1, device=s.device)[None] != r[:, None]
        truth = contrast(y, y[i, r, None])[mask].reshape(len(s), n, 3)
        graph_z, w, pk = pred["graph"]
        if split == "val":
            means[task] = w.reshape(len(s), 5, 3).sum(2).mean(0).tolist()
        if args.refined:
            prior = (
                s.new_tensor(means["causalworld" if task == "p9" else task]).repeat_interleave(3)
                / 3
            )
            pred["static"] = (graph_z, prior.expand_as(w), pk)
            pred["uniform"] = (graph_z, torch.full_like(w, 1 / 15), pk)
        for name, (z, w, peak) in pred.items():
            delta = contrast(z, z[i, r, None])[mask].reshape(len(s), n, -1, 3)
            cf, per = metrics(delta, truth, w)
            absolute = metrics(z[..., :3], y[..., :3], w)[0]
            res[name] = dict(effect=cf, pose=absolute)
            arrays[f"{task}-{name}-effect"] = per.cpu().numpy()
            if task in ("pusht", "near", "wall"):
                res[name]["readout"] = readouts(z, w, peak, d)
        if task in ("pusht", "near"):
            res["graph"]["rollout"] = rollout(m, d, pred["graph"])
        st = s[:, None].expand_as(y)
        actual = d["edge"].cpu().numpy().astype(bool)
        ref = r.cpu().numpy()
        source = (
            cases(
                (
                    "wall"
                    if task == "wall"
                    else "causalworld" if task in ("p9", "causalworld") else "pusht"
                ),
                "train",
                "cpu",
            )["edge"]
            .float()
            .mean((0, 1))
            .to(s.device)
        )
        probs = {
            "graph": edge,
            "ncii": m.ncii.edges(st, d["cmd"]).sigmoid(),
            "source_only": source.expand_as(edge),
        }
        if split == "val":
            th[task] = {}
        for name, pr in probs.items():
            pp = pr.cpu().numpy()
            if split == "val":
                trials = np.linspace(0.025, 0.975, 39)

                def criterion(v):
                    scores = edge_metrics(actual, pp, ref, v)
                    return scores["balanced_error"] + 1 - scores["switch_f1"]

                th[task][name] = float(min(trials, key=criterion))
            cutoff = th["causalworld" if task == "p9" else task][name]
            res.setdefault(name, {})["edges"] = edge_metrics(
                actual, pp, ref, cutoff, breakdown=True
            )
        res["layouts"] = len(s)
        summary[task] = res
        arrays[f"{task}-seed"] = d["seed"].cpu().numpy()
    if split == "val":
        threshold_file.write_text(json.dumps(th, indent=2) + "\n")
        if args.refined:
            mean_file.write_text(json.dumps(means, indent=2) + "\n")
    stem = ("refined" if args.refined else "frozen") + "-" + split
    (OUT / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(OUT / f"{stem}.npz", **arrays)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("split", choices=["val", "test"])
    p.add_argument("--refined", action="store_true")
    p.add_argument("--device", default="cuda")
    with torch.no_grad():
        main(p.parse_args())
