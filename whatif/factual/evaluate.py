import argparse
import json
import numpy as np
import torch
from whatif.crossbench.evaluate import metrics
from whatif.crossbench.models import geometry_cover
from whatif.contact_switch.evaluate import edge_metrics
from whatif.shared.model import grid, posterior, unpack, task_value, log_measured
from whatif.shared.evaluate import contrast
from .model import AdaptedModels, evidence, cases
from .collect import OUT


@torch.no_grad()
def evaluate(split, device, crossed=False):
    torch.set_num_threads(2)
    m = AdaptedModels(device)
    calibration = OUT / ("crossed-calibration.json" if crossed else "calibration.json")
    cal = {} if split == "val" else json.loads(calibration.read_text())
    results = {}
    paired = {}
    for task in ("wall_crossed",) if crossed else ("causalworld", "pusht", "wall"):
        d = cases(task, split, device)
        s, a, y, ref = (d[k] for k in ("state", "cmd", "end", "ref"))
        i = torch.arange(len(s), device=device)
        b = len(s)
        na = a.shape[1]
        st = s[:, None].expand_as(y)
        mask = torch.arange(na, device=device)[None] != ref[:, None]
        n = na - 1
        ctx = evidence(d["previous"], d["fact"], s)
        root = grid(m.graph, d["previous"], d["fact"][:, None])[0][:, 0]
        old = (
            posterior(d["previous"], s, root[:, None], torch.zeros_like(ref), 0.055)
            .reshape(b, 5, 3)
            .sum(-1)
        )
        actual_res = contrast(root.mean(1), s)
        scale = actual_res.square().mean(0).sqrt().clamp_min(0.025)
        if split == "val":
            cal[task] = {"sigma": scale.tolist(), "threshold": {}}
        sig = s.new_tensor(cal[task]["sigma"])
        errors = contrast(root, s[:, None]) / sig
        cal_ll = -0.5 * errors.square().sum(-1) + log_measured(s).repeat_interleave(3, -1)
        scaled = cal_ll.softmax(-1).reshape(b, 5, 3).sum(-1)
        learned = m.graph.belief(ctx)
        uniform = torch.full_like(learned, 0.2)
        if split == "val":
            cal[task]["mean_posterior"] = learned.mean(0).tolist()
        static = s.new_tensor(cal[task]["mean_posterior"]).expand_as(learned)
        # Swap completed-stroke evidence between layouts; keep the receiving state's velocity
        # and coasting observations, and every input to the future graph, intact.
        g = torch.Generator(device=device).manual_seed(11)
        order = torch.randperm(b, generator=g, device=device)
        swap = order.clone()
        swap[order] = order.roll(-1)
        misplaced = torch.cat((ctx[swap, :46], ctx[:, 46:]), -1)
        shuffled = m.graph.belief(misplaced)
        z, e, pk, rank = grid(m.graph, s, a)
        value = 0.55 * task_value(st[:, :, None], z, pk) + 0.45 * rank
        pred = {}
        probs = {}
        scores = {}
        for name, weight in [
            ("ours", learned),
            ("uniform", uniform),
            ("legacy", old),
            ("scaled", scaled),
            ("static", static),
            ("shuffled", shuffled),
        ]:
            w = weight.repeat_interleave(3, -1) / 3
            pred[name] = (z, w)
            scores[name] = (value * w[:, None]).sum(-1)
            if name == "ours":
                probs[name] = (e.sigmoid() * w[:, None, :, None]).sum(2)
        co, _ = m.cophy(d["previous"], s, d["fact"], s, a)
        coret, _ = m.cophy(s, y[i, ref], a[i, ref], s, a)
        # Standard CoPhyNet A/B-to-C encoding accepts the just observed factual branch.
        pred["cophy"] = (co[:, :, None], s.new_ones(b, 1))
        cs = st.clone()
        cs[..., :3] = co
        cs[..., 12] = geometry_cover(co[..., :2], co[..., 2], st)
        scores["cophy"] = task_value(st, cs, cs[..., 12])
        c = ctx[:, None, :].expand(b, na, -1).reshape(b * na, 52)
        mu, sd = m.predict(st.reshape(-1, 19), a.reshape(-1, 7), c)
        torch.manual_seed(11)
        eps = (
            torch.randn(5, 1, 1, 1, 16, device=device)
            * s.new_tensor([-1.22474487, 0.0, 1.22474487])[None, None, None, :, None]
        )
        dy = mu.reshape(5, b, na, 1, 16) + sd.reshape(5, b, na, 1, 16) * eps
        pets, peak = unpack(
            st[:, :, None].expand(-1, -1, 15, -1), dy.permute(1, 2, 0, 3, 4).reshape(b, na, 15, 16)
        )
        pred["pets"] = (pets, s.new_full((b, 15), 1 / 15))
        scores["pets"] = task_value(st[:, :, None], pets, peak).mean(2)
        true = contrast(y, y[i, ref, None])[mask].reshape(b, n, 3)
        end = y[mask].reshape(b, n, 19)[..., :3]
        truth_value = task_value(st, y, d["peak"])
        true_edge = d["edge"].cpu().numpy().astype(bool)
        ref_np = ref.cpu().numpy()
        result = {}
        for name, (p, w) in pred.items():
            delta = contrast(p, p[i, ref, None])[mask].reshape(b, n, -1, 3)
            final = p[mask].reshape(b, n, -1, p.shape[-1])[..., :3]
            raw, xy = metrics(delta, true, w)
            absolute = metrics(final, end, w)[0]
            rw = w
            if name == "ours":
                residual = contrast(p[i, ref], y[i, ref, None]) / sig
                rw = (w.clamp_min(1e-9).log() - 0.5 * residual.square().sum(-1)).softmax(-1)
            if name == "cophy":
                delta = contrast(coret, y[i, ref, None])[mask].reshape(b, n, 1, 3)
            retro = metrics(delta, true, rw)[0]
            vv = scores[name]
            regret = truth_value.amax(1) - truth_value[i, vv.argmax(1)]
            result[name] = dict(
                future=raw, absolute=absolute, same_event=retro, regret=float(regret.mean())
            )
            paired[f"{task}_{name}_xy"] = xy.cpu().numpy()
            # Proper two-dimensional distribution score per independent layout.
            pxy = (contrast(p, p[i, ref, None])[mask].reshape(b, n, -1, 3))[..., :2].contiguous()
            distance = torch.cdist(pxy, pxy, compute_mode="donot_use_mm_for_euclid_dist")
            energy = ((pxy - true[:, :, None, :2]).norm(dim=-1) * w[:, None]).sum(2) - 0.5 * (
                distance * w[:, None, :, None] * w[:, None, None, :]
            ).sum((2, 3))
            paired[f"{task}_{name}_energy"] = energy.mean(1).cpu().numpy()
        train = cases(task, "train", device)
        base_rate = train["edge"].float().mean((0, 1))
        probs["source_only"] = base_rate.expand_as(e[:, :, 0])
        probs["ncii"] = m.ncii.edges(st, a).sigmoid()
        for name, prob in probs.items():
            pp = prob.cpu().numpy()
            if split == "val":
                thresholds = np.linspace(0.025, 0.975, 39)
                stats = [edge_metrics(true_edge, pp, ref_np, th) for th in thresholds]
                j = min(
                    range(len(stats)),
                    key=lambda j: stats[j]["balanced_error"] + 1 - stats[j]["switch_f1"],
                )
                cal[task]["threshold"][name] = float(thresholds[j])
            ed = edge_metrics(true_edge, pp, ref_np, cal[task]["threshold"][name], breakdown=True)
            result.setdefault(name, {})["edges"] = ed
        known = s[:, 18] > 0
        groups = {}
        for g in d["protocol"].unique().tolist():
            j = d["protocol"] == g
            groups[str(g)] = dict(
                n=int(j.sum()),
                old_mass=old[j].mean(0).tolist(),
                learned_mass=learned[j].mean(0).tolist(),
                root_rmse=actual_res[j].square().mean(0).sqrt().tolist(),
                hits_prior_boundary=float((old[j].argmax(1).remainder(4) == 0).float().mean()),
                measured=float(known[j].float().mean()),
                rho=float(s[j, 17][known[j]].mean()) if known[j].any() else None,
            )
        result["diagnosis"] = dict(
            val_scale=sig.tolist(),
            learned_neff=float((1 / learned.square().sum(1)).mean()),
            old_neff=float((1 / old.square().sum(1)).mean()),
            mass_l1=float((learned - 0.2).abs().sum(1).mean()),
            groups=groups,
        )
        paired[task + "_seed"] = d["seed"].cpu().numpy()
        results[task] = result
    if split == "val":
        calibration.write_text(json.dumps(cal, indent=2) + "\n")
    label = ("crossed-" if crossed else "") + split
    (OUT / f"prediction-{label}.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(OUT / f"paired-{label}.npz", **paired)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--crossed", action="store_true")
    args = ap.parse_args()
    evaluate(args.split, args.device, args.crossed)
