import argparse
import json
import numpy as np
import torch
from whatif.crossbench.evaluate import classify, metrics
from whatif.crossbench.models import geometry_cover
from whatif.shared.evaluate import contrast
from whatif.shared.model import grid, posterior, unpack, task_value
from whatif.shared.planning import Models
from .collect import DATA, OUT


def load_state(task, split, device):
    with np.load(DATA / f"{task}-{split}.npz") as f:
        d = {k: torch.as_tensor(f[k], device=device) for k in f.files}
    return {k: v.float() if v.is_floating_point() else v for k, v in d.items()}


def graph_belief(model, d):
    device = d["state"].device
    n = len(d["b"])
    z0 = grid(
        model, d["previous"], d["previous_actions"][torch.arange(n, device=device), d["b"]][:, None]
    )[0]
    w0 = posterior(d["previous"], d["state"], z0, torch.zeros_like(d["b"]), 0.055)
    return w0.reshape(n, 5, 3).sum(-1)


def edge_metrics(y, p, ref, threshold, breakdown=False):
    ix = np.arange(len(y))
    other = np.arange(y.shape[1])[None] != ref[:, None]
    yy = y[other].reshape(len(y), -1, 3)
    pp = p[other].reshape(yy.shape)
    stat = classify(yy, pp, threshold)
    real = yy != y[ix, ref, None]
    switch = (pp >= threshold) != (p[ix, ref, None] >= threshold)
    tp = int(np.sum(real & switch))
    pred = int(switch.sum())
    actual = int(real.sum())
    stat.update(
        switch_tp=tp,
        switch_predicted=pred,
        switch_actual=actual,
        switch_precision=tp / max(1, pred),
        switch_recall=tp / max(1, actual),
        switch_f1=2 * tp / max(1, pred + actual),
    )
    if breakdown:
        ref_true = y[ix, ref, None]
        ref_pred = p[ix, ref, None] >= threshold

        def counts(actual, predicted):
            tp = int((actual & predicted).sum())
            return dict(
                correct=tp,
                actual=int(actual.sum()),
                predicted=int(predicted.sum()),
                precision=tp / max(1, int(predicted.sum())),
                recall=tp / max(1, int(actual.sum())),
            )

        stat["birth"] = counts(~ref_true & yy, ~ref_pred & (pp >= threshold))
        stat["loss"] = counts(ref_true & ~yy, ref_pred & (pp < threshold))
        stat["by_source"] = [counts(real[..., j], switch[..., j]) for j in range(yy.shape[-1])]
    return stat


@torch.no_grad()
def evaluate(split, device):
    torch.set_num_threads(2)
    m = Models(device)
    from copy import deepcopy

    joint = deepcopy(m.graph)
    joint.load_state_dict(torch.load(OUT / "graph.pt", weights_only=True, map_location=device))
    thresholds = {} if split == "val" else json.loads((OUT / "thresholds.json").read_text())
    result = {}
    paired = {}
    for t in ("causalworld", "pusht"):
        d = load_state(t, split, device)
        s = d["state"]
        a = d["cmd"]
        r = d["ref"]
        i = torch.arange(len(s), device=device)
        y = d["end"]
        other = torch.arange(12, device=device)[None] != r[:, None]
        st = s[:, None].expand_as(y)
        target = contrast(y, y[i, r, None])[other].reshape(len(s), 11, 3)
        targets = y[other].reshape(len(s), 11, 19)[..., :3]
        pred = {}
        probs = {}
        values = {}
        result[t] = {}
        true_value = task_value(st, y, d["peak"])
        switchable = (d["edge"].bool() != d["edge"][i, r, None].bool()).any(-1).any(-1)
        for name, net in [("ours", joint), ("old", m.graph)]:
            mass = graph_belief(net, d)
            w = mass.repeat_interleave(3, -1) / 3
            z, logit, peak, rank = grid(net, s, a)
            value = 0.55 * task_value(st[:, :, None], z, peak) + 0.45 * rank
            pred[name] = (z, w)
            probs[name] = (logit.sigmoid() * w[:, None, :, None]).sum(2)
            values[name] = (value * w[:, None]).sum(2)
            if name == "ours":
                flat = torch.full_like(w, 1 / 15)
                pred["uniform"] = (z, flat)
                values["uniform"] = (value * flat[:, None]).sum(2)
        probs["ncii"] = m.ncii.edges(st, a).sigmoid()
        co, _ = m.cophy(d["previous"], s, d["previous_actions"][i, d["b"]], s, a)
        pred["cophy"] = (co[:, :, None], s.new_ones(len(s), 1))
        cs = st.clone()
        cs[..., :3] = co
        cs[..., 12] = geometry_cover(co[..., :2], co[..., 2], st)
        values["cophy"] = task_value(st, cs, cs[..., 12])
        same_co = m.cophy(s, y[i, r], a[i, r], s, a)[0]
        mean, sd = m.predict(st.reshape(-1, 19), a.reshape(-1, 7))
        torch.manual_seed(11)
        noise = (
            torch.randn(5, 1, 1, 1, 16, device=device)
            * s.new_tensor([-1.22474487, 0.0, 1.22474487])[None, None, None, :, None]
        )
        delta = mean.reshape(5, len(s), 12, 1, 16) + sd.reshape(5, len(s), 12, 1, 16) * noise
        pets, pk = unpack(
            st[:, :, None].expand(-1, -1, 15, -1),
            delta.permute(1, 2, 0, 3, 4).reshape(len(s), 12, 15, 16),
        )
        pred["pets"] = (pets, s.new_full((len(s), 15), 1 / 15))
        values["pets"] = task_value(st[:, :, None], pets, pk).mean(2)
        for name, (z, w) in pred.items():
            rel = contrast(z, z[i, r, None])[other].reshape(len(s), 11, -1, 3)
            absolute = z[other].reshape(len(s), 11, -1, z.shape[-1])[..., :3]
            fv, per = metrics(rel, target, w)
            regret = true_value.amax(1) - true_value[i, values[name].argmax(-1)]
            exposed = w
            if name in ("ours", "old"):
                er = contrast(z[i, r], y[i, r, None]) / s.new_tensor([0.055, 0.055, 0.088])
                exposed = (w.clamp_min(1.0e-10).log() - 0.5 * er.square().sum(-1)).softmax(-1)
            ret_rel = rel
            if name == "cophy":
                ret_rel = contrast(same_co, y[i, r, None])[other].reshape(len(s), 11, 1, 3)
            result[t][name] = {
                "future": fv,
                "absolute": metrics(absolute, targets, w)[0],
                "same_event": metrics(ret_rel, target, exposed)[0],
                "regret": float(regret.mean()),
                "regret_on_switchable": float(regret[switchable].mean()),
                "switchable_layouts": int(switchable.sum()),
            }
            paired[t + "_" + name] = per.cpu().numpy()
        thresholds.setdefault(t, {})
        for name, p in probs.items():
            yy = d["edge"].cpu().numpy().astype(bool)
            pp = p.cpu().numpy()
            ref = r.cpu().numpy()
            if split == "val":
                # Minimize existence BMR and switch F1 loss equally for all three methods.
                candidates = np.r_[
                    np.arange(0.01, 0.10, 0.01),
                    np.arange(0.10, 0.91, 0.025),
                    np.arange(0.91, 1.0, 0.01),
                ]
                all_metrics = [edge_metrics(yy, pp, ref, h) for h in candidates]
                idx = min(
                    range(len(candidates)),
                    key=lambda j: all_metrics[j]["balanced_error"]
                    + 1
                    - all_metrics[j]["switch_f1"],
                )
                thresholds[t][name] = float(candidates[idx])
            edge = edge_metrics(yy, pp, ref, thresholds[t][name], breakdown=True)
            result[t].setdefault(name, {})["edges"] = edge
            observed_switch = yy != yy[np.arange(len(yy)), ref, None]
            hard = pp >= thresholds[t][name]
            predicted_switch = hard != hard[np.arange(len(yy)), ref, None]
            paired[t + "_" + name + "_switch"] = np.stack(
                [
                    (observed_switch & predicted_switch).sum((1, 2)),
                    predicted_switch.sum((1, 2)),
                    observed_switch.sum((1, 2)),
                ],
                axis=-1,
            )
        paired[t + "_seed"] = d["seed"].cpu().numpy()
    if split == "val":
        (OUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    (OUT / f"prediction-{split}.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / f"paired-{split}.npz", **paired)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["val", "test"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    evaluate(args.split, args.device)
