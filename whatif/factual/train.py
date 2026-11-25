import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
from whatif.shared.model import (
    panel,
    grid,
    CoPhy,
    Null,
    represent,
    pets_targets,
    body_delta,
    task_value,
    wrap,
)
from whatif.shared.train import losses, huber
from whatif.shared.evaluate import contrast
from whatif.contact_switch.train import contact_terms
from .collect import OUT, DATA
from .model import FactualGraph, evidence, cases, selected


def relative(x, ref):
    i = torch.arange(len(x), device=x.device)
    return contrast(x, x[i, ref, None])


def graph_objective(model, d, switch):
    s, a, y = d["state"], d["cmd"], d["end"]
    w = model.belief(evidence(d["previous"], d["fact"], s)).repeat_interleave(3, -1) / 3
    z, edges, peak, rank = grid(model, s, a)
    v = w[:, None, :]
    pred = relative(z, d["ref"])
    truth = relative(y, d["ref"])
    # XY in body-scale units; yaw converted to an equivalent lever-arm displacement.
    scale = s.new_tensor([1.0, 1.0, 0.47])
    err = pred - truth[:, :, None]
    err = torch.cat((err[..., :2], wrap(err[..., 2:3])), -1) * scale
    es = (v * err.norm(dim=-1)).sum(-1)
    xy = pred * scale
    sep = xy[:, :, None] - xy[:, :, :, None]
    sep = torch.cat((sep[..., :2], 0.47 * wrap(sep[..., 2:3] / 0.47)), -1).norm(dim=-1)
    es = es - 0.5 * (sep * v[:, :, :, None] * v[:, :, None, :]).sum((-1, -2))
    pose_error = body_delta(s[:, None, None], z) - body_delta(s[:, None], y)[:, :, None]
    pose_error = torch.cat((pose_error[..., :2], wrap(pose_error[..., 2:3])), -1) * scale
    pose_loss = huber((pose_error * v[..., None]).sum(2), torch.zeros_like(truth))
    value = 0.55 * task_value(s[:, None, None], z, peak) + 0.45 * rank
    value = (value * v).sum(-1)
    true = task_value(s[:, None], y, d["peak"])
    advantage = lambda x: x - x.mean(1, keepdim=True)
    order = (
        huber(advantage(value), advantage(true))
        + 0.12 * (-(true / 0.12).softmax(-1) * (value / 0.12).log_softmax(-1)).sum(1).mean()
    )
    edge = (edges.sigmoid() * v[..., None]).sum(2).clamp(0.00001, 0.99999)
    ce = F.binary_cross_entropy(edge, d["edge"].float())
    pos, rank_edge = contact_terms(edge.logit(), d["edge"])
    terminal = (z * v[..., None]).sum(2)
    aux = huber(terminal[..., 3:9], y[..., 3:9]) + 0.6 * huber(terminal[..., 14:19], y[..., 14:19])
    aux += F.mse_loss(terminal[..., 12], y[..., 12]) + F.mse_loss((peak * v).sum(2), d["peak"])
    # Balanced validation score is kept fixed across the two candidate training weights.
    loss = (
        es.mean()
        + 0.65 * pose_loss
        + order
        + 0.20 * ce
        + switch * (pos + 0.55 * rank_edge)
        + 0.40 * aux
    )
    return loss


def cophy_objective(model, d):
    s, a, y = d["state"], d["cmd"], d["end"]
    z, stable = model(d["previous"], s, d["fact"], s, a)
    err = contrast(z, y)
    r = relative(z, d["ref"]) - relative(y, d["ref"])
    return (
        huber(err, torch.zeros_like(err))
        + 0.75 * huber(r[..., :2], torch.zeros_like(r[..., :2]))
        + 0.10
        * F.binary_cross_entropy_with_logits(
            stable, (body_delta(s[:, None], y).norm(dim=-1) < 0.025).float()
        )
    )


def pets_train(train, valid, base, device):
    from mbrl.models import GaussianMLP

    def pack(d, has_fact):
        s = d["state"][:, None].expand_as(d["end"])
        a = d["cmd"]
        context = (
            evidence(d["previous"], d["fact"], d["state"]) if has_fact else s.new_zeros(len(s), 52)
        )
        x = torch.cat((represent(s, a), context[:, None].expand(-1, a.shape[1], -1)), -1)
        return x.flatten(0, 1), pets_targets(s, d["end"], d["peak"]).flatten(0, 1)

    pairs = [pack(base, False)] + [pack(d, True) for d in train]
    x, y = [torch.cat(q) for q in zip(*pairs)]
    xm, xs = x.mean(0), x.std(0).clamp_min(0.025)
    ym, ys = y.mean(0), y.std(0).clamp_min(0.025)
    x = (x - xm) / xs
    y = (y - ym) / ys
    net = GaussianMLP(x.shape[1], 16, device, num_layers=3, ensemble_size=5, hid_size=144)
    vp = [pack(d, True) for d in valid]
    opt = torch.optim.AdamW(net.parameters(), lr=0.0012)
    bags = torch.randint(len(x), (5, len(x)), device=device)
    best = np.inf
    for epoch in range(75):
        for ix in torch.randperm(len(x), device=device).split(768):
            j = bags[:, ix]
            loss = net.loss(x[j], y[j])[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 4 == 2 or epoch == 74:
            with torch.no_grad():
                vals = []
                for xx, yy in vp:
                    mu, _ = net((xx - xm) / xs, use_propagation=False)
                    vals.append(float((mu - (yy - ym) / ys).square().mean()))
            if np.mean(vals) < best:
                best = np.mean(vals)
                torch.save(dict(net=net.state_dict(), xm=xm, xs=xs, ym=ym, ys=ys), OUT / "pets.pt")
    print("pets validation", best, flush=True)


def main(name, device):
    torch.manual_seed(11)
    torch.set_num_threads(2)
    train = [cases(t, "train", device) for t in ("causalworld", "pusht", "wall")]
    val = [cases(t, "val", device) for t in ("causalworld", "pusht", "wall")]
    base = panel("train", device)
    if name == "pets":
        return pets_train(train, val, base, device)
    if name == "ncii":
        with np.load(DATA / "wall-train.npz") as f:
            wall = {
                k: torch.as_tensor(f[k], device=device)
                for k in ["nullpose", "source", "edge", "state", "cmd", "end", "ref"]
            }
        model = Null().to(device)
        model.load_state_dict(torch.load("results/shared/ncii.pt", weights_only=True))
    elif name == "cophy":
        model = CoPhy().to(device)
        model.load_state_dict(torch.load("results/shared/cophy.pt", weights_only=True))
    else:
        model = FactualGraph().to(device)
        model.load_state_dict(
            torch.load("results/shared/graph.pt", weights_only=True), strict=False
        )
    switch = 0.12 if name == "g12" else 0.32
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008 if name == "cophy" else 0.0006)
    best = np.inf
    history = []
    for epoch in range(45 if name != "ncii" else 32):
        for _ in range(22):
            choice = lambda n, bs: torch.randint(n, (bs,), device=device)
            if name == "ncii":
                loss = losses(
                    "ncii", model, wall, choice(len(wall["ref"]), 48), 35
                ) + 0.65 * losses("ncii", model, base, choice(len(base["ref"]), 64), 35)
            else:
                ds = [selected(d, choice(len(d["ref"]), 24)) for d in train]
                if name == "cophy":
                    loss = sum(cophy_objective(model, d) for d in ds) / 3 + 0.22 * losses(
                        "cophy", model, base, choice(len(base["ref"]), 40), 0
                    )
                else:
                    loss = sum(
                        w * graph_objective(model, d, switch) for d, w in zip(ds, [0.4, 0.43, 0.17])
                    ) + 0.30 * losses("graph", model, base, choice(len(base["ref"]), 48), 0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if epoch in (27, 38):
            for g in opt.param_groups:
                g["lr"] *= 0.45
        if epoch % 4 == 3 or epoch == 44:
            with torch.no_grad():
                vals = []
                for d in val:
                    losses_v = []
                    for ix in torch.arange(len(d["ref"]), device=device).split(48):
                        row = selected(d, ix)
                        if name == "ncii":
                            s = row["state"][:, None].expand_as(row["end"])
                            v = F.binary_cross_entropy_with_logits(
                                model.edges(s, row["cmd"]), row["edge"].float()
                            )
                        elif name == "cophy":
                            v = cophy_objective(model, row)
                        else:
                            v = graph_objective(model, row, 0.20)
                        losses_v.append(len(ix) * v.item())
                    vals.append(sum(losses_v) / len(d["ref"]))
                score = float(np.dot([0.4, 0.43, 0.17], vals))
                history.append([epoch + 1, score, vals])
            if score < best:
                best = score
                torch.save(model.state_dict(), OUT / f"{name}.pt")
            print(name, epoch + 1, score, flush=True)
    (OUT / f"{name}-fit.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args.name, args.device)
