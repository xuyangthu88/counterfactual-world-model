"""Fit graph response and a common peak-coverage distribution to native stroke recordings."""

import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
from whatif.shared.model import body_delta
from whatif.crossbench.models import wrap
from whatif.factual.model import AdaptedModels, evidence, cases
from whatif.precision.evaluate import panel
from whatif.precision.model import install as precision
from .model import PassageGraph, physical_history, OUT


@torch.no_grad()
def prepare(net, d):
    s = d["state"]
    a = d["cmd"]
    z = d["end"]
    n = a.shape[1]
    pair = torch.cartesian_prod(
        s.new_tensor([-1.0, -0.5, 0.0, 0.5, 1.0]), s.new_tensor([-1.0, 0.0, 1.0])
    )
    hist = evidence(d["previous"], d["fact"], s)
    features = []
    for ids in torch.arange(len(s), device=s.device).split(32):
        sz = (len(ids), n, 15)
        features.append(
            net.extract(
                s[ids, None, None].expand(*sz, 19),
                a[ids, :, None].expand(*sz, 7),
                pair[:, 0].expand(sz),
                pair[:, 1].expand(sz),
            )[-1]
        )
    phases = d.get("phase")
    if phases is None:
        hands = body_delta(s[:, None], z).unsqueeze(-2).expand(-1, -1, 3, -1)
    else:
        hands = body_delta(s[:, None, None], phases)
    return dict(
        s=s,
        local=torch.cat(features),
        context=hist,
        hand=hands,
        peak=d["peak"],
        phase_valid=s.new_full((len(s),), float(phases is not None)),
        ref=d["ref"],
    )


def loss(net, d, ix, validation=False):
    s = d["s"][ix]
    local = d["local"][ix]
    ev = d["context"][ix]
    st = s[:, None, None].expand(*local.shape[:-1], 19)
    history = physical_history(ev)[:, None, None]
    mass = net.belief(ev).repeat_interleave(3, -1) / 3
    w = mass[:, None, :]
    hands = net.decode(local, history)
    factor = s.new_tensor([1.0, 1.0, 0.47])
    target = d["hand"][ix]
    pred = hands[-1]
    # Direct paired energy score on body displacement and wrapped yaw in a common start frame.
    i = torch.arange(len(ix), device=ix.device)
    ref = d["ref"][ix]
    effect = pred - pred[i, ref, None]
    obs = target[:, :, -1] - target[i, ref, None, -1]
    diff = effect - obs[:, :, None]
    err = torch.cat((diff[..., :2], wrap(diff[..., 2:3])), -1) * factor
    pair = effect[:, :, :, None] - effect[:, :, None, :]
    pair = torch.cat((pair[..., :2], wrap(pair[..., 2:3])), -1) * factor
    es = (err.norm(dim=-1) * w).sum(-1) - 0.5 * (
        pair.norm(dim=-1) * w[:, :, :, None] * w[:, :, None, :]
    ).sum((-1, -2))
    mean = lambda x: (x * w[..., None]).sum(2)
    pose = (mean(pred) - target[:, :, -1]) * factor
    phase = torch.stack([mean(h) for h in hands[:2]], 2) - target[:, :, :2]
    helper = F.smooth_l1_loss(
        phase * factor, torch.zeros_like(phase), beta=0.025, reduction="none"
    ).mean((1, 2, 3))
    _, loc, width, prob = net.read_passage(st, local, history, hands)
    pk = d["peak"][ix]
    # Continuous peak regression and a proper binary score at multiple coverage levels.
    levels = s.new_tensor([0.55, 0.7, 0.8, 0.86, 0.9, 0.925, 0.95, 0.97])
    logits = (loc[..., None] - levels) / width[..., None]
    cdf = (logits.sigmoid() * w[..., None]).sum(2)
    labels = (pk[..., None] > levels).float()
    mask = (s[:, None, None, 12] < levels).float()
    likelihood = F.binary_cross_entropy(cdf.clamp(1e-6, 1 - 1e-6), labels, reduction="none")
    coverage = F.smooth_l1_loss((loc * w).sum(2), pk, beta=0.02) + 0.12 * (mask * likelihood).mean()
    positions = (
        es.mean()
        + 1.2 * F.smooth_l1_loss(pose, torch.zeros_like(pose), beta=0.025)
        + 0.65 * (helper * d["phase_valid"][ix]).mean()
    )
    if validation:
        return [float(es.mean()), float(pose.norm(dim=-1).mean()), float(coverage)]
    return positions + 0.7 * coverage


def run(device):
    torch.manual_seed(11)
    torch.set_num_threads(2)
    m = AdaptedModels(device)
    precision(m, device, ("graph",))
    net = PassageGraph().to(device)
    net.load_state_dict(m.graph.state_dict(), strict=False)
    for p in net.parameters():
        p.requires_grad_(False)
    for mod in (net.response, net.encoder, net.passage):
        for p in mod.parameters():
            p.requires_grad_(True)
    tasks = ("pusht", "near", "causalworld", "wall")

    def get(task, split):
        if task == "wall":
            return cases("wall", split, device)
        return panel(task, split, device)

    train = [prepare(net, get(t, "train")) for t in tasks]
    valid = [prepare(net, get(t, "val")) for t in tasks]
    with torch.no_grad():
        feats = []
        for d in train:
            l = d["local"].flatten(0, 2)
            h = (
                physical_history(d["context"])[:, None, None]
                .expand(*d["local"].shape[:-1], 11)
                .flatten(0, 2)
            )
            ids = torch.randperm(len(l), device=device)[:18000]
            feats.append(torch.cat((l[ids], h[ids]), -1))
        f = torch.cat(feats)
        net.location.copy_(f.mean(0))
        net.scale.copy_(f.std(0).clamp_min(0.03))
    opt = torch.optim.AdamW(
        [
            {"params": net.response.parameters(), "lr": 8e-4},
            {"params": net.passage.parameters(), "lr": 1.2e-3},
            {"params": net.encoder.parameters(), "lr": 1.5e-4},
        ]
    )
    best = np.inf
    history = []
    for epoch in range(61):
        if epoch:
            for batch in range(36):
                t = int(torch.multinomial(torch.tensor([0.46, 0.22, 0.24, 0.08]), 1))
                d = train[t]
                ix = torch.randint(len(d["s"]), (32,), device=device)
                v = loss(net, d, ix)
                opt.zero_grad()
                v.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
                opt.step()
        if epoch in (32, 48):
            for group in opt.param_groups:
                group["lr"] *= 0.45
        if epoch % 4 == 0 or epoch == 60:
            with torch.no_grad():
                vals = []
                for d in valid:
                    vals.append(
                        np.average(
                            [
                                loss(net, d, ix, True)
                                for ix in torch.arange(len(d["s"]), device=device).split(24)
                            ],
                            axis=0,
                            weights=[len(ix) for ix in torch.arange(len(d["s"])).split(24)],
                        ).tolist()
                    )
                score = float(
                    np.array([0.5, 0.20, 0.25, 0.05]) @ np.array(vals) @ np.array([1.0, 0.35, 0.75])
                )
                history.append(dict(epoch=epoch, score=score, values=vals))
                if score < best:
                    best = score
                    torch.save(net.state_dict(), OUT / "graph.pt")
                print(epoch, round(score, 5), np.array(vals)[:, :2].round(4).tolist(), flush=True)
    (OUT / "fit.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run(a.device)
