"""Same new factual and phase-labelled layouts for the three model adaptations."""

import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
from whatif.factual.model import AdaptedModels, evidence, cases, selected
from whatif.factual.train import cophy_objective, relative
from whatif.crossbench.models import mlp, wrap
from whatif.shared.model import body_delta, rotate, represent, pets_targets
from .collect import OUT
from .evaluate import panel
from .model import PhaseGraph


def smooth(x, y):
    return F.smooth_l1_loss(x, y.expand_as(x), beta=0.035)


def objective(key, m, d, aux):
    s, a, y = d["state"], d["cmd"], d["end"]
    b, n = a.shape[:2]
    st = s[:, None].expand_as(y)
    phase = d.get("phase")
    target = body_delta(st, y)
    ctx = evidence(d["previous"], d["fact"], s)
    if key == "graph":
        k = s.new_tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).repeat_interleave(3).expand(b, n, 15)
        u = s.new_tensor([-1.0, 0.0, 1.0]).repeat(5).expand_as(k)
        z, e, pk, rank, (approach, stroke, speed) = m.graph.transition(
            st[:, :, None].expand(-1, -1, 15, -1), a[:, :, None].expand(-1, -1, 15, -1), k, u
        )
        w = m.graph.belief(ctx).repeat_interleave(3, -1)[:, None, :, None] / 3
        mean = lambda x: (w * x).sum(2)
        pred = body_delta(st[:, :, None], z)
        effect = relative(z, d["ref"])
        real = relative(y, d["ref"])
        residual = effect - real[:, :, None]
        err = torch.cat((residual[..., :2], 0.47 * wrap(residual[..., 2:3])), -1)
        spread = effect.unsqueeze(-2) - effect.unsqueeze(-3)
        diff = torch.cat((spread[..., :2], 0.47 * wrap(spread[..., 2:3])), -1).norm(dim=-1)
        energy = (w[..., 0] * err.norm(dim=-1)).sum(-1) - 0.5 * (
            w * w[..., 0].unsqueeze(2) * diff
        ).sum((-2, -1))
        loss = (
            energy.mean()
            + 1.2 * smooth(mean(pred), target)
            + 0.5 * smooth(mean(z[..., 14:17]), y[..., 14:17])
            + 0.25 * smooth(mean(z[..., 3:9]), y[..., 3:9])
        )
        if phase is not None:
            h1, h2 = phase[:, :, 0], phase[:, :, 1]
            vel = torch.cat((rotate(h2[..., 14:16], -st[..., 2]), h2[..., 16:17]), -1)
            loss = (
                loss
                + 0.55 * smooth(mean(approach), body_delta(st, h1))
                + 0.7 * smooth(mean(stroke), body_delta(st, h2))
                + 0.7 * smooth(mean(speed), vel)
            )
        return loss
    if key == "cophy":
        co, _ = m.cophy(d["previous"], s, d["fact"], s, a)
        loss = cophy_objective(m.cophy, d)
        delta = body_delta(st, co)
    else:
        x = torch.cat((represent(st, a), ctx[:, None].expand(-1, n, -1)), -1).reshape(-1, 92)
        x = (x - m.xm) / m.xs
        yy = (pets_targets(st, y, d["peak"]).reshape(-1, 16) - m.ym) / m.ys
        # Bootstrap resampling over entire layouts within each ensemble member.
        if torch.is_grad_enabled():
            ids = torch.randint(b, (5, b), device=s.device)
            ix = (ids[..., None] * n + torch.arange(n, device=s.device)).flatten(1, 2)
            loss = m.pets.loss(x[ix], yy[ix])[0]
        else:
            loss = m.pets.loss(x[None].expand(5, -1, -1), yy[None].expand(5, -1, -1))[0]
        mu, _ = m.pets(x, use_propagation=False)
        dy = (mu.mean(0) * m.ys + m.ym).reshape(b, n, 16)
        delta = body_delta(
            st, torch.cat((st[..., :2] + dy[..., :2], st[..., 2:3] + dy[..., 2:3]), -1)
        )
        loss = loss + 0.3 * smooth(delta, target)
    if phase is not None:
        p = aux(torch.cat((represent(st, a), delta), -1))
        intermediate = torch.cat([body_delta(st, phase[:, :, j]) for j in (0, 1)], -1)
        loss = loss + 0.45 * smooth(p, intermediate)
    return loss


def run(name, device):
    torch.manual_seed(11)
    torch.set_num_threads(2)
    m = AdaptedModels(device)
    aux = mlp(43, 6, 64).to(device)
    if name == "graph":
        net = PhaseGraph().to(device)
        net.load_state_dict(m.graph.state_dict(), strict=False)
        for p in net.parameters():
            p.requires_grad_(False)
        for module in (net.phases, net.encoder):
            for p in module.parameters():
                p.requires_grad_(True)
        m.graph = net
    model = getattr(m, name).train()
    trains = [panel(t, "train", device) for t in ("pusht", "near", "causalworld")] + [
        cases("wall", "train", device)
    ]
    valid = [panel(t, "val", device) for t in ("pusht", "near", "causalworld", "wall")]
    rate = {"graph": 0.0008, "cophy": 0.0003, "pets": 0.00030}[name]
    opt = torch.optim.AdamW(list(model.parameters()) + list(aux.parameters()), lr=rate)
    hist = []
    best = np.inf

    # Checkpoint selection uses prospective action effects, with a fixed scaled metric per domain.
    @torch.no_grad()
    def evaluate():
        from .evaluate import particles
        from whatif.shared.evaluate import contrast

        vals = []
        for data in valid:
            xy = []
            for ix in torch.arange(len(data["ref"]), device=device).split(24):
                d = selected(data, ix)
                p = particles(m, d)[0][name]
                z, w = p[:2]
                r = d["ref"]
                i = torch.arange(len(r), device=device)
                # Scale here only for validation aggregation: effect+absolute errors in object units.
                eff = contrast(z, z[i, r, None])
                y = d["end"]
                truth = contrast(y, y[i, r, None])
                delta = (eff[..., :2] * w[:, None, :, None]).sum(2) - truth[..., :2]
                absolute = (z[..., :2] * w[:, None, :, None]).sum(2) - y[..., :2]
                yaw = wrap(
                    torch.atan2(
                        (eff[..., 2].sin() * w[:, None]).sum(2),
                        (eff[..., 2].cos() * w[:, None]).sum(2),
                    )
                    - truth[..., 2]
                ).abs()
                xy.append(
                    float((delta.norm(dim=-1) + 0.5 * absolute.norm(dim=-1) + 0.22 * yaw).mean())
                    * len(r)
                )
            vals.append(sum(xy) / len(data["ref"]))
        return vals, float(np.dot(vals, [0.50, 0.22, 0.24, 0.04]))

    for epoch in range(31):
        if epoch:
            for step in range(28):
                # Keep wall contacts and both ordinary domains in every training epoch.
                t = int(torch.multinomial(torch.tensor([0.44, 0.22, 0.25, 0.09]), 1))
                d = trains[t]
                ix = torch.randint(len(d["ref"]), (32,), device=device)
                loss = objective(name, m, selected(d, ix), aux)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
                opt.step()
        if epoch in (19, 26):
            for g in opt.param_groups:
                g["lr"] *= 0.45
        if epoch % 3 == 0:
            vals, score = evaluate()
            hist.append([epoch, score, vals])
            if score < best:
                best = score
                torch.save(model.state_dict(), OUT / f"{name}.pt")
            print(name, epoch, round(score, 5), flush=True)
    (OUT / f"{name}-fit.json").write_text(json.dumps(hist, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=["graph", "pets", "cophy"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run(a.name, a.device)
