"""Learn long-horizon paired action returns from matched native feedback continuations."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from whatif.crossbench.models import geometry_cover, wrap, mlp
from whatif.shared.model import represent, body_delta, task_value, unpack
from whatif.shared.planning import menus, mechanical, scale, weighted_state
from whatif.factual.model import AdaptedModels, evidence
from whatif.precision.model import install as install_precision
from .model import OUT, install, passage_grid, native_cover, physical_history
from .duration import duration

POLICIES = {"light": 0.30, "balanced": 0.60, "strong": 0.90}


def models(method, device):
    m = AdaptedModels(device)
    if method == "graph":
        install(m, device)
    else:
        install_precision(m, device)
    return m


def outcomes(m, method, s, a, ctx, prev, last):
    b, n = a.shape[:2]
    st = s[:, None].expand(-1, n, -1)
    if method == "graph":
        z, e, peak, rank, p = passage_grid(m.graph, s, a, ctx)
        w = m.graph.belief(ctx).repeat_interleave(3, -1) / 3
        edges = (e.sigmoid() * w[:, None, :, None]).sum(2)
    elif method == "pets":
        mu, sd = m.predict(
            st.reshape(-1, 19), a.reshape(-1, 7), ctx[:, None, :].expand(-1, n, -1).reshape(-1, 52)
        )
        gen = torch.Generator(device=s.device).manual_seed(11)
        eps = (
            torch.randn(5, 1, 1, 1, 16, generator=gen, device=s.device)
            * s.new_tensor([-1.224745, 0.0, 1.224745])[None, None, None, :, None]
        )
        dy = (
            (mu.reshape(5, b, n, 1, 16) + sd.reshape(5, b, n, 1, 16) * eps)
            .permute(1, 2, 0, 3, 4)
            .reshape(b, n, 15, 16)
        )
        z, peak = unpack(st[:, :, None].expand(-1, -1, 15, -1), dy)
        w = s.new_full((b, 15), 1 / 15)
        rank = task_value(st[:, :, None], z, peak)
        p = (peak > (0.9 + 0.05 * st[:, :, None, 13])).float()
        edges = s.new_zeros(b, n, 3)
    else:
        co, _ = m.cophy(prev, s, last, s, a)
        z = st[:, :, None].clone()
        z[..., :3] = co[:, :, None]
        z[..., 12] = native_cover(z[..., :3], z)
        peak = z[..., 12]
        p = (peak > (0.9 + 0.05 * st[:, :, None, 13])).float()
        w = s.new_ones(b, 1)
        rank = task_value(st[:, :, None], z, peak)
        edges = s.new_zeros(b, n, 3)
    return z, w, peak, rank, p, edges


def features(m, method, s, a, ctx, prev, last, remaining, task):
    b, n = a.shape[:2]
    st = s[:, None].expand(-1, n, -1)
    z, w, peak, rank, p, edge = outcomes(m, method, s, a, ctx, prev, last)
    zmean = weighted_state(z, w)
    ww = w[:, None, :, None]
    d = body_delta(st[:, :, None], z)
    d0 = (d * ww).sum(2)
    spread = ((d - d0[:, :, None]).square() * ww).sum(2).clamp_min(1e-9).sqrt()
    # The nominal action is index 1 in the native menu for every method.
    paired = d - d[:, 1:2]
    effect = (paired * ww).sum(2)
    effect_spread = ((paired - effect[:, :, None]).square() * ww).sum(2).clamp_min(1e-9).sqrt()
    goal = body_delta(z, z[..., 9:12])
    gmean = (goal * ww).sum(2)
    c = native_cover(z[..., :3], z)
    v = task_value(st[:, :, None], z, peak)
    quantities = torch.stack((c, peak, rank, p, v), -1)
    qs = (quantities * ww).sum(2)
    qs_sd = ((quantities - qs[:, :, None]).square() * ww).sum(2).clamp_min(1e-9).sqrt()
    rep = represent(st, a)
    g0 = body_delta(st, st[..., 9:12])
    next_actions = menus(zmean, task)
    next_values = mechanical(zmean[:, :, None], next_actions)
    follow = next_values.topk(2, -1).values
    horizon = 420 - 140 * st[..., 13]
    clock = duration(m, st, a) / horizon
    time = remaining[:, None].expand(-1, n) / horizon
    obs = torch.cat(
        (
            rep[..., :4],
            g0,
            rep[..., 8:22],
            rep[..., 22:40],
            physical_history(ctx)[:, None].expand(-1, n, -1),
            time[..., None],
            clock[..., None],
        ),
        -1,
    )
    prediction = torch.cat(
        (
            d0,
            spread,
            effect,
            effect_spread,
            gmean,
            qs,
            qs_sd,
            edge,
            zmean[..., 14:19],
            follow,
            mechanical(st, a)[..., None],
        ),
        -1,
    )
    return torch.cat((obs, prediction), -1)


class ReturnHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.register_buffer("location", torch.zeros(d))
        self.register_buffer("scale", torch.ones(d))
        self.encoder = mlp(2 * d, 80, 96)
        self.value = mlp(160, 3, 64)

    def forward(self, x):
        norm = ((x - self.location) / self.scale).clamp(-8, 8)
        h = self.encoder(torch.cat((norm, norm - norm.mean(1, keepdim=True)), -1))
        return self.value(torch.cat((h, h.mean(1, keepdim=True).expand_as(h)), -1))


def return_value(y):
    return y[..., 0] + 0.45 * y[..., 1] - 0.12 * y[..., 2]


def scores(head, x):
    y = head(x).sigmoid()
    return return_value(y)


def objective(head, x, y):
    raw = head(x)
    components = raw.sigmoid()
    q = return_value(components)
    target = return_value(y)
    pair = q[:, :, None] - q[:, None, :]
    delta = target[:, :, None] - target[:, None, :]
    anchor = F.binary_cross_entropy_with_logits(raw[..., 0], y[..., 0]) + F.mse_loss(
        components[..., 1:], y[..., 1:]
    )
    rank = F.binary_cross_entropy_with_logits(pair / 0.22, (delta / 0.22).sigmoid())
    centered = F.mse_loss(q - q.mean(1, keepdim=True), target - target.mean(1, keepdim=True))
    return 0.45 * anchor + rank + 0.65 * centered


@torch.no_grad()
def prepare(m, method, task, split, device):
    path = Path("data/passage") / f"return-{task}-{split}.npz"
    d = np.load(path)
    xs = []
    tensor = lambda k: torch.as_tensor(d[k], device=device, dtype=torch.float32)
    s = tensor("state")
    a = tensor("cmd")
    prev = tensor("previous")
    last = tensor("fact")
    rem = tensor("remaining")
    ctx = evidence(prev, last, s)
    for ids in torch.arange(len(s), device=device).split(24):
        xs.append(
            features(m, method, s[ids], a[ids], ctx[ids], prev[ids], last[ids], rem[ids], task)
        )
    y = torch.stack(
        (tensor("success"), tensor("coverage"), tensor("steps") / rem[:, None].clamp_min(1)), -1
    )
    return torch.cat(xs), y


def evaluate(head, valid):
    records = []
    with torch.no_grad():
        for x, y in valid:
            q = scores(head, x)
            target = return_value(y)
            idx = q.argmax(-1, keepdim=True)
            loss = target.amax(1) - target.gather(1, idx).squeeze(-1)
            pair = (q[:, :, None] - q[:, None, :]) * (target[:, :, None] - target[:, None, :])
            non_ties = (target[:, :, None] - target[:, None, :]).abs() > 0.04
            top3 = q.topk(3, dim=-1).indices
            hit = y[..., 0].gather(1, idx).mean()
            records.append(
                dict(
                    regret=float(loss.mean()),
                    top3_regret=float((target.amax(1) - target.gather(1, top3).mean(1)).mean()),
                    success=float(hit),
                    rank_accuracy=float((pair[non_ties] > 0).float().mean()),
                )
            )
    return records


def train(method, device):
    torch.manual_seed(11)
    torch.set_num_threads(2)
    m = models(method, device)
    tasks = ("pusht", "causalworld")
    train = [prepare(m, method, t, "train", device) for t in tasks]
    val = [prepare(m, method, t, "val", device) for t in tasks]
    d = train[0][0].shape[-1]
    head = ReturnHead(d).to(device)
    flat = torch.cat([x.flatten(0, 1) for x, _ in train])
    head.location.copy_(flat.mean(0))
    head.scale.copy_(flat.std(0).clamp_min(0.02))
    opt = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.002)
    best = np.inf
    history = []
    for ep in range(101):
        for _ in range(18):
            t = int(torch.randint(0, 2, ()).item())
            x, y = train[t]
            ids = torch.randint(len(x), (64,), device=device)
            loss = objective(head, x[ids], y[ids])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 3.0)
            opt.step()
        if ep in (45, 76):
            for group in opt.param_groups:
                group["lr"] *= 0.45
        if ep % 4 == 0 or ep == 100:
            v = evaluate(head, val)
            objective_val = float(np.mean([r["regret"] + 0.5 * r["top3_regret"] for r in v]))
            history.append(dict(epoch=ep + 1, validation=v, score=objective_val))
            if objective_val < best:
                best = objective_val
                torch.save(dict(d=d, state=head.state_dict()), OUT / f"advantage-{method}.pt")
            print(method, ep + 1, round(objective_val, 4), flush=True)
    (OUT / f"return-fit-{method}.json").write_text(json.dumps(history, indent=2) + "\n")


def load_head(method, device):
    ck = torch.load(OUT / f"advantage-{method}.pt", map_location=device, weights_only=True)
    head = ReturnHead(ck["d"]).to(device)
    head.load_state_dict(ck["state"])
    return head.eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=["graph", "pets", "cophy"])
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    train(a.method, a.device)
