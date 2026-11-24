"""Factual residual transport and a fact-conditioned readout of graph particles."""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .chain_eval import bank, calibrate, predict, weights
from .chain_model import Branches, ContactTransfer, LENGTH

DATA = Path('data/pilot')
OUT = Path('results/pilot')
CHAINS = Path('data/chains')
SEED = 11


def load_data(name, device):
    path = CHAINS if name in ('train', 'val') else DATA
    return Branches(path / (name + '.npz'), device)


def graph_model(phase, family, device):
    path = (Path('results/chains') / f'{family}-11.pt' if phase == 'zero'
            else OUT / f'{phase}-{family}.pt')
    model = ContactTransfer(family == 'graph').to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


def motion_units(x):
    return x / x.new_tensor([LENGTH, LENGTH, 1.])


def index_set(d, time):
    i = torch.arange(len(d.ids), device=d.state.device)
    target = d.target
    s = d.state[:, time]
    # Closest upstream object, measured along the active push direction.
    separation = s[..., 0] - s[i, target, 0, None]
    allowed = d.mask & (separation > 1e-5)
    parent = separation.masked_fill(~allowed, torch.inf).argmin(dim=1)
    parent = torch.where(allowed.any(dim=1), parent, target)
    return i, target, parent


def geometry(d, time):
    i, j, p = index_set(d, time)
    s = d.state[:, time]
    target = s[i, j]
    def object_info(x):
        return torch.cat(((x[:, :2] - target[:, :2]) / LENGTH,
                          x[:, 2:3].sin(), x[:, 2:3].cos(), x[:, 3:5] / LENGTH,
                          x[:, 5:6] / 10), dim=-1)
    # Each panel supplies the commanded action in the canonical push frame.
    state = torch.cat((target[:, :2] / LENGTH, object_info(target),
                       object_info(s[:, 0]), object_info(s[i, p]),
                       (d.tip[:, time, :2] - target[:, :2]) / LENGTH,
                       d.tip[:, time, 2:3] / .1, (p != j)[:, None],
                       (j != 0)[:, None]), dim=-1)
    a = d.actions / LENGTH
    return torch.cat((state[:, None].expand(-1, len(a), -1),
                      a[None].expand(len(i), -1, -1)), dim=-1)


class Readout(nn.Module):
    def __init__(self, width, residual):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 128),
                                 nn.SiLU(), nn.Linear(128, 2))
        if residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


@dataclass
class Panel:
    features: torch.Tensor
    base: torch.Tensor    # target XY particle displacements, units of cube length
    weight: torch.Tensor
    truth: torch.Tensor
    goal: torch.Tensor    # relative to the current target position, units of cube length
    anchor: int          # centre the learned correction for same-snapshot hindsight
    reference: int       # executed factual command


def simple_panel(d, time):
    i, j, _ = index_set(d, time)
    truth = d.delta[i, time, :, j, :2]
    goal = (d.goal - d.state[i, time, j, :2]) / LENGTH
    return Panel(geometry(d, time), torch.zeros_like(truth[:, :, None]),
                 truth.new_ones(len(i), 1), truth, goal, -1, d.probe)


@torch.no_grad()
def graph_panels(model, d, sigma, second=True):
    p0 = predict(model, d, 0)
    q = bank(d)
    result = []
    for time, ref in ([(0, 1), (1, d.probe)] if second else [(0, 1)]):
        i, j, parent = index_set(d, time)
        factual = d.delta[:, 0, ref] * d.scale
        post = weights(p0[:, ref], factual, sigma[ref], d.mask)
        w = post if time == 0 else post.reshape(-1, 9, 3).sum(-1).repeat_interleave(3, -1) / 3
        pred = p0 if time == 0 else predict(model, d, 1)
        if time == 0:
            pred = pred - p0[:, ref:ref+1] + factual[:, None, None]
        distribution = motion_units(pred)
        mean = (distribution * w[:, None, :, None, None]).sum(dim=2)
        std = ((distribution - mean[:, :, None]).square() * w[:, None, :, None, None]).sum(dim=2).sqrt()
        base = distribution[i, :, :, j, :2]
        obs = motion_units(factual)
        error = obs - motion_units((p0[:, ref] * post[:, :, None, None]).sum(dim=1))
        def broadcast(feature):
            return feature[:, None].expand(-1, len(d.actions), -1)
        physics = (post @ q) / q.new_tensor([.075, .5, .0025])
        extra = torch.cat((obs[i, j], obs[i, parent], obs[:, 0], error[i, j],
                           error[i, parent], error[:, 0], physics,
                           d.actions[ref].expand(len(i), -1) / LENGTH,
                           torch.full_like(physics[:, :1], time)), dim=-1)
        features = torch.cat((geometry(d, time), mean[i, :, j], mean[i, :, parent],
                              mean[:, :, 0], std[i, :, j], broadcast(extra)), dim=-1)
        goal = (d.goal - d.state[i, time, j, :2]) / LENGTH
        result.append(Panel(features, base, w, d.delta[i, time, :, j, :2],
                            goal, ref if time == 0 else -1, ref))
    return result


def read(panel, head, index):
    delta = head(panel.features[index])
    if panel.anchor >= 0:
        delta = delta - delta[:, panel.anchor:panel.anchor+1]
    return panel.base[index] + delta[:, :, None]


def fit_head(train, valid, residual, device, path, epochs=100):
    torch.manual_seed(SEED)
    head = Readout(train[0].features.shape[-1], residual).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=.0015)
    best = float('inf')
    def objective(panel, indices):
        particles = read(panel, head, indices)
        w = panel.weight[indices, None]
        mean = (particles * w[..., None]).sum(dim=2)
        truth = panel.truth[indices]
        if not residual:
            return F.mse_loss(mean, truth)
        pose = F.smooth_l1_loss(mean, truth, beta=.06)
        # Pair actions at the same goal. This calibrates the predicted advantage
        # in distance units and preserves each original particle's relative offset.
        target = panel.goal[indices, None]
        predicted_cost = ((particles - target[:, :, None]).norm(dim=-1) * w).sum(dim=2)
        actual_cost = (truth - target).norm(dim=-1)
        cost_error = predicted_cost - actual_cost
        advantage_error = cost_error[:, :, None] - cost_error[:, None, :]
        return pose + .7 * F.smooth_l1_loss(advantage_error, torch.zeros_like(advantage_error), beta=.05)
    for epoch in range(epochs):
        batches = [(p, b) for p in train for b in
                   torch.randperm(len(p.truth), device=device).split(48)]
        for k in torch.randperm(len(batches)).tolist():
            panel, indices = batches[k]
            loss = objective(panel, indices)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if epoch in (60, 85):
            for g in optim.param_groups:
                g['lr'] *= .4
        if (epoch+1) % 4 == 0:
            with torch.no_grad():
                scores = []
                for p in valid:
                    k = torch.arange(len(p.truth), device=device)
                    if residual and p.anchor < 0:
                        cost = ((read(p, head, k) - p.goal[:, None, None]).norm(dim=-1) *
                                p.weight[:, None]).sum(dim=2)
                        scores.append((p.truth[k, cost.argmin(dim=1)] - p.goal).norm(dim=-1).mean().item())
                    elif not residual:
                        scores.append(objective(p, k).item())
                score = sum(scores)/len(scores)
            if score < best:
                best = score
                torch.save(head.state_dict(), path)
    head.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    return head.eval()


def load_head(features, residual, path):
    head = Readout(features.shape[-1], residual).to(features.device)
    head.load_state_dict(torch.load(path, map_location=features.device, weights_only=True))
    return head.eval()
