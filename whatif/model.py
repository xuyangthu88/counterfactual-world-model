"""A shared active-finger graph edge and a macro-step pose decoder."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

POSE_SCALE = np.array([0.06, 0.025, 0.7])


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def pose_diff(a, b):
    delta = a - b
    delta = delta.copy()
    delta[..., 2] = wrap(delta[..., 2])
    return delta


def features(state, action, physics, noise):
    """Inputs broadcast over independent scene, action and particle axes."""
    x, y, yaw, fx, fy, fz = np.moveaxis(state, -1, 0)
    off, travel = np.moveaxis(action, -1, 0)
    mass, mu = np.moveaxis(physics, -1, 0)
    effective_travel = travel + noise
    values = [x/.065, y/.04, np.sin(yaw), np.cos(yaw), (fx-x)/.12,
              (fy-y)/.08, fz/.1, np.log(mass/.075), np.log(mu/.5),
              off/.04, effective_travel/.05]
    return np.stack(np.broadcast_arrays(*values), axis=-1).astype("float32")


def network(dim, outputs):
    return nn.Sequential(nn.Linear(dim, 80), nn.SiLU(), nn.Linear(80, 80),
                         nn.SiLU(), nn.Linear(80, outputs))


class World(nn.Module):
    def __init__(self, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        self.gate = network(11, 1) if use_graph else None
        self.message = network(11, 3)
        self.free = network(9, 3)

    def forward(self, x):
        msg = self.message(x)
        if self.use_graph:
            msg = self.gate(x).sigmoid() * msg
        return self.free(x[..., :9]) + msg


def state_pair_loss(pred, target):
    delta = pred - target
    # Periodic yaw, including differences between branch contrasts.
    yaw = delta[..., 2] * POSE_SCALE[2]
    delta = torch.cat([delta[..., :2],
                       (torch.atan2(yaw.sin(), yaw.cos()) / POSE_SCALE[2]).unsqueeze(-1)], dim=-1)
    return F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=0.24)


def train_model(data, ids, valid, seed, use_graph, device):
    torch.manual_seed(seed)
    model = World(use_graph).to(device)
    x = features(data["states"][:, :, None], data["actions"],
                 data["physics"][:, None, None], data["noise"][:, :, None])
    y = pose_diff(data["ends"], data["states"][:, :, None, :3]) / POSE_SCALE
    x, y, contact = [torch.tensor(arr, dtype=torch.float32, device=device)
                     for arr in (x, y, data["contact"])]
    if use_graph:
        optimizer = torch.optim.AdamW(model.gate.parameters(), lr=0.003)
        for epoch in range(45):
            for batch in torch.as_tensor(ids, device=device)[torch.randperm(len(ids), device=device)].split(32):
                logits = model.gate(x[batch]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, contact[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.gate.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.003)
    best = math.inf
    for epoch in range(150):
        for batch in torch.as_tensor(ids, device=device)[torch.randperm(len(ids), device=device)].split(32):
            pred = model(x[batch])
            target = y[batch]
            # Each pair shares snapshot, physical particle and transient disturbance.
            other = torch.randperm(pred.shape[-2], device=device)
            loss = state_pair_loss(pred, target) + state_pair_loss(
                pred - pred[:, :, other], target - target[:, :, other])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch in (89, 119):
            for group in optimizer.param_groups:
                group["lr"] *= .45
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                pred = model(x[valid])
                target = y[valid]
                score = (state_pair_loss(pred, target) + state_pair_loss(
                    pred[:, :, 1:] - pred[:, :, :1], target[:, :, 1:] - target[:, :, :1])).item()
            if score < best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), best
