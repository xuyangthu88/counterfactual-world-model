"""Exact T overlap: integrate the height clipped to a goal rectangle along each moving edge."""

import torch

# Local coordinates in body units (100 pixels). Both polygons have counter-clockwise winding.
RECT = ((-0.6, 0.0, 0.6, 0.3), (-0.15, 0.3, 0.15, 1.2))


def positive_mean(a, b):
    upper = torch.maximum(a, b)
    lower = torch.minimum(a, b)
    positive_fraction = torch.where(
        lower >= 0, 1.0, (upper.clamp_min(0) / (upper - lower).clamp_min(1e-12)).clamp_max(1)
    )
    return 0.5 * (a.clamp_min(0) + b.clamp_min(0)) * positive_fraction


def torch_overlap(pose, goal):
    offset = pose[..., :2] - goal[..., :2]
    rot = goal[..., 2]
    x = offset[..., 0] * rot.cos() + offset[..., 1] * rot.sin()
    y = offset[..., 1] * rot.cos() - offset[..., 0] * rot.sin()
    angle = pose[..., 2] - rot
    co = angle.cos()[..., None]
    si = angle.sin()[..., None]
    vertices = pose.new_tensor([[(l, b), (r, b), (r, t), (l, t)] for l, b, r, t in RECT]).reshape(
        8, 2
    )
    xx = x[..., None] + co * vertices[:, 0] - si * vertices[:, 1]
    yy = y[..., None] + si * vertices[:, 0] + co * vertices[:, 1]
    nxt = [1, 2, 3, 0, 5, 6, 7, 4]
    x1 = xx[..., None]
    x2 = xx[..., nxt, None]
    y1 = yy[..., None]
    y2 = yy[..., nxt, None]
    bounds = pose.new_tensor(RECT).T
    dx = x2 - x1
    safe = torch.where(dx.abs() > 1e-10, dx, 1.0)
    a = ((bounds[0] - x1) / safe).clamp(0, 1)
    b = ((bounds[2] - x1) / safe).clamp(0, 1)
    start = torch.minimum(a, b)
    end = torch.maximum(a, b)
    lo = y1 + start * (y2 - y1)
    hi = y1 + end * (y2 - y1)
    height = positive_mean(lo - bounds[1], hi - bounds[1]) - positive_mean(
        lo - bounds[3], hi - bounds[3]
    )
    return (-(end - start) * dx * height).sum((-2, -1)).div(0.63).clamp(0, 1)


def overlap(pose, goal):
    if pose.is_cuda and pose.dtype == torch.float32:
        from .overlap_gpu import compute

        return compute(pose, goal)
    return torch_overlap(pose, goal)
