"""Universal multi-convex (M=2 sub-OBB) rigid-body geometry loader.

Every rigid body in any physics simulator (PyBullet, Pymunk, MuJoCo) is represented as a union
of up to M=2 body-local convex bounding boxes [min_x, min_y, min_z, max_x, max_y, max_z]
plus a binary part_mask (B, N, 2). Single-convex bodies activate part 0 only; non-convex
compound bodies (such as the Push-T T-block crossbar + stem) activate parts 0 and 1.
"""
import numpy as np
import torch

from whatif.benchmarks.data import ALL_DOMAINS, NEW_DOMAINS, UNIT
from whatif.geometry.data import load as geo_load

MAX_PARTS = 2


def domain_multiconvex(domain, role_np, geom_np):
    """Return parts (B, N, 2, 6) and part_mask (B, N, 2) strictly from simulator collision primitives."""
    b, n = role_np.shape[:2]
    present = role_np.sum(axis=-1) > 0.5
    parts = np.zeros((b, n, MAX_PARTS, 6), dtype=np.float32)
    part_mask = np.zeros((b, n, MAX_PARTS), dtype=np.float32)

    parts[:, :, 0] = geom_np
    part_mask[:, :, 0] = present.astype(np.float32)

    if domain == 'pusht':
        # Exact Pymunk 2-box convex decomposition of the T-block (node 0) in 100px object units:
        #   shape 0 (crossbar): [-60, 0] x [60, 30]
        #   shape 1 (stem):     [-15, 30] x [15, 120]
        parts[:, 0, 0] = [-0.60, 0.00, -0.15, 0.60, 0.30, 0.15]
        parts[:, 0, 1] = [-0.15, 0.30, -0.15, 0.15, 1.20, 0.15]
        part_mask[:, 0, 0] = 1.0
        part_mask[:, 0, 1] = 1.0

    return parts, part_mask


def load(domain, split, device='cpu', *, labels=True):
    d = geo_load(domain, split, device, labels=labels)
    role_np = d['role'].detach().cpu().numpy()
    geom_np = d['geom'].detach().cpu().numpy()
    parts_np, pmask_np = domain_multiconvex(domain, role_np, geom_np)
    parts = torch.as_tensor(parts_np, device=device, dtype=d['x'].dtype)
    part_mask = torch.as_tensor(pmask_np, device=device, dtype=d['x'].dtype)
    return dict(d, parts=parts, part_mask=part_mask)
