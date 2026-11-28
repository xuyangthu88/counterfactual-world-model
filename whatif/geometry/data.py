"""Universal rigid-body geometry enrichment (3D body-local [b_min, b_max] + true wall surface normals).

STRICTLY SEPARATE protocol from original 9D benchmarks; both our model and all baselines read these exact tensors.
"""
import math
import numpy as np
import torch

from whatif.benchmarks.data import ALL_DOMAINS, NEW_DOMAINS, UNIT, load as base_load
from whatif.mechanism.native import N

# Exact simulator-derived body-local bounding boxes [min_x, min_y, min_z, max_x, max_y, max_z] in object units.
# Push-T walls also receive their true inward-normal orientation angles [0, +pi/2, -pi/2, pi].
PUSHT_WALL_ANGLES = (0.0, math.pi / 2.0, -math.pi / 2.0, math.pi)


def _planar_rot6(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c, s, 0.0, -s, c, 0.0], dtype=np.float32)


def domain_geometry(domain, role_np):
    """Return (B, N, 6) body-local [b_min, b_max] strictly from physical simulator collision shapes."""
    b, n = role_np.shape[:2]
    present = role_np.sum(axis=-1) > 0.5
    geom = np.zeros((b, n, 6), dtype=np.float32)
    if domain == 'cophy':
        box = np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        geom[present] = box
    elif domain == 'ballscf':
        sph = np.array([-0.25, -0.25, -0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        geom[present] = sph
    elif domain == 'causalworld':
        geom[:, 0] = [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]       # 0.065m cube block
        geom[:, 1:4] = [-0.23, -0.23, -0.23, 0.23, 0.23, 0.23]  # TriFinger tip radius ~0.015m / 0.065
        geom[:, 4] = [-3.0, -3.0, -0.05, 3.0, 3.0, 0.0]      # arena floor surface at z=0
    elif domain == 'stacking2':
        geom[:, 0:2] = [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]     # two 0.065m cube blocks
        geom[:, 2:5] = [-0.23, -0.23, -0.23, 0.23, 0.23, 0.23]  # 3 TriFinger tips
        geom[:, 5] = [-3.0, -3.0, -0.05, 3.0, 3.0, 0.0]      # arena floor
    elif domain == 'pusht':
        # Exact Pymunk local vertices (in 100px units) relative to block.position
        geom[:, 0] = [-0.60, 0.00, -0.15, 0.60, 1.20, 0.15]  # T-block local AABB
        geom[:, 1] = [-0.15, -0.15, -0.15, 0.15, 0.15, 0.15]  # pusher circle r=15px
        geom[:, 2:6] = [-0.02, -2.505, -0.50, 0.02, 2.505, 0.50]  # 4 walls: e1=inward normal, e2=tangent
    else:
        raise ValueError(f'Unknown domain: {domain}')
    geom[~present] = 0.0
    return geom


def load(domain, split, device='cpu', *, labels=True):
    d = base_load(domain, split, device, labels=labels)
    role_np = d['role'].detach().cpu().numpy()
    geom = torch.as_tensor(domain_geometry(domain, role_np), device=device, dtype=d['x'].dtype)
    d = dict(d, geom=geom)

    if domain == 'pusht':
        # Fix native.py:100 omission where all 4 static walls had angle=0 instead of true inward normals.
        rot6 = torch.as_tensor(np.stack([_planar_rot6(ang) for ang in PUSHT_WALL_ANGLES]),
                               device=device, dtype=d['x'].dtype)
        d['x'] = d['x'].clone()
        d['x'][:, 2:6, 3:9] = rot6[None]
        d['fact'] = d['fact'].clone()
        d['fact'][:, :, 2:6, 3:9] = rot6[None, None]
        d['pose'] = d['pose'].clone()
        d['pose'][:, :, :, 2:6, 3:9] = rot6[None, None, None]
        if 'native_pose' in d:
            d['native_pose'] = d['native_pose'].clone()
            d['native_pose'][:, :, :, 2:6, 3:9] = rot6[None, None, None]
    return d
