"""Fuse rectangle integration so that particle planning can evaluate many overlaps per step."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pos(a, b):
    hi = tl.maximum(a, b)
    lo = tl.minimum(a, b)
    frac = tl.where(lo >= 0, 1.0, tl.minimum(1.0, tl.maximum(hi, 0.0) / tl.maximum(hi - lo, 1e-12)))
    return 0.5 * (tl.maximum(a, 0.0) + tl.maximum(b, 0.0)) * frac


@triton.jit
def _area(P, G, O, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(P + 3 * i, mask=i < N, other=0) - tl.load(G + 3 * i, mask=i < N, other=0)
    y = tl.load(P + 3 * i + 1, mask=i < N, other=0) - tl.load(G + 3 * i + 1, mask=i < N, other=0)
    g = tl.load(G + 3 * i + 2, mask=i < N, other=0)
    a = tl.load(P + 3 * i + 2, mask=i < N, other=0) - g
    cx = x * tl.cos(g) + y * tl.sin(g)
    cy = y * tl.cos(g) - x * tl.sin(g)
    co = tl.cos(a)
    si = tl.sin(a)
    area = tl.full((B,), 0, tl.float32)
    for src in tl.static_range(2):
        l = -0.6 if src == 0 else -0.15
        r = 0.6 if src == 0 else 0.15
        bot = 0.0 if src == 0 else 0.3
        top = 0.3 if src == 0 else 1.2
        for j in tl.static_range(4):
            x0 = l if j == 0 or j == 3 else r
            y0 = bot if j < 2 else top
            x1 = r if j < 2 else l
            y1 = top if j == 1 or j == 2 else bot
            xx = cx + co * x0 - si * y0
            yy = cy + si * x0 + co * y0
            dx = co * (x1 - x0) - si * (y1 - y0)
            dy = si * (x1 - x0) + co * (y1 - y0)
            inv = 1.0 / tl.where(tl.abs(dx) > 1e-10, dx, 1.0)
            for tgt in tl.static_range(2):
                left = -0.6 if tgt == 0 else -0.15
                right = 0.6 if tgt == 0 else 0.15
                low = 0.0 if tgt == 0 else 0.3
                high = 0.3 if tgt == 0 else 1.2
                u = tl.minimum(1.0, tl.maximum(0.0, (left - xx) * inv))
                v = tl.minimum(1.0, tl.maximum(0.0, (right - xx) * inv))
                aa = tl.minimum(u, v)
                bb = tl.maximum(u, v)
                start = yy + aa * dy
                end = yy + bb * dy
                area -= (
                    dx * (bb - aa) * (_pos(start - low, end - low) - _pos(start - high, end - high))
                )
    tl.store(O + i, tl.minimum(1.0, tl.maximum(0.0, area / 0.63)), mask=i < N)


def compute(pose, goal):
    p, g = torch.broadcast_tensors(pose[..., :3], goal[..., :3])
    shape = p.shape[:-1]
    p = p.contiguous()
    g = g.contiguous()
    out = p.new_empty(shape)
    n = out.numel()
    # Size is a runtime argument so varying episode batches share compiled kernels.
    _area[(triton.cdiv(n, 128),)](p, g, out, n, 128)
    return out
