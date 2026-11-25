"""Two-step paired action returns with native passage as an absorbing outcome."""

import json
import torch
from whatif.shared.model import task_value
from whatif.shared.planning import menus, mechanical
from whatif.prospective.planner import cem
from .model import physical_history, passage_grid, OUT


def beam(m, s, a, ctx, task, mass, mode, return_scores=False):
    z, _, peak, rank, p = passage_grid(m.graph, s, a, ctx)
    st = s[:, None, None]
    cmd = a[:, :, None]
    nohit = mode.startswith("noevent-")
    thresh = 0.90 + 0.05 * s[:, 13, None, None]
    # The endpoint ablation uses the same predicted state, graph and search.
    if nohit:
        peak = z[..., 12]
        p = (peak > thresh).float()
    event = 0.0 if nohit else 0.50
    timed = mode.endswith("timed")
    mix = 0.28 if mode.endswith("guided") or timed else 0.50

    def immediate(st, a, z, peak, rank, p):
        value = 0.60 * task_value(st, z, peak) + 0.40 * rank
        utility = mix * value + (1 - mix) * mechanical(st, a) + event * p
        if timed:
            from .duration import duration

            utility = utility - 0.0035 * duration(m, st.expand_as(z), a.expand(*z.shape[:-1], 7))
        return utility

    now = immediate(st, cmd, z, peak, rank, p)
    n, b, particles, _ = z.shape
    # Each second-step template denotes a contact site/length choice relative to
    # that predicted particle's state. Its index is optimized across all particles.
    opts = menus(z, task)
    mech = mechanical(z[..., None, :], opts)
    w = mass.repeat_interleave(3, -1) / 3
    ids = (mech * w[:, None, :, None]).sum(2).topk(4, -1).indices
    sel = opts.gather(3, ids[:, :, None, :, None].expand(n, b, particles, 4, 7))
    state = z[..., None, None, :].expand(n, b, particles, 4, 3, 19)
    future_cmd = sel.unsqueeze(-2).expand(n, b, particles, 4, 3, 7)
    shape = future_cmd.shape[:-1]
    k = (
        s.new_tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        .repeat_interleave(3)[None, None, :, None, None]
        .expand(shape)
    )
    u = s.new_tensor([-1.0, 0.0, 1.0]).expand(shape)
    hh = physical_history(ctx)[:, None, None, None, None, :]
    zz, _, q, rr, pp = m.graph.predict_event(state, future_cmd, k, u, hh)
    if nohit:
        q = zz[..., 12]
        pp = (q > thresh[..., None, None]).float()
    later = immediate(state, future_cmd, zz, q, rr, pp) + 0.24 * m.value_readout(zz)
    future = later.mean(-1)
    best = (future * w[:, None, :, None]).sum(2).argmax(-1)
    future = future.gather(-1, best[:, :, None, None].expand(n, b, particles, 1)).squeeze(-1)
    returns = now + 0.85 * (p * now + (1 - p) * future)
    paired = returns + 0.12 * ((returns - returns[:, 1:2]) / 0.30).tanh()
    scores = (paired * w[:, None]).sum(-1)
    return scores if return_scores else scores.argmax(1)


def choose(m, s, a, ctx, task, mass, mode, step=0, seeds=None):
    if mode in ("selected", "noevent", "averagefact"):
        selected = json.loads((OUT / "freeze.json").read_text())[task]["policy"]
        mode = ("noevent-" + selected) if mode == "noevent" else selected
    if mode in ("cem", "noevent-cem"):
        picks = []
        for j in range(len(s)):
            m.graph.context = physical_history(ctx[j])
            picks.append(
                cem(
                    m,
                    s[j],
                    task,
                    int(11 + seeds[j] * 73 + step),
                    mass[j],
                    torch.zeros_like(mass[j]),
                    event_weight=0.45 if mode == "cem" else None,
                    endpoint_ablation=mode != "cem",
                )
            )
        m.graph.context = None
        return torch.as_tensor(picks, device=s.device)
    return torch.cat(
        [
            beam(m, s[j : j + 3], a[j : j + 3], ctx[j : j + 3], task, mass[j : j + 3], mode)
            for j in range(0, len(s), 3)
        ]
    )
