"""Two-step paired-control lookahead with exactly independent future process draws."""

import torch
from whatif.shared.model import grid, task_value
from whatif.shared.planning import menus, mechanical


def immediate(st, action, z, peak, rank):
    learned = 0.55 * task_value(st, z, peak) + 0.45 * rank
    return 0.5 * learned + 0.5 * mechanical(st, action)


def two_step(m, s, actions, mass, task, root_only=False):
    # [batch, first action, physical support, first noise, second action, second noise]
    z, _, p, r = grid(m.graph, s, actions)
    n, b = len(s), actions.shape[1]
    z = z.reshape(n, b, 5, 3, 19)
    p = p.reshape(n, b, 5, 3)
    r = r.reshape_as(p)
    first = immediate(s[:, None, None, None], actions[:, :, None, None], z, p, r)
    cmds = menus(z, task)
    shape = (*cmds.shape[:-1], 3)
    state = z[..., None, None, :].expand(*shape, 19)
    cmd = cmds[..., None, :].expand(*shape, 7)
    k = s.new_tensor([-1.0, -0.5, 0.0, 0.5, 1.0])[None, None, :, None, None, None].expand(shape)
    u = s.new_tensor([-1.0, 0.0, 1.0]).expand(shape)
    end, _, pk, rank = m.graph(state, cmd, k, u)
    later = immediate(state, cmd, end, pk, rank) + 0.24 * m.value_readout(end)
    finished = z[..., 12] > (0.90 + 0.05 * z[..., 13])
    returns = first[..., None] + 0.85 * torch.where(
        finished[..., None], first[..., None], later.mean(-1)
    )
    nominal = returns[:, 1:2, :, :, 1:2]
    criterion = returns + 0.12 * ((returns - nominal) / 0.30).tanh()
    if root_only:
        # Apply the observed-stroke correction only where it was trained: the next event.
        paired_first = first + 0.12 * ((first - first[:, 1:2]) / 0.30).tanh()
        correction = (paired_first.mean(3) * (mass[:, None] - 0.2)).sum(2)
        joint = criterion.mean((2, 3)) + correction[..., None]
    else:
        joint = (criterion.mean(3) * mass[:, None, :, None]).sum(2)
    return joint.amax(-1).argmax(-1)
