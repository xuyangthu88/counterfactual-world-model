"""Common object, factor, action, and factual-history tensors."""
import numpy as np
import torch
from torch.nn.functional import pad

from .collect import DATA, PUSH_WINDOWS

WINDOWS = 5
BRANCHED = ('a', 'reset', 'pose', 'edge', 'reward', 'off')
UNIT = {'cophy': 1., 'causalworld': 65., 'pusht': 100.}


def load(domain, split, device='cpu'):
    with np.load(DATA / f'{domain}-{split}.npz') as file:
        d = {key: torch.as_tensor(file[key], device=device) for key in file.files}
    d = {k: v.float() if k != 'seed' else v for k, v in d.items()}
    x, action = d['x'], d['a']
    observed_alternative = 'off' in d
    # A measured fork may specify a new initial pose directly. Commanded waypoints
    # are targets, so they leave the initial physical state unchanged.
    reset = pad(action[..., 1:4], (0, 6)) if observed_alternative else x.new_zeros(*action.shape[:-1], 9)
    widths = [6]*WINDOWS if observed_alternative else np.diff([0, *PUSH_WINDOWS]).tolist()
    reward = d.get('rewards', x.new_zeros(*action.shape[:2], WINDOWS))
    return dict(x=x, a=action, reset=reset, role=d['role'], goal=d['goal'], fa=d['fa'],
                fact=d['fact'], fe=d['fe'], pose=d['traj'], edge=d['contact'], reward=reward,
                widths=x.new_tensor(widths)[None].expand(len(x), -1),
                same=x.new_full((len(x),), float(observed_alternative)), seed=d['seed'],
                **({'off': d['off']} if observed_alternative else {}))


def subset(d, rows, actions=None):
    b = {key: value[rows] for key, value in d.items()}
    if actions is not None:
        for key in BRANCHED:
            if key in b:
                b[key] = b[key][:, actions]
    return b


def clock(width):
    return torch.stack([torch.log1p(width/8), width.cumsum(-1)/width.sum(-1, keepdim=True)], -1)


def pair_ids(n, device):
    return torch.tril_indices(n, n, offset=-1, device=device)


def omission(d, count=2):
    """Draw valid pairs, increasing the sampling weight of consequential removals."""
    present = d['role'].sum(-1)
    i, j = pair_ids(present.size(1), present.device)
    valid = (present[:, i]*present[:, j])[:, None].expand(*d['a'].shape[:2], -1)
    amount = d['off'].square().sum(-1).amax((-1, -2)).sqrt()
    active = (amount > .1) & valid.bool()
    weights = valid*(1+4*active)
    picks = torch.multinomial(weights.flatten(0, 1), count, replacement=False).unflatten(0, valid.shape[:2])
    return picks
