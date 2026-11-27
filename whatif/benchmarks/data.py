"""Unified 5-domain loader supporting original 3 domains + Stage 33 BallsCF & Stacking2."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import pad

from whatif.effects.collect import DATA as EFFECTS_DATA, PUSH_WINDOWS
from whatif.effects.data import BRANCHED, WINDOWS, clock, pair_ids, subset
from whatif.shortstep.collect import DATA as SHORTSTEP_DATA
from whatif.shortstep.data import paired_removal
from .collect import DATA as BENCH_DATA, NATIVE_DATA as BENCH_NATIVE_DATA, PROTOCOL

ALL_DOMAINS = tuple(PROTOCOL['all_domains'])
NEW_DOMAINS = tuple(PROTOCOL['domains'])
UNIT = {'cophy': 1.0, 'ballscf': 1.0, 'causalworld': 65.0, 'stacking2': 65.0, 'pusht': 100.0}


def coarse_path(domain, split):
    root = BENCH_DATA if domain in NEW_DOMAINS else EFFECTS_DATA
    return root / f'{domain}-{split}.npz'


def native_path(domain, split):
    root = BENCH_NATIVE_DATA if domain in NEW_DOMAINS else SHORTSTEP_DATA
    return root / f'{domain}-{split}.npz'


def coarse_load(domain, split, device='cpu'):
    with np.load(coarse_path(domain, split)) as file:
        d = {key: torch.as_tensor(file[key], device=device) for key in file.files}
    d = {k: v.float() if k != 'seed' else v for k, v in d.items()}
    x, action = d['x'], d['a']
    observed_alternative = 'off' in d
    reset = pad(action[..., 1:4], (0, 6)) if observed_alternative else x.new_zeros(*action.shape[:-1], 9)
    widths = [6] * WINDOWS if observed_alternative else np.diff([0, *PUSH_WINDOWS]).tolist()
    reward = d.get('rewards', x.new_zeros(*action.shape[:2], WINDOWS))
    return dict(
        x=x, a=action, reset=reset, role=d['role'], goal=d['goal'], fa=d['fa'],
        fact=d['fact'], fe=d['fe'], pose=d['traj'], edge=d['contact'], reward=reward,
        widths=x.new_tensor(widths)[None].expand(len(x), -1),
        same=x.new_full((len(x),), float(observed_alternative)), seed=d['seed'],
        **({'off': d['off']} if observed_alternative else {}),
    )


def schedule(d, domain):
    clk_key = 'cophy_clock' if domain in ('cophy', 'ballscf') else 'push_clock'
    clk = PROTOCOL[clk_key]
    ends = d['widths'][0].cumsum(0)
    times = ends.new_tensor(clk)
    parent = torch.searchsorted(ends, times)
    widths = torch.diff(times, prepend=times.new_zeros(1))
    take = torch.searchsorted(times, ends)
    if take[-1] >= len(times) or not torch.equal(ends, times[take]):
        raise ValueError(f'Every coarse endpoint must have a native label ({domain})')
    d['end_idx'] = take.expand(len(d['x']), -1)
    d['window'] = (torch.arange(len(ends), device=parent.device)[:, None] == parent).expand(len(d['x']), -1, -1)
    d['native_widths'] = widths.expand(len(d['x']), -1)
    d['nominal_widths'] = d['widths'][:, parent]
    return d


def load(domain, split, device='cpu', *, labels=True):
    d = schedule(coarse_load(domain, split, device), domain)
    if not labels:
        return d
    with np.load(native_path(domain, split)) as file:
        for key in file.files:
            if key == 'seed':
                if not np.array_equal(file[key], d[key].cpu().numpy()):
                    raise ValueError(f'Different layout order for {domain} {split}')
            else:
                d[key] = torch.as_tensor(file[key], device=device)
    d['native_off'] = d['native_off'].float()
    return d
