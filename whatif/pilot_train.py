"""One training seed in each coverage regime; validation chooses the checkpoints."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .chain_eval import calibrate
from .chain_model import Branches, fit
from .pilot_models import (CHAINS, DATA, OUT, SEED, fit_head, graph_model, graph_panels,
                           load_data, simple_panel)


def pooled(names, device, dest):
    keys = ('states', 'seed', 'count', 'tips', 'goal', 'edges', 'physics', 'noise')
    tables = {k: [] for k in keys}
    for name in names:
        with np.load((CHAINS if name in ('train', 'val') else DATA)/(name+'.npz')) as f:
            for k in keys:
                tables[k].append(f[k])
            actions = f['actions']
    np.savez(dest, actions=actions, probe=5, **{k: np.concatenate(v) for k, v in tables.items()})
    return Branches(dest, device)


def fit_phase(phase, device):
    train_names = ['train']
    val_names = ['val']
    if phase == 'adapt':
        train_names += ['train3_41', 'train3_54']
        val_names += ['val3']
        with TemporaryDirectory() as temp:
            train = pooled(train_names, device, Path(temp)/'train.npz')
            val = pooled(val_names, device, Path(temp)/'val.npz')
            for family in ('graph', 'dense'):
                fit(train, val, SEED, family == 'graph', OUT/f'{phase}-{family}.pt', 100)
                print(phase, family, flush=True)
    train = [load_data(n, device) for n in train_names]
    valid = [load_data(n, device) for n in val_names]
    # The 2-block likelihood calibration set is identical across all physics models.
    graph = graph_model(phase, 'graph', device)
    sigma = calibrate(graph, valid[0])
    train_graph = [panel for d in train for panel in graph_panels(graph, d, sigma)]
    val_graph = [panel for d in valid for panel in graph_panels(graph, d, sigma)]
    fit_head(train_graph, val_graph, True, device, OUT/f'{phase}-correction.pt')
    train_mean = [simple_panel(d, t) for d in train for t in (0, 1)]
    val_mean = [simple_panel(d, t) for d in valid for t in (0, 1)]
    head = fit_head(train_mean, val_mean, False, device, OUT/f'{phase}-transport.pt')
    # The paper uses persistence to transport an event residual to subsequent
    # transitions. Select its scalar carry-over on validation, using actual costs.
    options = [0., .25, .5, .75, 1.]
    costs = [[] for _ in options]
    with torch.no_grad():
        for d in valid:
            p0, p1 = (simple_panel(d, t) for t in (0, 1))
            e = p0.truth[:, d.probe] - head(p0.features)[:, d.probe]
            base = head(p1.features)
            real = (p1.truth - p1.goal[:, None]).norm(dim=-1)
            ix = torch.arange(len(d.ids), device=device)
            for k, alpha in enumerate(options):
                action = (base + alpha*e[:, None] - p1.goal[:, None]).norm(dim=-1).argmin(dim=1)
                costs[k].append(real[ix, action].mean().item())
    mean_costs = np.mean(costs, axis=1)
    (OUT/f'{phase}-transport.json').write_text(json.dumps(
        dict(alpha=options[int(mean_costs.argmin())], candidates=options,
             validation_mm=(mean_costs * 65).tolist()), indent=2)+'\n')
    print(phase, 'heads complete', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['zero', 'adapt'], required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    fit_phase(args.phase, args.device)


if __name__ == '__main__':
    main()
