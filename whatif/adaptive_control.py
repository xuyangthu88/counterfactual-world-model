"""Select a real first stroke, update the physics belief, and select a second stroke."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .chain_eval import calibrate, predict, weights
from .chain_model import Branches, ContactTransfer


@torch.no_grad()
def run_model(model, panels, val, seed, family):
    base = panels[0]
    scene = torch.arange(len(base.ids), device=base.state.device)
    probes = [panel.probe for panel in panels]
    fixed = probes.index(5)
    pred0 = predict(model, base, 0)
    start = base.state[scene, 0, base.target, :2]
    midpoint = (start + base.goal) / 2
    first_ends = start[:, None, None] + pred0[scene, :, :, base.target, :2]
    midpoint_cost = (first_ends[:, probes] - midpoint[:, None, None]).norm(dim=-1).mean(dim=-1)
    adaptive = midpoint_cost.argmin(dim=1)
    sigma = calibrate(model, val)
    real_by_probe, selected_by_probe, oracle_by_probe = [], [], []
    for panel in panels:
        probe = panel.probe
        observation = panel.delta[:, 0, probe] * panel.scale
        posterior = weights(pred0[:, probe], observation, sigma[probe], panel.mask)
        fresh = posterior.reshape(-1, 9, 3).sum(dim=-1).repeat_interleave(3, dim=-1) / 3
        pred1 = predict(model, panel, 1)[scene, :, :, panel.target, :2]
        pred1 = pred1 + panel.state[scene, 1, panel.target, :2][:, None, None]
        distance = (pred1 - panel.goal[:, None, None]).norm(dim=-1)
        action = (fresh[:, None] * distance).sum(dim=-1).argmin(dim=1)
        real = (panel.end[scene, 1, :, panel.target, :2] - panel.goal[:, None]).norm(dim=-1)
        real_by_probe.append(real[scene, action])
        selected_by_probe.append(action)
        oracle_by_probe.append(real.amin(dim=1))
    end = torch.stack(real_by_probe, dim=1)
    selected = torch.stack(selected_by_probe, dim=1)
    oracle = torch.stack(oracle_by_probe, dim=1)
    rows = []
    for rule, first in [('fixed', torch.full_like(adaptive, fixed)), ('midpoint', adaptive)]:
        end_distance = end[scene, first]
        fields = torch.stack((first, selected[scene, first], end_distance * 1000,
                              end_distance <= .012, oracle[scene, first] * 1000), dim=1)
        for scene_id, (first_i, second_i, mm, success, best_mm) in zip(base.ids, fields.cpu().tolist()):
            rows.append(dict(scene=int(scene_id), seed=seed, model=family, rule=rule,
                             first=probes[int(first_i)], second=int(second_i), end_mm=mm,
                             success=int(success), best_second_mm=best_mm))
    return rows


def summarize(rows):
    rng = np.random.default_rng(20261124)
    scenes = sorted({r['scene'] for r in rows})
    ix = rng.integers(len(scenes), size=(5000, len(scenes)))
    fields = ['end_mm', 'success', 'best_second_mm']
    by_scene, results = {}, {}
    for family in ('graph', 'dense'):
        for rule in ('fixed', 'midpoint'):
            group = sorted((r for r in rows if r['model'] == family and r['rule'] == rule),
                           key=lambda r: (r['seed'], r['scene']))
            matrix = np.array([[r[k] for k in fields] for r in group]).reshape(-1, len(scenes), len(fields))
            by_scene[family, rule] = matrix.mean(axis=0)
            first, counts = np.unique([r['first'] for r in group], return_counts=True)
            results[f'{family}/{rule}'] = dict(
                means=dict(zip(fields, matrix.mean(axis=(0, 1)).tolist())),
                per_seed=dict(zip(fields, matrix.mean(axis=1).T.tolist())),
                first_counts=dict(zip(map(str, first), counts.tolist())))
    for left, right in [(('graph', 'midpoint'), ('graph', 'fixed')),
                        (('dense', 'midpoint'), ('dense', 'fixed')),
                        (('graph', 'midpoint'), ('dense', 'midpoint'))]:
        difference = by_scene[left] - by_scene[right]
        limits = np.quantile(difference[ix].mean(axis=1), [.025, .975], axis=0)
        key = '/'.join(left) + ' minus ' + '/'.join(right)
        results[key] = {k: [float(difference[:, i].mean()), *limits[:, i].tolist()] for i, k in enumerate(fields)}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('data/chains'))
    ap.add_argument('--out', type=Path, default=Path('results/chains'))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seeds', nargs='+', type=int, default=[11, 29, 47])
    args = ap.parse_args()
    torch.set_num_threads(2)
    panels = [Branches(args.data / (name + '.npz'), args.device)
              for name in ('probe32', 'probe41', 'probe54', 'probe69')]
    base = panels[0]
    for panel in panels[1:]:
        assert np.array_equal(base.ids, panel.ids)
        for a, b in [(base.state[:, 0], panel.state[:, 0]), (base.end[:, 0], panel.end[:, 0]),
                     (base.physics, panel.physics), (base.noise, panel.noise), (base.goal, panel.goal)]:
            torch.testing.assert_close(a, b, atol=1e-7, rtol=0)
    val = Branches(args.data / 'val.npz', args.device)
    rows = []
    for seed in args.seeds:
        for family in ('graph', 'dense'):
            model = ContactTransfer(family == 'graph').to(args.device)
            model.load_state_dict(torch.load(args.out / f'{family}-{seed}.pt',
                                             map_location=args.device, weights_only=True))
            rows.extend(run_model(model.eval(), panels, val, seed, family))
    with (args.out / 'adaptive_per_scene.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (args.out / 'adaptive_summary.json').write_text(json.dumps(summarize(rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
