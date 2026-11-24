"""Paired prediction, second-action ranking and complete two-push evaluation."""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np
import torch

from .chain_eval import calibrate, predict
from .chain_model import LENGTH
from .pilot_models import (OUT, graph_model, graph_panels, index_set, load_data, load_head,
                           read, simple_panel)


class Planner:
    def __init__(self, phase, device):
        val = load_data('val', device)
        self.graph = {f: graph_model(phase, f, device) for f in ('graph', 'dense')}
        self.sigma = {f: calibrate(m, val) for f, m in self.graph.items()}
        feature = graph_panels(self.graph['graph'], val, self.sigma['graph'], second=False)[0].features
        self.correction = load_head(feature, True, OUT/f'{phase}-correction.pt')
        self.transport = load_head(simple_panel(val, 0).features, False, OUT/f'{phase}-transport.pt')
        self.alpha = json.loads((OUT/f'{phase}-transport.json').read_text())['alpha']

    @torch.no_grad()
    def evaluate(self, d):
        i, j, _ = index_set(d, 0)
        samples = {}
        for family, model in self.graph.items():
            prior = predict(model, d, 0)[i, :, :, j, :2] / LENGTH
            past, future = graph_panels(model, d, self.sigma[family])
            samples[family] = (prior, past.base, past.weight, future.base, future.weight)
            if family == 'graph':
                samples['corrected'] = (prior, read(past, self.correction, i), past.weight,
                                        read(future, self.correction, i), future.weight)
        p0, p1 = simple_panel(d, 0), simple_panel(d, 1)
        m0, m1 = self.transport(p0.features), self.transport(p1.features)
        ref = 1
        anchor = p0.truth[:, ref] - m0[:, ref]
        carry = p0.truth[:, d.probe] - m0[:, d.probe]
        samples['transport'] = (m0[:, :, None], (m0+anchor[:, None])[:, :, None], p0.weight,
                                (m1+self.alpha*carry[:, None])[:, :, None], p1.weight)
        result = {}
        for name, (prior, hindsight, w0, second, w1) in samples.items():
            def score(x, goal, weight):
                return ((x-goal[:, None, None]).norm(dim=-1)*weight[:, None]).sum(dim=2)
            midpoint_cost = score(prior, p0.goal*.5, torch.ones_like(prior[:, 0, :, 0])/prior.shape[2])
            second_cost = score(second, p1.goal, w1)
            choice = second_cost.argmin(dim=1)
            real = (p1.truth-p1.goal[:, None]).norm(dim=-1) * (1000*LENGTH)
            final = real[i, choice]
            regret = final - real.amin(dim=1)
            true_diff = real[:, :, None]-real[:, None, :]
            model_diff = second_cost[:, :, None]-second_cost[:, None, :]
            informative = true_diff.abs() > 2.
            correct = ((true_diff*model_diff > 0) & informative).sum(dim=(1,2))
            alt = torch.arange(len(d.actions), device=i.device) != ref
            x = (hindsight[:, alt]-hindsight[:, ref:ref+1]) * (1000*LENGTH)
            y = (p0.truth[:, alt]-p0.truth[:, ref:ref+1]) * (1000*LENGTH)
            err = ((x*w0[:, None, :, None]).sum(dim=2)-y).norm(dim=-1).mean(dim=1)
            score1 = ((x-y[:, :, None]).norm(dim=-1)*w0[:, None]).sum(dim=2)
            dist = torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist')
            score2 = (dist*w0[:, None, None, :]*w0[:, None, :, None]).sum(dim=(2,3))
            metrics = torch.stack((err, (score1-.5*score2).mean(dim=1), final, final <= 12.,
                                   regret, correct, informative.sum(dim=(1,2)), choice), dim=1)
            result[name] = (midpoint_cost.cpu().numpy(), metrics.cpu().numpy())
        return result


KEYS = ['effect_mm', 'energy_mm', 'end_mm', 'success', 'regret_mm', 'rank_correct', 'rank_total', 'second']


def collect(phase, device):
    planner = Planner(phase, device)
    rows, branches = [], []
    panel_names = ['test2', 'test3_41', 'shift', 'test3_32', 'test3_54', 'test3_69']
    for split in panel_names:
        d = load_data(split, device)
        predictions = planner.evaluate(d)
        if split.startswith('test3'):
            if branches:
                assert np.array_equal(branches[0][1].ids, d.ids)
                torch.testing.assert_close(branches[0][1].end[:, 0], d.end[:, 0])
            branches.append((d.probe, d, predictions))
        if split in ('test2', 'test3_41', 'shift'):
            for name, (_, scores) in predictions.items():
                for scene, entry in zip(d.ids, scores.tolist()):
                    rows.append(dict(phase=phase, task=split, scene=int(scene), model=name,
                                     first=d.probe, **dict(zip(KEYS, entry))))
    probes = [r[0] for r in branches]
    for name in branches[0][2]:
        costs = branches[0][2][name][0][:, probes]
        first = costs.argmin(axis=1)
        for i, k in enumerate(first):
            probe, d, output = branches[k]
            rows.append(dict(phase=phase, task='adaptive3', scene=int(d.ids[i]), model=name,
                             first=probe, **dict(zip(KEYS, output[name][1][i].tolist()))))
    return rows


def summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['phase'], r['task'], r['model']].append(r)
    values, tables, contrasts = {}, {}, {}
    keys = KEYS[:5]
    for label, group in groups.items():
        group.sort(key=lambda r: r['scene'])
        mat = np.array([[r[k] for k in keys] for r in group])
        values[label] = mat
        tables['/'.join(label)] = {**dict(zip(keys, mat.mean(axis=0).tolist())),
                                   'rank_accuracy': sum(r['rank_correct'] for r in group) /
                                   sum(r['rank_total'] for r in group), 'layouts': len(group)}
    for phase, task, model in values:
        if model != 'corrected':
            continue
        x = values[phase,task,model]
        ix = np.random.default_rng(20261124).integers(len(x), size=(3000,len(x)))
        for other in ('graph','transport','dense'):
            delta = x-values[phase,task,other]
            ci = np.quantile(delta[ix].mean(axis=1), [.025,.975], axis=0)
            contrasts['/'.join([phase,task,other])] = {
                k: [float(delta[:,i].mean()), *ci[:,i].tolist()] for i,k in enumerate(keys)}
    return dict(means=tables, corrected_minus_reference=contrasts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    torch.set_num_threads(2)
    rows = []
    for phase in ('zero','adapt'):
        rows.extend(collect(phase, args.device))
    with (OUT/'per_scene.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(summary(rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
