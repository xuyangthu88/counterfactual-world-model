"""Score the frozen distal XY action-effect distribution against real sibling branches."""
import argparse
import csv
from pathlib import Path

import torch

from .chain_eval import calibrate, predict, weights
from .chain_model import Branches, ContactTransfer


@torch.no_grad()
def evaluate(model, data, sigma, seed, split):
    ref = 1
    ids = torch.arange(len(data.ids), device=data.state.device)
    alt = torch.arange(len(data.actions), device=ids.device) != ref
    pred = predict(model, data, 0)
    posterior = weights(pred[:, ref], data.delta[:, 0, ref] * data.scale, sigma[ref], data.mask)
    xy = pred[ids, :, :, data.target, :2]
    particles = (xy[:, alt] - xy[:, ref:ref + 1]) * 1000
    ends = data.end[ids, 0, :, data.target, :2]
    true = (ends[:, alt] - ends[:, ref:ref + 1]) * 1000
    distance = (particles - true[:, :, None]).norm(dim=-1)
    pairwise = torch.cdist(particles, particles, compute_mode='donot_use_mm_for_euclid_dist')
    rows = []
    for mode, weight in [('prior', torch.full_like(posterior, 1 / posterior.shape[-1])), ('reflect', posterior)]:
        w = weight[:, None, :]
        first = (w * distance).sum(dim=-1)
        second = (w[..., None] * w[..., None, :] * pairwise).sum(dim=(-1, -2))
        score = (first - .5 * second).mean(dim=1)
        for scene, es in zip(data.ids, score.tolist()):
            rows.append(dict(scene=int(scene), seed=seed, split=split, mode=mode, energy_mm=es))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('data/chains'))
    ap.add_argument('--out', type=Path, default=Path('results/chains'))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    torch.set_num_threads(2)
    val = Branches(args.data / 'val.npz', args.device)
    splits = {name: Branches(args.data / (name + '.npz'), args.device) for name in ('two', 'three', 'physical')}
    rows = []
    for seed in (11, 29, 47):
        model = ContactTransfer(True).to(args.device)
        model.load_state_dict(torch.load(args.out / f'graph-{seed}.pt', map_location=args.device, weights_only=True))
        sigma = calibrate(model.eval(), val)
        for name, data in splits.items():
            rows.extend(evaluate(model, data, sigma, seed, name))
    with (args.out / 'distribution_per_scene.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
