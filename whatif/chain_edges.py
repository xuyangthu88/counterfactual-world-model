"""Measure whether a new push opens or removes physical contact on the target path."""
import argparse
import csv
from pathlib import Path

import torch

from .chain_eval import bank, calibrate, predict, weights
from .chain_model import Branches, ContactTransfer


@torch.no_grad()
def evaluate(model, data, sigma, name, seed):
    pred = predict(model, data, 0)
    particles = bank(data)
    device = data.state.device
    actions = torch.arange(len(data.actions), device=device)
    ids = torch.arange(len(data.ids), device=device)
    nodes = torch.arange(data.n - 1, device=device)
    contact = torch.cat((data.hand[:, 0, :, :1], data.pair[:, 0, :, nodes, nodes + 1]), dim=-1).bool()
    out = []
    for chunk in ids.split(12):
        i, a, k = torch.meshgrid(chunk, actions, torch.arange(len(particles), device=device), indexing='ij')
        args = data.inputs(i, torch.zeros_like(i), data.actions[a], particles[k, :2], particles[k, 2])
        _, hand, pair, valid, possible = model(*args)
        hand_prob = hand[..., :1].sigmoid() * possible[..., :1]
        pair_prob = (pair.sigmoid() * valid)[..., nodes, nodes + 1]
        out.append(torch.cat((hand_prob, pair_prob), dim=-1))
    edge_probs = torch.cat(out)
    rows = []
    for ref in (0, 1):
        weight = weights(pred[:, ref], data.delta[:, 0, ref] * data.scale, sigma[ref], data.mask)
        alt = actions != ref
        predicted = (edge_probs[:, alt] * weight[:, None, :, None]).sum(dim=2) >= .5
        factual = contact[:, ref:ref + 1]
        actual = contact[:, alt]
        for edge in range(data.n):
            f, y, p = factual[..., edge], actual[..., edge], predicted[..., edge]
            for change, eligible, obs, estimate in [('open', ~f, y, p), ('close', f, ~y, ~p)]:
                hit = (eligible & obs & estimate).sum().item()
                positives = (eligible & obs).sum().item()
                false = (eligible & ~obs & estimate).sum().item()
                negatives = (eligible & ~obs).sum().item()
                rows.append(dict(split=name, seed=seed, reference=ref, receiver=edge, event=change,
                                 hits=hit, changed=positives, false_alarms=false, unchanged=negatives))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('data/chains'))
    ap.add_argument('--out', type=Path, default=Path('results/chains'))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    torch.set_num_threads(2)
    val = Branches(args.data / 'val.npz', args.device)
    splits = {key: Branches(args.data / (key + '.npz'), args.device) for key in ('two', 'three', 'physical')}
    rows = []
    for seed in (11, 29, 47):
        model = ContactTransfer(True).to(args.device)
        model.load_state_dict(torch.load(args.out / f'graph-{seed}.pt', map_location=args.device, weights_only=True))
        sigma = calibrate(model.eval(), val)
        for key, data in splits.items():
            rows.extend(evaluate(model, data, sigma, key, seed))
    with (args.out / 'edge_changes.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
