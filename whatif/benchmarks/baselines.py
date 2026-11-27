"""Original upstream CoPhyNet, MBRL-Lib PETS, and NCII reproduction on Stage 33 benchmarks.

Strictly unmodified original configs and standard recorded budgets from Stage 27 (whatif/effects):
  - CoPhyNet: 50 epochs, Adam lr=1e-3, batch 32 single-action sequences, original get_losses(w_stab=1, w_pose=10)
  - PETS: MBRL-Lib 7-member GaussianMLP (4x200 SiLU), 5 elites, lr=1e-5, wd=5e-4, 25 epochs, 20-particle TS-inf
  - NCII: PointNet reproduction (hidden=512), lr=1e-4, 2400 updates
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from whatif.crossbench.models import cophy_class
from whatif.effects.baseline import (
    cophy_inputs,
    cophy_loss,
    cophy_predict,
    pets_batch,
    pets_build,
    pets_predict,
)
from whatif.effects.collect import ROOT
from whatif.effects.data import subset
from whatif.effects.model import average
from whatif.effects.ncii import (
    NCII,
    command_inputs,
    max_peers,
    nll,
    predict as ncii_predict,
    training_examples,
    valid_sources,
)
from .collect import PROTOCOL
from .data import NEW_DOMAINS, coarse_load

OUT = ROOT / 'results/benchmarks/baselines'


def train_cophy(domain):
    device = 'cuda'
    train, val = [coarse_load(domain, split, device) for split in ('train', 'val')]
    model = cophy_class()().to(device)
    original = cophy_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    start = time.monotonic()
    for epoch in range(1, PROTOCOL['budgets']['cophynet_epochs']):
        model.train()
        order = torch.randperm(train['a'].shape[0] * train['a'].shape[1], device=device)
        for ix in order.split(32):
            rows, cols = ix // train['a'].shape[1], ix % train['a'].shape[1]
            d = subset(train, rows)
            for key in ('a', 'reset', 'pose', 'edge', 'reward', 'off'):
                if key in d:
                    d[key] = d[key][torch.arange(len(ix), device=device), cols][:, None]
            o = cophy_predict(model, d)
            seq = torch.cat([o['initial'][:, None], o['truth']], 1)
            indices = torch.arange(o['truth'].size(1), device=device)
            stable = ((seq[:, (indices - 2).clamp_min(0)] - seq[:, (indices + 2).clamp_max(seq.size(1) - 1)])
                      .abs().sum(-1) > 0.05).float() * o['mask'][:, None]
            loss, _ = original(o['raw'], o['stability'], o['mask'], o['truth'], stable, o['mask'],
                               w_stab=1.0, w_pose=10.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == PROTOCOL['budgets']['cophynet_epochs'] - 1:
            model.eval()
            with torch.no_grad():
                mse = []
                for ix in torch.arange(len(val['x']), device=device).split(32):
                    d = subset(val, ix)
                    err = cophy_predict(model, d)['pose'] - d['pose'][..., :3]
                    mse.append(float(((err.square().sum(-1) * d['role'][:, None, None, :, 0]).sum()
                                      / (d['role'][..., 0].sum() * err.size(1) * err.size(2))).sqrt()))
            history.append(dict(epoch=epoch, validation_pose_batch_rmse=float(np.mean(mse)),
                                seconds=round(time.monotonic() - start, 1)))
            print(domain, 'cophynet', history[-1], flush=True)
            (OUT / f'cophynet-{domain}-fit.json').write_text(json.dumps(history, indent=2) + '\n')
    torch.save(model.state_dict(), OUT / f'cophynet-{domain}.pt')


def train_pets(domain):
    from mbrl.models import ModelTrainer
    from mbrl.util.replay_buffer import BootstrapIterator
    device = 'cuda'
    data = pets_batch(coarse_load(domain, 'train'))
    rewards = domain != 'ballscf'
    out_size = data.obs.shape[-1] + int(rewards)
    in_size = data.obs.shape[-1] + data.act.shape[-1]
    model = pets_build(in_size, out_size, device, rewards)
    model.update_normalizer(data)
    batches = BootstrapIterator(data, 32, 7, shuffle_each_epoch=True, permute_indices=True,
                                rng=np.random.default_rng(PROTOCOL['seed']))
    trainer = ModelTrainer(model, optim_lr=1e-5, weight_decay=5e-4)
    log = []
    start = time.monotonic()

    def record(_model, _iteration, epoch, loss, score, best):
        log.append(dict(epoch=epoch + 1, nll=float(loss), mse=float(score.mean()),
                        seconds=round(time.monotonic() - start, 1)))
        print(domain, 'pets', log[-1], flush=True)
        (OUT / f'pets-{domain}-fit.json').write_text(json.dumps(log, indent=2) + '\n')

    trainer.train(batches, num_epochs=PROTOCOL['budgets']['pets_epochs'], patience=25,
                  callback=record, silent=True)
    norm = model.input_normalizer
    torch.save(dict(model=model.state_dict(), mean=norm.mean, std=norm.std,
                    elites=[int(i) for i in model.model.elite_models],
                    in_size=in_size, out_size=out_size, learned_rewards=rewards),
               OUT / f'pets-{domain}.pt')


def train_ncii(domain):
    dev = 'cuda'
    d = coarse_load(domain, 'train', dev)
    data = training_examples(d)
    n = d['x'].size(1)
    model = NCII(n).to(dev)
    model.scale.copy_(data[-1][data[2].sum(-1).bool()].std(0).clamp_min(0.08))
    of = torch.optim.Adam([*model.embed.parameters(), *model.forward_model.parameters()],
                          lr=1e-4, fused=True)
    oh = torch.optim.Adam([*model.interaction_embed.parameters(), *model.interaction.parameters()],
                          lr=1e-4, fused=True)
    idx = torch.arange(n, device=dev)
    self_edges = F.pad(torch.eye(n, device=dev, dtype=torch.bool), (0, 0, 0, 1))
    histories = []
    start = time.monotonic()
    h_loss = torch.tensor(0.0, device=dev)
    for step in range(1, PROTOCOL['budgets']['ncii_updates'] + 1):
        batch = torch.randint(len(data[0]), (64,), device=dev)
        *args, target = (v[batch] for v in data)
        present = args[2].sum(-1)
        valid = valid_sources(args[2])[:, :, None]
        target = target / model.scale
        own, pair = model.embed(*args)
        full = model.distribution(own, max_peers(pair, valid))
        passive = model.distribution(own, pair[:, idx, idx])
        passive_nll = nll(*passive, target)
        objective = average(nll(*full, target) + 0.5 * passive_nll, present)
        with torch.no_grad():
            null_ok = (passive_nll <= passive_nll[present.bool()].median())[:, None]
            drop = torch.rand_like(pair[..., 0]) < 0.25
            keep = valid & (~(drop & null_ok) | self_edges)
            if step >= 400:
                learned = valid & ((model.logits(*args) >= 0) | self_edges)
                keep = torch.where(null_ok, keep, learned)
        masked = model.distribution(own, max_peers(pair, keep))
        objective = objective + 0.5 * average(nll(*masked, target), present)
        of.zero_grad(set_to_none=True)
        objective.backward()
        of.step()
        if step >= 400:
            with torch.no_grad():
                own, pair = model.embed(*args)
                value, source = pair.masked_fill(~valid[..., None], -1e4).topk(2, dim=1)
                inds = torch.arange(n + 1, device=dev)[None, :, None, None]
                omitted = torch.where(inds == source[:, :1], value[:, 1:2], value[:, :1])
                omitted = torch.where(self_edges[..., None], value[:, :1], omitted)
                full = model.distribution(own, value[:, 0])
                null = model.distribution(own[:, None].expand(-1, n + 1, -1, -1), omitted)
                label = (nll(*null, target[:, None]) - nll(*full, target)[:, None] > 1).float()
            logits = model.logits(*args)
            bce = F.binary_cross_entropy_with_logits(logits, label, reduction='none')
            h_loss = average(bce, valid * present[:, None] * (~self_edges))
            oh.zero_grad(set_to_none=True)
            h_loss.backward()
            oh.step()
        if step % 400 == 0:
            rec = dict(step=step, loss=float(objective.detach()),
                       interaction_loss=float(h_loss.detach()),
                       seconds=round(time.monotonic() - start, 1))
            histories.append(rec)
            (OUT / f'ncii-{domain}-fit.json').write_text(json.dumps(histories, indent=2) + '\n')
            print(domain, 'ncii', rec, flush=True)
    torch.save(dict(model=model.state_dict(), n=n), OUT / f'ncii-{domain}.pt')


def restore_cophy(domain, device):
    path = (OUT if domain in NEW_DOMAINS else ROOT / 'results/effects') / f'cophynet-{domain}.pt'
    net = cophy_class()().to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net.eval()


def restore_pets(domain, device):
    path = (OUT if domain in NEW_DOMAINS else ROOT / 'results/effects') / f'pets-{domain}.pt'
    ck = torch.load(path, map_location=device)
    model = pets_build(ck['in_size'], ck['out_size'], device, ck['learned_rewards'])
    model.load_state_dict(ck['model'])
    model.input_normalizer.mean = ck['mean']
    model.input_normalizer.std = ck['std']
    model.model.set_elite(ck['elites'])
    return model.eval()


def restore_ncii(domain, device):
    path = (OUT if domain in NEW_DOMAINS else ROOT / 'results/effects') / f'ncii-{domain}.pt'
    state = torch.load(path, map_location=device)
    net = NCII(state['n']).to(device)
    net.load_state_dict(state['model'])
    return net.eval()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name', choices=('cophynet', 'pets', 'ncii'))
    p.add_argument('domain', choices=NEW_DOMAINS)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(PROTOCOL['seed'])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    {'cophynet': train_cophy, 'pets': train_pets, 'ncii': train_ncii}[args.name](args.domain)
