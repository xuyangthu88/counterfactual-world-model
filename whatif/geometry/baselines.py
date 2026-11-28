"""Stage 34: Fair baseline retraining on the EXACT same universal geometry protocol (x_geo + geom).

Every baseline preserves its exact original architecture, optimizer, and recorded training budget:
  - CoPhyNet-Geo: upstream CoPhyNet with input dim widened to read (xyz, rot6, geom) instead of xyz only (50 epochs).
  - PETS-Geo: MBRL-Lib 7-member GaussianMLP (4x200 SiLU, 25 epochs, 20-particle TS-inf) with geom in context.
  - NCII-Geo: PointNet reproduction (hidden=512, 2400 updates) with geom in object_embedding.
"""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from whatif.crossbench.models import cophy_class
from whatif.effects.baseline import cophy_loss, pets_build, pets_predict
from whatif.effects.collect import ROOT
from whatif.effects.data import clock, subset
from whatif.effects.model import average
from whatif.effects.ncii import HIDDEN, max_peers, nll, point_mlp, valid_sources
from .data import ALL_DOMAINS, load

OUT = ROOT / 'results/geometry/baselines'
PROTOCOL = json.loads((ROOT / 'whatif/geometry/protocol.json').read_text())


# --- 1. CoPhyNet with identical upstream architecture, widened ONLY at input to read (xyz, rot6, geom) ---
class CoPhyNetGeo(nn.Module):
    """Exact upstream CoPhyNet (H=32) whose input projection reads [xyz(3), rot6(6), geom(6)] = 15D."""

    def __init__(self):
        super().__init__()
        self.base = cophy_class()()
        H = 32
        D_in = 15
        self.base.mlp_inter[0] = nn.Linear(2 * D_in, H)
        self.base.mlp_out[0] = nn.Linear(D_in + H + H, H)
        self.base.mlp_inter_stab[0] = nn.Linear(2 * (H + D_in), H)
        self.base.mlp_stab[0] = nn.Linear(H + H + H + D_in, H)
        self.base.mlp_inter_delta[0] = nn.Linear(2 * (H + D_in), H)
        self.base.mlp_gcn_delta[0] = nn.Linear(H * 3 + D_in, H)

    @staticmethod
    def _aggreg(E, presence):
        K = E.size(1)
        p = presence.unsqueeze(-1)
        p12 = p.unsqueeze(1) * p.unsqueeze(2)
        eye = torch.eye(K, device=E.device, dtype=torch.bool)[None, :, :, None]
        w = p12.masked_fill(eye, 0.0)
        e_node = (E * w).sum(2) / (0.01 + w.sum(2))
        e_sum = (e_node * p).sum(1, keepdim=True) / (0.01 + p.sum(1, keepdim=True))
        return e_node, e_sum.expand(-1, K, -1)

    def forward_geo(self, fact15, present, initial15, T):
        m = self.base
        K = fact15.size(2)
        ab_out = []
        for t in range(fact15.size(1)):
            x = fact15[:, t]
            x12 = torch.cat([x.unsqueeze(1).expand(-1, K, -1, -1),
                             x.unsqueeze(2).expand(-1, -1, K, -1)], dim=-1)
            E, E_sum = self._aggreg(m.mlp_inter(x12), present)
            ab_out.append(m.mlp_out(torch.cat([x, E, E_sum], dim=-1)))
        seq_o = torch.stack(ab_out, dim=1)
        confounders = m.rnn_on_AB_up(seq_o)

        pose15 = initial15
        B = pose15.size(0)
        list_pose, list_stab = [], []
        last_h = None
        for i in range(T):
            if i == 0:
                xs = torch.cat([confounders, pose15], dim=-1)
                x12 = torch.cat([xs.unsqueeze(1).expand(-1, K, -1, -1),
                                 xs.unsqueeze(2).expand(-1, -1, K, -1)], dim=-1)
                Es, Es_sum = self._aggreg(m.mlp_inter_stab(x12), present)
                stability = m.mlp_stab(torch.cat([xs, Es, Es_sum], dim=-1))
            list_stab.append(stability)

            xd = torch.cat([pose15, confounders], dim=-1)
            x12 = torch.cat([xd.unsqueeze(1).expand(-1, K, -1, -1),
                             xd.unsqueeze(2).expand(-1, -1, K, -1)], dim=-1)
            Ed, Ed_sum = self._aggreg(m.mlp_inter_delta(x12), present)
            _in = m.mlp_gcn_delta(torch.cat([xd, Ed, Ed_sum], dim=-1))
            new_h = []
            for k_idx in range(K):
                if i == 0:
                    hk, *_ = m.rnn_delta(_in[:, [k_idx]])
                else:
                    hk, *_ = m.rnn_delta(_in[:, [k_idx]], last_h[k_idx].reshape(1, B, -1))
                new_h.append(hk)
            last_h = new_h
            hidden = torch.cat(last_h, dim=1)
            delta = m.fc_delta(hidden)
            if self.training:
                delta = delta * (1.0 - torch.sigmoid(stability / 0.01))
            else:
                delta = delta * (1.0 - (stability > 0).float())
            next_xyz = pose15[..., :3] + delta
            pose15 = torch.cat([next_xyz, pose15[..., 3:]], dim=-1)
            list_pose.append(next_xyz)
        return torch.stack(list_pose, dim=1), torch.stack(list_stab, dim=1)


def cophy_geo_inputs(d):
    b, k, n = d['a'].shape[:3]
    present = d['role'].sum(-1)
    geom = d['geom']  # (B, N, 6)
    fact15 = torch.cat([d['fact'], geom[:, None].expand(-1, d['fact'].size(1), -1, -1)], dim=-1)
    init9 = d['x'][:, None] + d['reset']
    init15 = torch.cat([init9, geom[:, None].expand(-1, k, -1, -1)], dim=-1)
    truth = d['pose'][..., :3]
    if 'off' not in d:
        def points(x, a):
            return x[..., None, :] + a[..., 1:].unflatten(-1, (2, 3))
        fp = (points(d['fact'][:, 0, :, :3], d['fa']) * d['fa'][..., 0, None, None]).sum(-3)
        ap = (points(d['x'][:, None, :, :3], d['a']) * d['a'][..., 0, None, None]).sum(-3)
        fp15 = F.pad(fp, (0, 12))
        ap15 = F.pad(ap, (0, 12))
        fact15 = torch.cat([fact15, fp15[:, None].expand(-1, fact15.size(1), -1, -1)], dim=-2)
        init15 = torch.cat([init15, ap15], dim=-2)
        truth = torch.cat([truth, ap[:, :, None].expand(-1, -1, truth.size(2), -1, -1)], dim=-2)
        present = F.pad(present, (0, 2), value=1)
    rep = lambda z: z[:, None].expand(-1, k, *z.shape[1:]).flatten(0, 1)
    return rep(fact15), rep(present), init15.flatten(0, 1), truth.flatten(0, 1), (b, k, n)


def cophy_geo_predict(model, d):
    fact15, present, init15, truth, (b, k, n) = cophy_geo_inputs(d)
    pred, logits = model.forward_geo(fact15, present, init15, T=truth.size(1))
    return dict(raw=pred, stability=logits[..., 0], mask=present, truth=truth,
                initial=init15[..., :3], pose=pred[:, :, :n].unflatten(0, (b, k)))


def train_cophy_geo(domain):
    device = 'cuda'
    train, val = [load(domain, s, device, labels=False) for s in ('train', 'val')]
    model = CoPhyNetGeo().to(device)
    original = cophy_loss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
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
            o = cophy_geo_predict(model, d)
            seq = torch.cat([o['initial'][:, None], o['truth']], 1)
            indices = torch.arange(o['truth'].size(1), device=device)
            stable = ((seq[:, (indices - 2).clamp_min(0)] - seq[:, (indices + 2).clamp_max(seq.size(1) - 1)])
                      .abs().sum(-1) > 0.05).float() * o['mask'][:, None]
            loss, _ = original(o['raw'], o['stability'], o['mask'], o['truth'], stable, o['mask'],
                               w_stab=1.0, w_pose=10.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % 5 == 0 or epoch == PROTOCOL['budgets']['cophynet_epochs'] - 1:
            model.eval()
            with torch.no_grad():
                mse = []
                for ix in torch.arange(len(val['x']), device=device).split(32):
                    d = subset(val, ix)
                    err = cophy_geo_predict(model, d)['pose'] - d['pose'][..., :3]
                    mse.append(float(((err.square().sum(-1) * d['role'][:, None, None, :, 0]).sum()
                                      / (d['role'][..., 0].sum() * err.size(1) * err.size(2))).sqrt()))
            history.append(dict(epoch=epoch, validation_pose_batch_rmse=float(np.mean(mse)),
                                seconds=round(time.monotonic() - start, 1)))
            print(domain, 'cophynet_geo', history[-1], flush=True)
            (OUT / f'cophynet_geo-{domain}-fit.json').write_text(json.dumps(history, indent=2) + '\n')
    torch.save(model.state_dict(), OUT / f'cophynet_geo-{domain}.pt')


# --- 2. PETS-Geo: identical MBRL-Lib 7-member PETS reading x_geo + geom ---
def pets_geo_context(d):
    b, k, n = d['a'].shape[:3]
    fixed = torch.cat([d['fact'].flatten(1), d['fa'].flatten(1), d['role'].flatten(1),
                       d['geom'].flatten(1), d['x'].flatten(1), d['goal']], -1)
    cmd = torch.cat([fixed[:, None].expand(-1, k, -1), d['a'].flatten(2)], -1)
    tm = clock(d['widths'])[:, None].expand(-1, k, -1, -1)
    return torch.cat([cmd[:, :, None].expand(-1, -1, tm.size(2), -1), tm], -1)


def pets_geo_batch(d):
    from mbrl.types import TransitionBatch
    pose = d['pose'].flatten(-2)
    initial = (d['x'][:, None] + d['reset']).flatten(-2)
    obs = torch.cat([initial[:, :, None], pose[:, :, :-1]], 2)
    kwargs = tuple(v.flatten(0, 2).cpu().numpy() for v in (obs, pets_geo_context(d), pose, d['reward']))
    empty = np.zeros(len(kwargs[0]), dtype=bool)
    return TransitionBatch(*kwargs, empty, empty)


def pets_geo_predict(model, d, rng):
    b, k, n = d['a'].shape[:3]
    particles = 20
    initial = (d['x'][:, None] + d['reset']).flatten(-2)
    position = initial[:, :, None].expand(-1, -1, particles, -1).reshape(-1, n * 9)
    state = model.reset(position, rng=rng)
    results = []
    for cmd in pets_geo_context(d).unbind(2):
        cmd = cmd[:, :, None].expand(-1, -1, particles, -1).flatten(0, 2)
        pose, reward, _, state = model.sample(cmd, state, deterministic=False, rng=rng)
        results.append((pose.reshape(b, k, particles, n, 9),
                        None if reward is None else reward.reshape(b, k, particles)))
    samples = torch.stack([p for p, _ in results], 3)
    reward = torch.stack([r for _, r in results], -1) if model.learned_rewards else None
    return dict(pose=samples.mean(2), draws=samples,
                reward=reward.mean(2) if reward is not None else None, reward_draws=reward)


def train_pets_geo(domain):
    from mbrl.models import ModelTrainer
    from mbrl.util.replay_buffer import BootstrapIterator
    device = 'cuda'
    data = pets_geo_batch(load(domain, 'train', labels=False))
    rewards = domain not in ('cophy', 'ballscf')
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
        print(domain, 'pets_geo', log[-1], flush=True)
        (OUT / f'pets_geo-{domain}-fit.json').write_text(json.dumps(log, indent=2) + '\n')

    trainer.train(batches, num_epochs=PROTOCOL['budgets']['pets_epochs'], patience=25,
                  callback=record, silent=True)
    norm = model.input_normalizer
    torch.save(dict(model=model.state_dict(), mean=norm.mean, std=norm.std,
                    elites=[int(i) for i in model.model.elite_models],
                    in_size=in_size, out_size=out_size, learned_rewards=rewards),
               OUT / f'pets_geo-{domain}.pt')


# --- 3. NCII-Geo: identical PointNet NCII reading x_geo + geom ---
class PointEmbeddingGeo(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.object_embedding = nn.Linear(9 + 9 + 3 + 6, HIDDEN)
        self.action_embedding = nn.Linear(n * 7 + 2, HIDDEN)
        self.source = nn.Linear(HIDDEN, HIDDEN)
        self.destination = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x, evidence, role, geom, action, time):
        own = F.leaky_relu(self.object_embedding(torch.cat([x, evidence, role, geom], -1)))
        act = F.leaky_relu(self.action_embedding(torch.cat([action.flatten(1), time], -1)))[:, None]
        src = self.source(torch.cat([own, act], 1))
        dst = self.destination(own)
        return own, F.leaky_relu(src[:, :, None] + dst[:, None])


class NCIIGeo(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.embed = PointEmbeddingGeo(n)
        self.forward_model = point_mlp(2 * HIDDEN, 18)
        self.interaction_embed = PointEmbeddingGeo(n)
        self.interaction = point_mlp(HIDDEN, 1)
        self.register_buffer('scale', torch.ones(9))

    def distribution(self, own, peers):
        mu, v = self.forward_model(torch.cat([own, peers], -1)).chunk(2, -1)
        return mu, F.softplus(v) + 0.001

    def logits(self, *args):
        _, pair = self.interaction_embed(*args)
        return self.interaction(pair)[..., 0]


def ncii_geo_inputs(d):
    k = d['a'].size(1)
    evidence = (d['fact'][:, -1] - d['fact'][:, -2])[:, None].expand(-1, k, -1, -1)
    role = d['role'][:, None].expand(-1, k, -1, -1)
    geom = d['geom'][:, None].expand(-1, k, -1, -1)
    times = clock(d['widths'])[:, None].expand(-1, k, -1, -1)
    return evidence, role, geom, d['a'], times


def ncii_geo_predict(model, d):
    b, k, n = d['a'].shape[:3]
    evidence, role, geom, act, t = ncii_geo_inputs(d)
    evidence, role, geom, act = [v.flatten(0, 1) for v in (evidence, role, geom, act)]
    valid = valid_sources(role)[..., None]
    x = (d['x'][:, None] + d['reset']).flatten(0, 1)
    logits = []
    for tm in t.flatten(0, 1).unbind(1):
        args = (x, evidence, role, geom, act, tm)
        logits.append(model.logits(*args)[:, :n].unflatten(0, (b, k)))
        own, pair = model.embed(*args)
        mu, _ = model.distribution(own, max_peers(pair, valid))
        x = x + mu * model.scale
    edges = torch.stack(logits, 2)
    return dict(edge=edges, causal=edges.cummax(2).values)


def train_ncii_geo(domain):
    dev = 'cuda'
    d = load(domain, 'train', dev, labels=False)
    evidence, role, geom, act, t = ncii_geo_inputs(d)
    old = torch.cat([(d['x'][:, None] + d['reset'])[:, :, None], d['pose'][:, :, :-1]], 2)
    expand = lambda v: v[:, :, None].expand(-1, -1, old.size(2), -1, -1).flatten(0, 2)
    data = (old.flatten(0, 2), expand(evidence), expand(role), expand(geom), expand(act),
            t.flatten(0, 2), (d['pose'] - old).flatten(0, 2))
    n = d['x'].size(1)
    model = NCIIGeo(n).to(dev)
    model.scale.copy_(data[-1][data[2].sum(-1).bool()].std(0).clamp_min(0.08))
    of = torch.optim.Adam([*model.embed.parameters(), *model.forward_model.parameters()], lr=1e-4, fused=True)
    oh = torch.optim.Adam([*model.interaction_embed.parameters(), *model.interaction.parameters()], lr=1e-4, fused=True)
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
            (OUT / f'ncii_geo-{domain}-fit.json').write_text(json.dumps(histories, indent=2) + '\n')
            print(domain, 'ncii_geo', rec, flush=True)
    torch.save(dict(model=model.state_dict(), n=n), OUT / f'ncii_geo-{domain}.pt')


def restore_cophy_geo(domain, device):
    net = CoPhyNetGeo().to(device)
    net.load_state_dict(torch.load(OUT / f'cophynet_geo-{domain}.pt', map_location=device))
    return net.eval()


def restore_pets_geo(domain, device):
    ck = torch.load(OUT / f'pets_geo-{domain}.pt', map_location=device)
    model = pets_build(ck['in_size'], ck['out_size'], device, ck['learned_rewards'])
    model.load_state_dict(ck['model'])
    model.input_normalizer.mean = ck['mean']
    model.input_normalizer.std = ck['std']
    model.model.set_elite(ck['elites'])
    return model.eval()


def restore_ncii_geo(domain, device):
    st = torch.load(OUT / f'ncii_geo-{domain}.pt', map_location=device)
    net = NCIIGeo(st['n']).to(device)
    net.load_state_dict(st['model'])
    return net.eval()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('name', choices=('cophynet_geo', 'pets_geo', 'ncii_geo'))
    p.add_argument('domain', choices=ALL_DOMAINS)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(PROTOCOL['seed'])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    {'cophynet_geo': train_cophy_geo, 'pets_geo': train_pets_geo, 'ncii_geo': train_ncii_geo}[args.name](args.domain)
