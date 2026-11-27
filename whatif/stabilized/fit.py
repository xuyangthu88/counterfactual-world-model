"""Train Stage 31 stabilized shared world model variants from scratch with EMA and calibration."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from whatif.cumulative.evaluate import restore as restore_stage30
from whatif.effects.collect import DOMAINS, ROOT
from whatif.effects.data import subset
from whatif.effects.fit import directed_truth, paired_actions, validation
from whatif.effects.model import average
from whatif.mechanism.model import labeled, pairmask
from whatif.shortstep.augment import rotate_layouts
from whatif.shortstep.calibrate import calibrate
from whatif.shortstep.data import load, paired_removal
from whatif.shortstep.fit import edit_loss, step_delta
from .model import build_stabilized

OUT = ROOT / 'results/stabilized'
PROTOCOL = json.loads(Path(__file__).with_name('protocol.json').read_text())


def robust_sq(e, beta=4.5):
    """Exact e^2 for |e| < beta, linear growth above beta to suppress tunneling outliers."""
    return 2.0 * beta * F.smooth_l1_loss(e, torch.zeros_like(e), beta=beta, reduction='none')


def stabilized_loss(o, d, domain_scale, beta=4.5):
    i, j = paired_actions(d['a'])
    movable = d['role'][:, None, None, :, 0:1]
    actor = d['role'][:, None, None, :, 1:2]
    active = 3.0 * movable + 1.2 * actor
    edge = labeled(d['role'])[:, None, None]
    last = d['widths'].new_tensor([0.7, 0.7, 1.0, 1.2, 1.4])[None, None, :, None, None]

    # Coarse 5-window loss (static walls excluded from position denominator; actor explicitly supervised)
    err = o['pose'] - d['pose']
    de = err[:, i] - err[:, j]
    receiver = (d['a'][:, i, :, 0] + d['a'][:, j, :, 0] == 0)[:, :, None, :, None]
    discord = (d['edge'][:, i] - d['edge'][:, j]).abs()
    affected = discord.any(-2) | discord.any(-1)
    move_w = movable * (1.0 + 0.7 * receiver + 0.8 * affected[..., None])

    contact = F.binary_cross_entropy_with_logits(o['edge'], d['edge'], reduction='none')
    change = (d['edge'][:, :, 1:] - d['edge'][:, :, :-1]).abs()

    inv_s2 = 1.0 / (domain_scale ** 2)
    motion_coarse = (
        average(robust_sq(err[..., :3], beta) + 0.12 * err[..., :3].abs(), active * last)
        + 0.30 * average(err[..., 3:].square(), active * last)
        + 0.95 * average(robust_sq(de[..., :3], beta) + 0.12 * de[..., :3].abs(), move_w * last)
        + 0.45 * average(robust_sq(de[..., :3], beta), actor * last)
        + 0.20 * average(de[..., 3:].square(), move_w * last)
    )
    # Auxiliary factual-contrast loss when same-start factual continuation exists (CoPhy)
    if 'off' in d:
        fc_err = (o['pose'][:, :, -1, :, :3] - o['ref_pose'][:, :, -1, :, :3]) - (
            d['pose'][:, :, -1, :, :3] - d['fact'][:, None, -1, :, :3]
        )
        motion_coarse = motion_coarse + 0.35 * average(robust_sq(fc_err, beta), movable[:, :, 0])

    graph_coarse = (
        0.58 * average(contact, edge)
        + 0.25 * average(contact[:, :, 1:], edge * change)
        + 0.20 * average(0.5 * (contact[:, i] + contact[:, j]), edge * discord)
        + 0.10 * average(F.binary_cross_entropy_with_logits(o['graph'], d['fe'], reduction='none'), edge[:, 0, 0])
        + 0.004 * average(o['edge'].sigmoid(), pairmask(d['role'])[:, None, None])
    )
    fact_loss = (
        0.10 * average(F.smooth_l1_loss(o['innovation'], torch.zeros_like(o['innovation']), beta=0.15, reduction='none'),
                       d['role'].sum(-1)[:, None, :, None])
        + 0.002 * o['kl']
        - 0.015 * o['mech_entropy']
    )

    loss = inv_s2 * motion_coarse + graph_coarse + fact_loss

    if 'off' in d:
        true_dir = directed_truth(d)
        loss = loss + 0.40 * average(F.binary_cross_entropy_with_logits(o['causal'], (true_dir > 0.1).to(true_dir),
                                                                        reduction='none'), edge)
        loss = loss + 0.10 * average(F.smooth_l1_loss(o['strength'], true_dir.log1p(), beta=0.15, reduction='none'), edge)
    else:
        rew = o['reward'] - d['reward']
        loss = loss + rew.square().mean() + 0.8 * (rew[:, i] - rew[:, j]).square().mean()

    # Native short-step loss with motion-difference propagation emphasis
    nat = o['native']
    n_err = nat['pose'] - d['native_pose']
    n_pos = n_err[..., :3]
    n_delta = n_pos[:, i] - n_pos[:, j]
    n_speed = step_delta(n_delta) / d['native_widths'][:, None, :, None, None].sqrt()
    n_edge_true = d['native_edge'].to(n_pos)
    n_ce = F.binary_cross_entropy_with_logits(nat['edge'], n_edge_true, reduction='none')
    n_change = (n_edge_true[:, :, 1:] - n_edge_true[:, :, :-1]).abs()
    n_discord = (n_edge_true[:, i] - n_edge_true[:, j]).abs()
    n_affected = (n_discord.any(-2) | n_discord.any(-1)).to(n_pos)
    # Cumulative contact disagreement so downstream coasting after a contact difference is emphasized
    n_affected_cum = n_affected.cummax(dim=2).values
    n_move_w = movable * (1.0 + 0.6 * receiver + 0.7 * n_affected_cum[..., None])

    motion_native = (
        0.40 * average(robust_sq(n_pos, beta), 2.5 * movable + 1.0 * actor)
        + 0.09 * average(n_err[..., 3:].square(), 2.5 * movable + 1.0 * actor)
        + 0.65 * average(robust_sq(n_delta, beta), n_move_w)
        + 0.55 * average(robust_sq(n_speed, beta), n_move_w)
        + 0.30 * average(robust_sq(n_delta, beta), actor)
        + 0.25 * average(robust_sq(n_speed, beta), actor)
    )
    graph_native = (
        0.31 * average(n_ce, edge)
        + 0.17 * average(n_ce[:, :, 1:], edge * n_change)
        + 0.14 * average(0.5 * (n_ce[:, i] + n_ce[:, j]), edge * n_discord)
    )
    loss = loss + inv_s2 * motion_native + graph_native
    if 'native_reward' in d:
        ne = nat['reward'] - d['native_reward']
        loss = loss + 0.18 * (ne.square().mean() + (ne[:, i] - ne[:, j]).square().mean())
    return loss


def domain_rms_scales(train_data):
    """Compute per-domain reference scale from training terminal action-diff RMS (clamped)."""
    scales = {}
    for dom, d in train_data.items():
        i, j = paired_actions(d['a'])
        mover = d['role'][:, None, :, 0].bool()
        diff = d['pose'][:, i, -1, :, :3] - d['pose'][:, j, -1, :, :3]
        sq = diff.square().sum(-1)[mover.expand_as(diff[..., 0])]
        # Clamp extreme outliers when computing scale
        rms = float(sq.clamp_max(25.0).mean().sqrt())
        scales[dom] = max(rms, 0.35)
    # Normalize around geometric mean so overall loss magnitude stays ~1.0
    gmean = float(np.exp(np.mean([np.log(v) for v in scales.values()])))
    return {k: v / gmean for k, v in scales.items()}


def model_domains(kind):
    return DOMAINS[:2] if kind == 'stabilized_unseen' else DOMAINS


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.mul_(decay).add_(p, alpha=1.0 - decay)


def fit(kind):
    dev = 'cuda'
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(PROTOCOL['seed'])
    domains = model_domains(kind)

    train_data = {dom: load(dom, 'train', dev) for dom in domains}
    valid_data = {dom: load(dom, 'val', dev, labels=False) for dom in domains}
    scales = domain_rms_scales(train_data)
    print(f'{kind} domain_scales={scales}', flush=True)

    # Validation normalization reference: Stage 30 continued-calibrated on original val layouts
    ref_name = 'cumulative_unseen-calibrated' if kind == 'stabilized_unseen' else 'continued-calibrated'
    ref_model = restore_stage30(ref_name, dev)
    ref_val = validation(ref_model, valid_data)
    del ref_model

    model = build_stabilized(kind, PROTOCOL).to(dev)
    ema_model = copy.deepcopy(model).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{kind} params={n_params:,} ref_val={ref_val}', flush=True)

    steps = PROTOCOL['variants'][kind]['steps']
    opt = torch.optim.AdamW(model.parameters(), lr=PROTOCOL['lr'], weight_decay=PROTOCOL['weight_decay'], fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=PROTOCOL['eta_min'])

    best = float('inf')
    history = []
    start = time.monotonic()
    OUT.mkdir(exist_ok=True)
    beta = PROTOCOL['robust_beta']
    rot_p = PROTOCOL['rotated_layout_probability']
    bs = PROTOCOL['batch_size']

    for step in range(1, steps + 1):
        model.train()
        # Smooth 400-step annealing from soft=0.85 to sparse STE gate soft=0.0
        soft = max(0.0, 0.85 * (1.0 - (step - 1) / 400.0))
        opt.zero_grad(set_to_none=True)
        values = []
        for dom, d in train_data.items():
            batch = subset(d, torch.randint(len(d['x']), (bs,), device=dev))
            batch = rotate_layouts(batch, rot_p)
            removed, target = paired_removal(batch)
            o = model(batch, removed, soft=soft, native=True, detach_dist=True)
            loss = stabilized_loss(o, batch, scales[dom], beta=beta)
            # Paired collision-removal supervision scaled consistently
            e_loss = edit_loss(o['native']['edge_effect'][:, :, 0], target, removed, batch, batch['native_widths'])
            loss = loss + e_loss / scales[dom]
            (loss / len(domains)).backward()
            values.append(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        sched.step()
        ema_decay = min(PROTOCOL['ema_decay'], (1.0 + step) / (10.0 + step))
        update_ema(ema_model, model, ema_decay)

        if step % 150 == 0 or step == steps:
            metrics_raw = validation(model, valid_data)
            metrics_ema = validation(ema_model, valid_data)
            crit_raw = float(np.mean([v / ref_val[k] for k, v in metrics_raw.items()]))
            crit_ema = float(np.mean([v / ref_val[k] for k, v in metrics_ema.items()]))
            use_ema = crit_ema <= crit_raw
            criterion = crit_ema if use_ema else crit_raw
            metrics = metrics_ema if use_ema else metrics_raw
            if criterion < best:
                best = criterion
                chosen = ema_model if use_ema else model
                torch.save(dict(
                    model=chosen.state_dict(),
                    step=step,
                    used_ema=use_ema,
                    domains=domains,
                    kind=kind,
                    domain_scales=scales,
                ), OUT / f'{kind}.pt')
            record = dict(
                step=step,
                loss=float(torch.stack(values).mean()),
                validation=metrics,
                validation_raw=metrics_raw,
                validation_ema=metrics_ema,
                ref_validation=ref_val,
                criterion=criterion,
                used_ema=use_ema,
                seconds=round(time.monotonic() - start, 1),
            )
            history.append(record)
            print(kind, dict(step=step, loss=round(record['loss'], 3), val={k: round(v, 4) for k, v in metrics.items()},
                             crit=round(criterion, 4), ema=use_ema, sec=record['seconds']), flush=True)
            (OUT / f'{kind}-fit.json').write_text(json.dumps(history, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=tuple(PROTOCOL['variants']))
    p.add_argument('--calibrate', action='store_true')
    args = p.parse_args()
    if args.calibrate:
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        torch.manual_seed(PROTOCOL['seed'])
        calibrate(
            args.kind,
            build_stabilized(args.kind, PROTOCOL),
            domains=model_domains(args.kind),
            directory=OUT,
            steps=PROTOCOL['calibration_steps'],
        )
    else:
        fit(args.kind)
