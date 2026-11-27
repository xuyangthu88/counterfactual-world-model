"""Train Stage 32 DeltaGraph variants with epoch-balanced sampling, direct impulse-contrast loss, and EMA."""
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
from whatif.stabilized.fit import domain_rms_scales, robust_sq, update_ema
from .model import build_deltagraph

OUT = ROOT / 'results/deltagraph'
PROTOCOL = json.loads(Path(__file__).with_name('protocol.json').read_text())


def add_pose_jitter(batch, std):
    """Mild domain-general Gaussian jitter on initial/factual xyz during training for transfer robustness."""
    if std <= 0:
        return batch
    x = batch['x']
    apply = (torch.rand(len(x), 1, 1, device=x.device) < 0.5).to(x)
    present = batch['role'].sum(-1, keepdim=True)
    noise_x = torch.randn_like(x[..., :3]) * std * apply * present
    noise_f = torch.randn_like(batch['fact'][..., :3]) * std * apply[:, None] * present[:, None]
    out = dict(batch)
    out['x'] = torch.cat([x[..., :3] + noise_x, x[..., 3:]], dim=-1)
    out['fact'] = torch.cat([batch['fact'][..., :3] + noise_f, batch['fact'][..., 3:]], dim=-1)
    return out


def deltagraph_loss(o, d, domain_scale, beta=4.5):
    i, j = paired_actions(d['a'])
    movable = d['role'][:, None, None, :, 0:1]
    actor = d['role'][:, None, None, :, 1:2]
    active = 3.0 * movable + 1.5 * actor
    edge = labeled(d['role'])[:, None, None]
    last = d['widths'].new_tensor([0.7, 0.7, 1.0, 1.2, 1.4])[None, None, :, None, None]

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
        + 1.05 * average(robust_sq(de[..., :3], beta) + 0.12 * de[..., :3].abs(), move_w * last)
        + 0.60 * average(robust_sq(de[..., :3], beta), actor * last)
        + 0.20 * average(de[..., 3:].square(), move_w * last)
    )

    # Full-horizon counterfactual-minus-factual contrast when same-start factual rollout is observed (CoPhy)
    if 'off' in d:
        fc_err = (o['pose'][..., :3] - o['ref_pose'][..., :3]) - (
            d['pose'][..., :3] - d['fact'][:, None, 1:, :, :3]
        )
        ref_err = o['ref_pose'][:, 0, ..., :3] - d['fact'][:, 1:, :, :3]
        motion_coarse = (
            motion_coarse
            + 0.45 * average(robust_sq(fc_err, beta), movable * last)
            + 0.35 * average(robust_sq(ref_err, beta), movable[:, 0] * last[:, 0])
        )

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

    # Native short-step + direct motion-difference & impulse-difference propagation supervision
    nat = o['native']
    n_err = nat['pose'] - d['native_pose']
    n_pos = n_err[..., :3]
    n_delta = n_pos[:, i] - n_pos[:, j]
    dt_w = d['native_widths'][:, None, :, None, None].clamp_min(1e-3)
    n_vel_diff = step_delta(n_delta) / dt_w.sqrt()
    # Second difference = net step-wise impulse contrast directly training graph propagation!
    n_imp_diff = step_delta(step_delta(n_delta) / dt_w)
    n_edge_true = d['native_edge'].to(n_pos)
    n_ce = F.binary_cross_entropy_with_logits(nat['edge'], n_edge_true, reduction='none')
    n_change = (n_edge_true[:, :, 1:] - n_edge_true[:, :, :-1]).abs()
    n_discord = (n_edge_true[:, i] - n_edge_true[:, j]).abs()
    n_affected = (n_discord.any(-2) | n_discord.any(-1)).to(n_pos)
    n_affected_cum = n_affected.cummax(dim=2).values
    n_move_w = movable * (1.0 + 0.6 * receiver + 0.8 * n_affected_cum[..., None])
    n_imp_w = movable * (1.0 + 0.8 * receiver + 1.2 * n_affected[..., None])

    motion_native = (
        0.40 * average(robust_sq(n_pos, beta), 2.5 * movable + 1.5 * actor)
        + 0.09 * average(n_err[..., 3:].square(), 2.5 * movable + 1.0 * actor)
        + 0.70 * average(robust_sq(n_delta, beta), n_move_w)
        + 0.55 * average(robust_sq(n_vel_diff, beta), n_move_w)
        + 0.35 * average(robust_sq(n_imp_diff, beta), n_imp_w)
        + 0.45 * average(robust_sq(n_delta, beta), actor)
        + 0.35 * average(robust_sq(n_vel_diff, beta), actor)
    )
    graph_native = (
        0.34 * average(n_ce, edge)
        + 0.18 * average(n_ce[:, :, 1:], edge * n_change)
        + 0.16 * average(0.5 * (n_ce[:, i] + n_ce[:, j]), edge * n_discord)
    )
    loss = loss + inv_s2 * motion_native + graph_native
    if 'native_reward' in d:
        ne = nat['reward'] - d['native_reward']
        loss = loss + 0.18 * (ne.square().mean() + (ne[:, i] - ne[:, j]).square().mean())
    return loss


def model_domains(kind):
    return DOMAINS[:2] if kind == 'deltagraph_unseen' else DOMAINS


def fit(kind):
    dev = 'cuda'
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(PROTOCOL['seed'])
    domains = model_domains(kind)

    train_data = {dom: load(dom, 'train', dev) for dom in domains}
    valid_data = {dom: load(dom, 'val', dev, labels=False) for dom in domains}
    scales = domain_rms_scales(train_data)
    max_n = max(len(d['x']) for d in train_data.values())
    base_bs = PROTOCOL['base_batch_size']
    # Epoch-balanced batch sizes purely proportional to dataset size (12 for 768, 24 for 1536)
    batch_sizes = {dom: max(12, min(base_bs, int(round(base_bs * len(d['x']) / max_n)))) for dom, d in train_data.items()}
    print(f'{kind} domain_scales={scales} batch_sizes={batch_sizes}', flush=True)

    ref_name = 'cumulative_unseen-calibrated' if kind == 'deltagraph_unseen' else 'continued-calibrated'
    ref_model = restore_stage30(ref_name, dev)
    ref_val = validation(ref_model, valid_data)
    del ref_model

    model = build_deltagraph(kind, PROTOCOL).to(dev)
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
    jitter = PROTOCOL['pose_noise_std']

    for step in range(1, steps + 1):
        model.train()
        soft = max(0.0, 0.85 * (1.0 - (step - 1) / 400.0))
        opt.zero_grad(set_to_none=True)
        values = []
        for dom, d in train_data.items():
            bs = batch_sizes[dom]
            batch = subset(d, torch.randint(len(d['x']), (bs,), device=dev))
            batch = rotate_layouts(batch, rot_p)
            batch = add_pose_jitter(batch, jitter)
            removed, target = paired_removal(batch)
            o = model(batch, removed, soft=soft, native=True, detach_dist=True)
            loss = deltagraph_loss(o, batch, scales[dom], beta=beta)
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
                    batch_sizes=batch_sizes,
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
            print(kind, dict(step=step, loss=round(record['loss'], 3),
                             val={k: round(v, 4) for k, v in metrics.items()},
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
        # Full-convergence calibration on frozen backbone (prevents wide initial reward scale)
        from whatif.effects.calibrate import loss as cal_loss
        dev = 'cuda'
        doms = model_domains(args.kind)
        ckpt = torch.load(OUT / f'{args.kind}.pt', map_location=dev)
        model = build_deltagraph(args.kind, PROTOCOL).to(dev)
        model.load_state_dict(ckpt['model'])
        model.eval().requires_grad_(False)
        for head in (model.position_dist, model.reward_dist, model.regime):
            head.requires_grad_(True)
        cal_steps = 1200
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0022, fused=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cal_steps, eta_min=0.0002)
        tr = {d: load(d, 'train', dev, labels=False) for d in doms}
        va = {d: load(d, 'val', dev, labels=False) for d in doms}
        log = []
        best = float('inf')
        for st in range(1, cal_steps + 1):
            rows = [subset(d, torch.randint(len(d['x']), (40,), device=dev)) for d in tr.values()]
            obj = torch.stack([cal_loss(model, d) for d in rows]).mean()
            opt.zero_grad(set_to_none=True)
            obj.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 4.0)
            opt.step()
            sched.step()
            if st % 150 == 0 or st == cal_steps:
                with torch.no_grad():
                    vals = {d: float(cal_loss(model, b)) for d, b in va.items()}
                val_mean = sum(vals.values()) / len(vals)
                if val_mean < best:
                    best = val_mean
                    torch.save(dict(ckpt, model=model.state_dict(), calibration_step=st), OUT / f'{args.kind}-calibrated.pt')
                log.append(dict(step=st, validation=vals, value=val_mean))
                print(args.kind, log[-1], flush=True)
                (OUT / f'{args.kind}-calibration.json').write_text(json.dumps(log, indent=2) + '\n')
    else:
        fit(args.kind)
