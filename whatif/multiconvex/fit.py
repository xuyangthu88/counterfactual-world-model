"""Stage 35: Train MultiConvexDeltaGraphWorld on universal M=2 multi-convex SAT + torque protocol."""
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from whatif.benchmarks.data import ALL_DOMAINS
from whatif.deltagraph.fit import add_pose_jitter, deltagraph_loss
from whatif.effects.calibrate import loss as cal_loss
from whatif.effects.collect import ROOT
from whatif.effects.data import subset
from whatif.effects.fit import paired_actions, validation
from whatif.effects.model import average
from whatif.shortstep.augment import rotate_layouts
from whatif.shortstep.data import paired_removal
from whatif.shortstep.fit import edit_loss
from whatif.stabilized.fit import domain_rms_scales, update_ema
from .data import load
from .model import MultiConvexDeltaGraphWorld

OUT = ROOT / 'results/multiconvex'
PROTOCOL = json.loads((ROOT / 'whatif/multiconvex/protocol.json').read_text())

CONFIGS = {
    'mc_shared3_hop3': dict(domains=('cophy', 'causalworld', 'pusht'),
                            use_delta_prop=True, use_exo=True, hops=3, width=40, feature=128, dropout=0.10, steps=3900),
    'mc_shared3_clean': dict(domains=('cophy', 'causalworld', 'pusht'),
                             use_delta_prop=True, use_exo=True, hops=3, width=40, feature=128, dropout=0.10, steps=3600),
    'mc_shared5_hop3': dict(domains=ALL_DOMAINS,
                            use_delta_prop=True, use_exo=True, hops=3, width=40, feature=128, dropout=0.10, steps=3600),
    'mc_shared5_noexo': dict(domains=ALL_DOMAINS,
                             use_delta_prop=True, use_exo=False, hops=3, width=40, feature=128, dropout=0.10, steps=3600),
    'mc_unseen_pusht4': dict(domains=('cophy', 'ballscf', 'causalworld', 'stacking2'),
                             use_delta_prop=True, use_exo=True, hops=2, width=32, feature=96, dropout=0.08, steps=3000),
    'mc_unseen_stacking2': dict(domains=('cophy', 'ballscf', 'causalworld', 'pusht'),
                                use_delta_prop=True, use_exo=True, hops=2, width=32, feature=96, dropout=0.08, steps=3000),
}


def build_model(kind):
    cfg = CONFIGS[kind]
    return MultiConvexDeltaGraphWorld(
        use_delta_prop=cfg['use_delta_prop'],
        use_exo=cfg['use_exo'],
        hops=cfg['hops'],
        width=cfg['width'],
        feature=cfg['feature'],
        dropout=cfg['dropout'],
    )


def multiconvex_loss(o, d, domain_scale, beta=4.5):
    """Pure Bayes-optimal DeltaGraph loss (no magnitude-inflation term that overfits stochastic outliers)."""
    return deltagraph_loss(o, d, domain_scale, beta=beta)


def fit(kind):
    dev = 'cuda'
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(PROTOCOL['seed'])
    cfg = CONFIGS[kind]
    domains = cfg['domains']

    train_data = {dom: load(dom, 'train', dev) for dom in domains}
    valid_data = {dom: load(dom, 'val', dev, labels=False) for dom in domains}
    scales = domain_rms_scales(train_data)
    val_scales = domain_rms_scales(valid_data)
    max_n = max(len(d['x']) for d in train_data.values())
    base_bs = 20
    batch_sizes = {dom: max(10, min(base_bs, int(round(base_bs * len(d['x']) / max_n))))
                   for dom, d in train_data.items()}
    print(f'{kind} domains={domains} scales={scales} batch_sizes={batch_sizes}', flush=True)

    model = build_model(kind).to(dev)
    ema_model = copy.deepcopy(model).eval()
    print(f'{kind} params={sum(p.numel() for p in model.parameters()):,}', flush=True)

    steps = cfg['steps']
    opt = torch.optim.AdamW(model.parameters(), lr=0.0010, weight_decay=0.015, fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.00010)

    best = float('inf')
    history = []
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    for step in range(1, steps + 1):
        model.train()
        soft = max(0.0, 0.85 * (1.0 - (step - 1) / 400.0))
        opt.zero_grad(set_to_none=True)
        values = []
        for dom, d in train_data.items():
            bs = batch_sizes[dom]
            batch = subset(d, torch.randint(len(d['x']), (bs,), device=dev))
            # parts & part_mask are strictly in each object's body frame, invariant under world SO(2) rotation!
            geom_save = batch['geom']
            parts_save = batch['parts']
            pmask_save = batch['part_mask']
            batch = rotate_layouts(batch, 0.5)
            batch['geom'] = geom_save
            batch['parts'] = parts_save
            batch['part_mask'] = pmask_save
            batch = add_pose_jitter(batch, 0.003)
            removed, target = paired_removal(batch)
            o = model(batch, removed, soft=soft, native=True, detach_dist=True)
            loss = multiconvex_loss(o, batch, scales[dom], beta=4.5)
            e_loss = edit_loss(o['native']['edge_effect'][:, :, 0], target, removed, batch, batch['native_widths'])
            loss = loss + e_loss / scales[dom]
            (loss / len(domains)).backward()
            values.append(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        sched.step()
        ema_decay = min(0.995, (1.0 + step) / (10.0 + step))
        update_ema(ema_model, model, ema_decay)

        if step % 150 == 0 or step == steps:
            metrics_raw = validation(model, valid_data)
            metrics_ema = validation(ema_model, valid_data)
            crit_raw = float(np.mean([v / val_scales[k] for k, v in metrics_raw.items()]))
            crit_ema = float(np.mean([v / val_scales[k] for k, v in metrics_ema.items()]))
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
                    domains=list(domains),
                    kind=kind,
                    domain_scales=scales,
                    val_scales=val_scales,
                ), OUT / f'{kind}.pt')
            record = dict(
                step=step,
                loss=float(torch.stack(values).mean()),
                validation=metrics,
                criterion=criterion,
                used_ema=use_ema,
                seconds=round(time.monotonic() - start, 1),
            )
            history.append(record)
            print(kind, dict(step=step, loss=round(record['loss'], 3),
                             val={k: round(v, 4) for k, v in metrics.items()},
                             crit=round(criterion, 4), ema=use_ema, sec=record['seconds']), flush=True)
            (OUT / f'{kind}-fit.json').write_text(json.dumps(history, indent=2) + '\n')


def calibrate_model(kind):
    dev = 'cuda'
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(PROTOCOL['seed'])
    domains = CONFIGS[kind]['domains']
    ckpt = torch.load(OUT / f'{kind}.pt', map_location=dev)
    model = build_model(kind).to(dev)
    model.load_state_dict(ckpt['model'])
    model.eval().requires_grad_(False)
    for head in (model.position_dist, model.reward_dist, model.regime):
        head.requires_grad_(True)

    cal_steps = 1200
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0020, fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cal_steps, eta_min=0.0002)
    tr = {d: load(d, 'train', dev, labels=False) for d in domains}
    va = {d: load(d, 'val', dev, labels=False) for d in domains}
    log = []
    best = float('inf')
    for st in range(1, cal_steps + 1):
        rows = [subset(d, torch.randint(len(d['x']), (32,), device=dev)) for d in tr.values()]
        obj = torch.stack([cal_loss(model, d) for d in rows]).mean()
        if not torch.isfinite(obj):
            continue
        opt.zero_grad(set_to_none=True)
        obj.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 3.0)
        opt.step()
        sched.step()
        if st % 150 == 0 or st == cal_steps:
            with torch.no_grad():
                vals = {d: float(cal_loss(model, b)) for d, b in va.items()}
            val_mean = sum(vals.values()) / len(vals)
            if np.isfinite(val_mean) and val_mean < best:
                best = val_mean
                torch.save(dict(ckpt, model=model.state_dict(), calibration_step=st),
                           OUT / f'{kind}-calibrated.pt')
            log.append(dict(step=st, validation=vals, value=val_mean))
            print(kind, 'cal', log[-1], flush=True)
            (OUT / f'{kind}-calibration.json').write_text(json.dumps(log, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=tuple(CONFIGS))
    p.add_argument('--calibrate', action='store_true')
    args = p.parse_args()
    if args.calibrate:
        calibrate_model(args.kind)
    else:
        fit(args.kind)
