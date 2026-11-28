"""Stage 34: Evaluate universal rigid-body geometry protocol (fair comparison vs *_geo baselines)."""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score

import whatif.effects.evaluate as _eff_eval
import whatif.shortstep.evaluate as _ss_eval
from whatif.benchmarks.data import ALL_DOMAINS, NEW_DOMAINS, UNIT, native_path
from whatif.effects.data import subset
from whatif.effects.evaluate import combine, contact_graph, edits, prediction
from whatif.effects.fit import paired_actions
from whatif.shortstep.evaluate import native_metrics, pushing_edits
from .baselines import (
    cophy_geo_predict,
    ncii_geo_predict,
    pets_geo_predict,
    restore_cophy_geo,
    restore_ncii_geo,
    restore_pets_geo,
)
from .data import load
from .fit import CONFIGS, OUT, build_model

_eff_eval.UNIT.update(UNIT)
_ss_eval.UNIT.update(UNIT)
KEEP_KEYS = ('pose', 'edge', 'causal', 'reward', 'w', 'p_dist', 'r_dist')


def transfer_split_name(domain):
    return 'transfer' if domain in NEW_DOMAINS else 'cumulativetransfer'


def restore_world(name, device):
    kind = name.removesuffix('-calibrated')
    model = build_model(kind).to(device)
    model.load_state_dict(torch.load(OUT / f'{name}.pt', map_location=device)['model'])
    return model.eval()


@torch.no_grad()
def pusht_geo_diagnostics(model, device):
    d_pt = load('pusht', 'cumulativetransfer', device, labels=True)
    o_pt = model(d_pt, native=True)
    i, j = paired_actions(d_pt['a'])
    y_diff = d_pt['pose'][:, i, -1, 0, :3] - d_pt['pose'][:, j, -1, 0, :3]
    p_diff = o_pt['pose'][:, i, -1, 0, :3] - o_pt['pose'][:, j, -1, 0, :3]
    sq = (p_diff - y_diff).square().sum(-1)
    push_c = d_pt['native_edge'][..., 0, 1].bool()
    wall_c = d_pt['native_edge'][..., 0, 2:6].any(-1)
    pinch_br = (push_c & wall_c).any(-1)
    pinch_pair = pinch_br[:, i] | pinch_br[:, j]
    nat_y = d_pt['native_edge'].bool()
    nat_p = o_pt['native']['edge'].sigmoid()
    return dict(
        pusht_pinch_rmse=float(100.0 * sq[pinch_pair].mean().sqrt()),
        pusht_nopinch_rmse=float(100.0 * sq[~pinch_pair].mean().sqrt()),
        pusht_pusher_ap=float(average_precision_score(nat_y[..., 0, 1].flatten().cpu().numpy(),
                                                      nat_p[..., 0, 1].flatten().cpu().numpy())),
        pusht_wall_ap=float(average_precision_score(nat_y[..., 0, 2:6].flatten().cpu().numpy(),
                                                    nat_p[..., 0, 2:6].flatten().cpu().numpy())),
        pusht_actor_rmse_end=float(100.0 * (o_pt['native']['pose'][..., -1, 1, :2]
                                            - d_pt['native_pose'][..., -1, 1, :2]).square().sum(-1).mean().sqrt()),
    )


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--domains', nargs='+', default=list(ALL_DOMAINS))
    p.add_argument('--models', nargs='+')
    args = p.parse_args()

    avail = [f'{k}-calibrated' for k in CONFIGS if (OUT / f'{k}-calibrated.pt').exists()]
    models = args.models or [*avail, 'cophynet_geo', 'pets_geo', 'ncii_geo']
    device = args.device
    results = {}

    for domain in args.domains:
        tr = transfer_split_name(domain)
        splits = ('test', tr)
        sets = {s: load(domain, s, device, labels=False) for s in ('val', *splits)}
        results[domain] = {s: {} for s in splits}
        for name in models:
            if name == 'cophynet_geo':
                model = restore_cophy_geo(domain, device)
                flavor = 'cophynet'
                run = lambda d, m=model: {'pose': cophy_geo_predict(m, d)['pose'], 'reward': None}
            elif name == 'pets_geo':
                model = restore_pets_geo(domain, device)
                flavor = 'pets'
                torch.manual_seed(345)
                rng = torch.Generator(device=device).manual_seed(345)
                run = lambda d, m=model, r=rng: pets_geo_predict(m, d, r)
            elif name == 'ncii_geo':
                model = restore_ncii_geo(domain, device)
                flavor = 'ncii'
                run = lambda d, m=model: ncii_geo_predict(m, d)
            else:
                model = restore_world(name, device)
                flavor = 'ours'
                run = lambda d, m=model: {k: v for k, v in m(d).items() if k in KEEP_KEYS}

            def infer(d):
                return combine([run(subset(d, ix)) for ix in torch.arange(len(d['x']), device=device).split(16)])

            cuts = {}
            if flavor in ('ours', 'ncii'):
                val_out = infer(sets['val'])
                val_g = contact_graph(sets['val'], val_out, {})
                cuts = {k: v['cut'] for k, v in val_g.items() if isinstance(v, dict) and 'cut' in v}

            for split in splits:
                d = sets[split]
                o = infer(d)
                row = {}
                if flavor != 'ncii':
                    row.update(prediction(d, o, domain, flavor))
                if flavor in ('ours', 'ncii'):
                    row['graph'] = contact_graph(d, o, cuts)
                if flavor == 'ours':
                    if 'off' in d:
                        row['edit'] = edits(model, d)
                    elif native_path(domain, split).exists():
                        d_full = load(domain, split, device, labels=True)
                        row['edit'] = pushing_edits(model, d_full, domain)
                    if native_path(domain, split).exists():
                        d_full = load(domain, split, device, labels=True)
                        row['native'] = native_metrics(model, d_full, domain)
                results[domain][split][name] = row
                print(f'geo {domain:11s} {split:18s} {name:24s} delta={row.get("delta_rmse")} '
                      f'ap={row.get("graph", {}).get("contact", {}).get("ap")}', flush=True)

    diags = {}
    for name in avail:
        diags[name] = pusht_geo_diagnostics(restore_world(name, device), device)
        print('geo_diag', name, json.dumps(diags[name]), flush=True)

    summary = {}
    for domain in args.domains:
        summary[domain] = {}
        for split in ('test', transfer_split_name(domain)):
            tbl = results[domain][split]
            cp = tbl.get('cophynet_geo', {}).get('delta_rmse')
            pt = tbl.get('pets_geo', {}).get('delta_rmse')
            if cp is not None and pt is not None:
                best_b = min(cp, pt)
                best_n = 'cophynet_geo' if cp <= pt else 'pets_geo'
            else:
                best_b, best_n = None, None
            row_sum = {'best_geo_baseline': best_n, 'best_geo_baseline_delta_rmse': best_b}
            for name, m in tbl.items():
                if 'delta_rmse' in m and best_b:
                    row_sum[name] = dict(
                        delta_rmse=round(m['delta_rmse'], 4),
                        reduction_pct=round(100.0 * (best_b - m['delta_rmse']) / best_b, 2),
                        contact_ap=round(m['graph']['contact']['ap'], 4) if 'graph' in m and m['graph']['contact']['ap'] is not None else None,
                    )
            summary[domain][split] = row_sum

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (OUT / 'diagnostics.json').write_text(json.dumps(diags, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
