"""Evaluate Stage 31 stabilized variants, Stage 30 reference, frozen baselines, and deep diagnostics."""
import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from torch.nn import functional as F

from whatif.cumulative.evaluate import restore as restore_stage30
from whatif.effects.collect import DOMAINS
from whatif.effects.data import UNIT, subset
from whatif.effects.evaluate import evaluate, root_mean
from whatif.effects.fit import paired_actions
from whatif.mechanism.model import pool
from whatif.shortstep.data import load
from whatif.shortstep.evaluate import native_metrics, pushing_edits
from .fit import OUT, PROTOCOL
from .model import build_stabilized

EARLIER = ('shared-calibrated', 'continued-calibrated', 'cumulative_unseen-calibrated')
TRAINED = tuple(PROTOCOL['variants'])
REFERENCES = ('cophynet', 'pets', 'ncii')


def restore(name, device):
    kind = name.removesuffix('-calibrated')
    if kind not in TRAINED:
        return restore_stage30(name, device)
    model = build_stabilized(kind, PROTOCOL).to(device)
    model.load_state_dict(torch.load(OUT / f'{name}.pt', map_location=device)['model'])
    return model.eval()


def selection():
    val = {}
    for kind in TRAINED:
        path = OUT / f'{kind}-fit.json'
        if not path.exists():
            continue
        hist = json.loads(path.read_text())
        val[kind + '-calibrated'] = min(hist, key=lambda r: r['criterion'])
    eligible = [f'{k}-calibrated' for k in ('fix_only', 'inertial_hop2', 'inertial_hop3') if f'{k}-calibrated' in val]
    selected = min(eligible, key=lambda name: val[name]['criterion'])
    choice = dict(selected=selected, heldout='stabilized_unseen-calibrated', validation=val)
    (OUT / 'selection.json').write_text(json.dumps(choice, indent=2) + '\n')
    return choice


@torch.no_grad()
def diagnostics(model, name, device):
    """Measure mechanism diagnostics: state norm, damping abduction, pusher RMSE, pinch/no-pinch, 5-block layers."""
    diag = {}
    # 1. Push-T damping probe on train + transfer slices (pinch vs no-pinch, wall AP, actor RMSE)
    d_pt_tr = load('pusht', 'train', device, labels=False)
    xyz = pool(d_pt_tr['x'][..., :3], d_pt_tr['role'][..., 0])
    shift = F.pad(xyz[..., :2], (0, 7))
    cd = dict(d_pt_tr, x=d_pt_tr['x'] - shift[:, None], fact=d_pt_tr['fact'] - shift[:, None, None], goal=d_pt_tr['goal'] - shift)
    facts_pt = model.facts(cd)
    X_m = facts_pt['m'][:, 0].cpu().numpy()
    X_u = facts_pt['u'][:, 0].cpu().numpy()
    X_mu = np.concatenate([X_m, X_u], axis=-1)
    damp_cls = d_pt_tr['seed'].cpu().numpy() % 3
    moved = ((d_pt_tr['fact'][:, -1, 0, :3] - d_pt_tr['fact'][:, 0, 0, :3]).norm(dim=-1) > 0.1).cpu().numpy()
    diag['pusht_damping_acc_all'] = float(cross_val_score(LogisticRegression(max_iter=500), X_mu, damp_cls, cv=5).mean())
    diag['pusht_damping_acc_moved'] = float(cross_val_score(LogisticRegression(max_iter=500), X_mu[moved], damp_cls[moved], cv=5).mean())
    diag['pusht_m_std'] = float(X_m.std(axis=0).mean())

    # Push-T transfer pinch/no-pinch, wall AP, pusher RMSE, state norm
    d_pt = load('pusht', PROTOCOL['test_split'], device, labels=True)
    o_pt = model(d_pt, native=True)
    i, j = paired_actions(d_pt['a'])
    y_diff = d_pt['pose'][:, i, -1, 0, :3] - d_pt['pose'][:, j, -1, 0, :3]
    p_diff = o_pt['pose'][:, i, -1, 0, :3] - o_pt['pose'][:, j, -1, 0, :3]
    sq = (p_diff - y_diff).square().sum(-1)
    push_c = d_pt['native_edge'][..., 0, 1].bool()
    wall_c = d_pt['native_edge'][..., 0, 2:6].any(-1)
    pinch_br = (push_c & wall_c).any(-1)
    pinch_pair = pinch_br[:, i] | pinch_br[:, j]
    diag['pusht_pinch_rmse'] = float(100.0 * sq[pinch_pair].mean().sqrt())
    diag['pusht_nopinch_rmse'] = float(100.0 * sq[~pinch_pair].mean().sqrt())
    act_err = (o_pt['native']['pose'][..., 1, :2] - d_pt['native_pose'][..., 1, :2]).norm(dim=-1)
    diag['pusht_actor_rmse_end'] = float(100.0 * act_err[..., -1].square().mean().sqrt())
    nat_y = d_pt['native_edge'].bool()
    nat_p = o_pt['native']['edge'].sigmoid()
    diag['pusht_pusher_ap'] = float(average_precision_score(nat_y[..., 0, 1].flatten().cpu().numpy(),
                                                            nat_p[..., 0, 1].flatten().cpu().numpy()))
    diag['pusht_wall_ap'] = float(average_precision_score(nat_y[..., 0, 2:6].flatten().cpu().numpy(),
                                                          nat_p[..., 0, 2:6].flatten().cpu().numpy()))

    # Push-T val by damping
    d_pv = load('pusht', 'val', device, labels=False)
    o_pv = model(d_pv)
    iv, jv = paired_actions(d_pv['a'])
    yv = d_pv['pose'][:, iv, -1, 0, :3] - d_pv['pose'][:, jv, -1, 0, :3]
    pv = o_pv['pose'][:, iv, -1, 0, :3] - o_pv['pose'][:, jv, -1, 0, :3]
    dv_arr = np.array([0.0, 0.08, 0.55])[d_pv['seed'].cpu().numpy() % 3]
    diag['pusht_val_by_damping'] = {}
    for dv in (0.0, 0.08, 0.55):
        sel = torch.as_tensor(dv_arr == dv, device=device)
        diag['pusht_val_by_damping'][str(dv)] = dict(
            rmse=float(100.0 * (pv[sel] - yv[sel]).square().sum(-1).mean().sqrt()),
            pred_mag=float(100.0 * pv[sel].norm(dim=-1).mean()),
            true_mag=float(100.0 * yv[sel].norm(dim=-1).mean()),
        )

    # CoPhy 5-block per-layer RMSE
    d_cp = load('cophy', PROTOCOL['test_split'], device, labels=False)
    o_cp = model(d_cp)
    ic, jc = paired_actions(d_cp['a'])
    e5 = (o_cp['pose'][:, ic, -1, :5, :3] - o_cp['pose'][:, jc, -1, :5, :3]) - (
        d_cp['pose'][:, ic, -1, :5, :3] - d_cp['pose'][:, jc, -1, :5, :3]
    )
    diag['cophy_5blk_by_layer'] = [float(e5[..., b, :].square().sum(-1).mean().sqrt()) for b in range(5)]

    # Measure terminal rollout state norm on Push-T & CoPhy
    diag['state_norm_end'] = {}
    for dom, d in [('cophy', d_cp), ('pusht', d_pt)]:
        xyz = pool(d['x'][..., :3], d['role'][..., 0])
        shift = F.pad(xyz[..., :2], (0, 7))
        cd = dict(d, x=d['x'] - shift[:, None], fact=d['fact'] - shift[:, None, None], goal=d['goal'] - shift)
        sub = subset(cd, torch.arange(min(64, len(cd['x'])), device=device))
        facts = model.facts(sub)
        if hasattr(model, 'bound_z'):
            from .model import safe_pose_input
            x_in = safe_pose_input(sub['x'], model.bound_z)
        else:
            x_in = sub['x']
        node = model.node(torch.cat([x_in, sub['fact'][:, -1] - sub['fact'][:, 0], sub['role'], facts['m'], facts['u']], -1))
        _, _, _, _, st = model.rollout(sub, node, facts, sub['a'], sub['reset'], None, 0.0)
        mv = sub['role'][:, None, :, 0].bool().expand(-1, st.size(1), -1)
        diag['state_norm_end'][dom] = float(st[:, :, -1][mv].norm(dim=-1).mean())
    return diag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()

    choice = selection()
    fresh = PROTOCOL['test_split']
    trained_avail = [f'{k}-calibrated' for k in TRAINED if (OUT / f'{k}-calibrated.pt').exists()]
    models = args.models or [*EARLIER, *trained_avail]

    def inputs(domain, split, device):
        return load(domain, split, device, labels=False)

    results = evaluate([*models, *REFERENCES], ('test', fresh), args.device, 'metrics.json',
                       load_data=inputs, restore_model=restore, output_dir=OUT)

    for domain in DOMAINS:
        data = load(domain, fresh, args.device)
        for name in models:
            model = restore(name, args.device)
            row = results[domain][fresh][name]
            if domain != 'cophy':
                row['edit'] = pushing_edits(model, data, domain)
            if name != 'shared-calibrated':
                row['native'] = native_metrics(model, data, domain)
            print('native/edit', domain, name, row.get('edit', {}).get('paired_edit_rmse'), flush=True)

    # Deep mechanism diagnostics
    diag_all = {}
    for name in models:
        if name == 'shared-calibrated':
            continue
        model = restore(name, args.device)
        diag_all[name] = diagnostics(model, name, args.device)
        print('diag', name, json.dumps({k: v for k, v in diag_all[name].items() if k != 'pusht_val_by_damping'}), flush=True)

    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    (OUT / 'diagnostics.json').write_text(json.dumps(diag_all, indent=2, allow_nan=False) + '\n')
    print('selected', choice['selected'], flush=True)
