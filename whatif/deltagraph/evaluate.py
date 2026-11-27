"""Evaluate Stage 32 DeltaGraph variants vs Stage 31, Stage 30, and frozen baselines."""
import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from torch.nn import functional as F

from whatif.effects.collect import DOMAINS
from whatif.effects.data import subset
from whatif.effects.evaluate import evaluate
from whatif.effects.fit import paired_actions
from whatif.mechanism.model import pool
from whatif.shortstep.data import load
from whatif.shortstep.evaluate import native_metrics, pushing_edits
from whatif.stabilized.evaluate import restore as restore_s31
from whatif.stabilized.model import safe_pose_input
from .fit import OUT, PROTOCOL
from .model import build_deltagraph

EARLIER = ('continued-calibrated', 'fix_only-calibrated', 'inertial_hop2-calibrated', 'inertial_hop3-calibrated')
TRAINED = tuple(PROTOCOL['variants'])
REFERENCES = ('cophynet', 'pets', 'ncii')


def restore(name, device):
    kind = name.removesuffix('-calibrated')
    if kind not in TRAINED:
        return restore_s31(name, device)
    model = build_deltagraph(kind, PROTOCOL).to(device)
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
    eligible = [f'{k}-calibrated' for k in ('layout_hop2', 'deltagraph_hop2', 'deltagraph_hop3') if f'{k}-calibrated' in val]
    selected = min(eligible, key=lambda name: val[name]['criterion'])
    choice = dict(selected=selected, heldout='deltagraph_unseen-calibrated', validation=val)
    (OUT / 'selection.json').write_text(json.dumps(choice, indent=2) + '\n')
    return choice


@torch.no_grad()
def diagnostics(model, name, device):
    diag = {}
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

    d_cp = load('cophy', PROTOCOL['test_split'], device, labels=False)
    o_cp = model(d_cp)
    ic, jc = paired_actions(d_cp['a'])
    e5 = (o_cp['pose'][:, ic, -1, :5, :3] - o_cp['pose'][:, jc, -1, :5, :3]) - (
        d_cp['pose'][:, ic, -1, :5, :3] - d_cp['pose'][:, jc, -1, :5, :3]
    )
    diag['cophy_5blk_by_layer'] = [float(e5[..., b, :].square().sum(-1).mean().sqrt()) for b in range(5)]
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
            row['native'] = native_metrics(model, data, domain)

    diag_all = {}
    for name in models:
        model = restore(name, args.device)
        diag_all[name] = diagnostics(model, name, args.device)
        print('diag', name, json.dumps(diag_all[name]), flush=True)

    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    (OUT / 'diagnostics.json').write_text(json.dumps(diag_all, indent=2, allow_nan=False) + '\n')
    print('selected', choice['selected'], flush=True)
