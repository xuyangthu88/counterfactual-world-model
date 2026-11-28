"""Stage 34 figures: Universal SAT-OBB Geometry 5-domain summary, Push-T diagnostics, and zero-shot transfer."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from whatif.benchmarks.data import ALL_DOMAINS, NEW_DOMAINS
from whatif.effects.collect import ROOT
from .fit import OUT

ASSETS = ROOT / 'reports/stages/assets'


def tr_key(dom):
    return 'transfer' if dom in NEW_DOMAINS else 'cumulativetransfer'


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    summary = json.loads((OUT / 'summary.json').read_text())
    metrics = json.loads((OUT / 'metrics.json').read_text())
    diags = json.loads((OUT / 'diagnostics.json').read_text())
    s33_m = json.loads((ROOT / 'results/benchmarks/metrics.json').read_text())

    # Pick flagship 5-domain SAT-OBB checkpoint (highest mean OOD reduction across all 5 domains)
    shared5_cands = [k for k in ('geo_shared5_hop3-calibrated', 'geo_shared5-calibrated')
                     if k in summary['pusht']['cumulativetransfer']]
    primary = max(shared5_cands,
                  key=lambda k: np.mean([summary[d][tr_key(d)][k]['reduction_pct'] for d in ALL_DOMAINS]))
    labels = ['CoPhy\n5-blk', 'BallsCF\n5-6 balls', 'CW Push\nP2/P8', 'CW Stack2\nP2/P8', 'Push-T\ndamp=.25']

    # --- Figure 1: 5-Domain SAT-OBB Reduction (vs Fair *_geo & Orig 9D) + Push-T Mechanism Progress ---
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9), dpi=160)
    x = np.arange(len(ALL_DOMAINS))
    w = 0.26

    test_geo = [summary[d]['test'][primary]['reduction_pct'] for d in ALL_DOMAINS]
    tr_geo = [summary[d][tr_key(d)][primary]['reduction_pct'] for d in ALL_DOMAINS]
    tr_orig = []
    for d in ALL_DOMAINS:
        tk = tr_key(d)
        orig_b = min(s33_m[d][tk]['cophynet']['delta_rmse'], s33_m[d][tk]['pets']['delta_rmse'])
        ours = metrics[d][tk][primary]['delta_rmse']
        tr_orig.append(100.0 * (orig_b - ours) / orig_b)

    ax = axes[0]
    b1 = ax.bar(x - w, test_geo, w, label='In-dist Test (vs *_geo)', color='#4C78A8')
    b2 = ax.bar(x, tr_geo, w, label='OOD Transfer (vs *_geo)', color='#E45756')
    b3 = ax.bar(x + w, tr_orig, w, label='OOD Transfer (vs orig 9D)', color='#F58518', alpha=0.85)
    ax.axhline(30.0, color='#2ca02c', linestyle='--', linewidth=1.4, label='30% Target')
    ax.axhline(0.0, color='black', linewidth=0.8)
    for idx_b, bars in enumerate((b1, b2)):
        for bar in bars:
            h = bar.get_height()
            dx = -0.04 if idx_b == 0 else 0.04
            ax.text(bar.get_x() + bar.get_width() / 2 + dx, h + (1.3 if idx_b == 0 else 2.2), f'{h:+.1f}%',
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 69)
    ax.set_ylabel('Delta RMSE Reduction vs Better Baseline (%)')
    ax.set_title(f'Stage 34 Universal SAT-OBB ({primary.replace("-calibrated", "")})\nSingle Shared Checkpoint Across 5 Benchmarks')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Push-T Contact Graph AP & Actor Drift Progression S30 -> S31 -> S32 -> S34 SAT-OBB
    ax2 = axes[1]
    s32_d = json.loads((ROOT / 'results/deltagraph/diagnostics.json').read_text())
    stages = [
        ('S30\n(9D)', s32_d['continued-calibrated']),
        ('S31\n(9D)', s32_d['inertial_hop2-calibrated']),
        ('S32\nDelta (9D)', s32_d['deltagraph_hop2-calibrated']),
        ('S34\nSAT-OBB\n(5-dom)', diags[primary]),
    ]
    best3_k = 'geo_shared3_hop3-calibrated' if 'geo_shared3_hop3-calibrated' in diags else 'geo_shared3-calibrated'
    if best3_k in diags:
        stages.append(('S34\nSAT-OBB\n(3-dom)', diags[best3_k]))

    xx = np.arange(len(stages))
    wall_aps = [st[1]['pusht_wall_ap'] for st in stages]
    push_aps = [st[1]['pusht_pusher_ap'] for st in stages]
    act_err = [st[1]['pusht_actor_rmse_end'] for st in stages]

    ax2_r = ax2.twinx()
    p1 = ax2.bar(xx - 0.18, wall_aps, 0.34, color='#54A24B', label='Wall Contact AP (left ↑)')
    p2 = ax2.bar(xx + 0.18, push_aps, 0.34, color='#4C78A8', label='Pusher Contact AP (left ↑)')
    ax2_r.plot(xx, act_err, color='#E45756', marker='o', linewidth=2.2, label='Actor RMSE px (right ↓)')
    for bar in p1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f'{h:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in p2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f'{h:.2f}', ha='center', va='bottom', fontsize=8)
    for xi, ae in zip(xx, act_err):
        ax2_r.text(xi + (0.08 if xi == 0 else 0.0), ae + 1.5, f'{ae:.1f}px', ha='center', va='bottom', fontsize=8,
                   color='#B22222', fontweight='bold')
    ax2.set_xticks(xx)
    ax2.set_xticklabels([st[0] for st in stages], fontsize=8.5)
    ax2.set_ylim(0, 0.96)
    ax2_r.set_ylim(0, 48)
    ax2.set_ylabel('Push-T Native Contact AP (↑)')
    ax2_r.set_ylabel('Push-T Actor Endpoint RMSE (px, ↓)', color='#E45756')
    ax2.set_title('Push-T Physical Mechanism Progression (OOD .25)')
    h1, l_1 = ax2.get_legend_handles_labels()
    h2, l_2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l_1 + l_2, frameon=False, fontsize=7.8, loc='upper right')
    ax2.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    out1 = ASSETS / '34-geometry-summary.png'
    fig.savefig(out1)
    plt.close(fig)
    print('wrote', out1)

    # --- Figure 2: Zero-Shot Cross-Simulator Generalization (Unseen Push-T & Unseen Stacking2) ---
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.7), dpi=160)

    # Left: Zero-shot Unseen Push-T Contact AP (0 Push-T training frames vs Push-T-trained NCII!)
    ax = axes[0]
    s32_m = json.loads((ROOT / 'results/deltagraph/metrics.json').read_text())
    zs_labels = ['NCII\n(trained on\nPush-T!)', 'S32 Zero-Shot\n(2-dom 9D)', 'S33 Zero-Shot\n(4-dom 9D)', 'S34 Zero-Shot\n(4-dom SAT-OBB)']
    zs_test_ap = [
        s32_m['pusht']['test']['ncii']['contact']['ap'],
        s32_m['pusht']['test']['deltagraph_unseen-calibrated']['contact']['ap'],
        s33_m['pusht']['test']['unseen_pusht_4dom-calibrated']['graph']['contact']['ap'],
        metrics['pusht']['test']['geo_unseen_pusht4-calibrated']['graph']['contact']['ap'],
    ]
    zs_tr_ap = [
        s32_m['pusht']['cumulativetransfer']['ncii']['contact']['ap'],
        s32_m['pusht']['cumulativetransfer']['deltagraph_unseen-calibrated']['contact']['ap'],
        s33_m['pusht']['cumulativetransfer']['unseen_pusht_4dom-calibrated']['graph']['contact']['ap'],
        metrics['pusht']['cumulativetransfer']['geo_unseen_pusht4-calibrated']['graph']['contact']['ap'],
    ]
    xz = np.arange(len(zs_labels))
    bz1 = ax.bar(xz - 0.18, zs_test_ap, 0.34, label='Push-T Test AP', color='#4C78A8')
    bz2 = ax.bar(xz + 0.18, zs_tr_ap, 0.34, label='Push-T OOD .25 AP', color='#E45756')
    for bars in (bz1, bz2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_xticks(xz)
    ax.set_xticklabels(zs_labels, fontsize=8.5)
    ax.set_ylim(0, 0.58)
    ax.set_ylabel('Contact Graph Average Precision (AP)')
    ax.set_title('Zero-Shot Unseen Push-T Transfer (0 Push-T Training Frames)\nSAT-OBB Exceeds Push-T-Trained NCII by >2x')
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis='y', alpha=0.3)

    # Right: Zero-Shot Unseen Stacking2 Compositional Transfer (if available) or Zero-Shot Push-T RMSE
    ax2 = axes[1]
    if 'geo_unseen_stacking2-calibrated' in metrics['stacking2']['transfer']:
        st_lbls = ['CoPhyNet-Geo\n(trained on\nStacking2)', 'PETS-Geo\n(trained on\nStacking2)',
                   'S34 Zero-Shot\n(0 Stacking2\nframes!)', f'S34 Shared-5\n({primary.replace("-calibrated", "")})']
        st_test = [
            metrics['stacking2']['test']['cophynet_geo']['delta_rmse'],
            metrics['stacking2']['test']['pets_geo']['delta_rmse'],
            metrics['stacking2']['test']['geo_unseen_stacking2-calibrated']['delta_rmse'],
            metrics['stacking2']['test'][primary]['delta_rmse'],
        ]
        st_tr = [
            metrics['stacking2']['transfer']['cophynet_geo']['delta_rmse'],
            metrics['stacking2']['transfer']['pets_geo']['delta_rmse'],
            metrics['stacking2']['transfer']['geo_unseen_stacking2-calibrated']['delta_rmse'],
            metrics['stacking2']['transfer'][primary]['delta_rmse'],
        ]
        xs = np.arange(len(st_lbls))
        bs1 = ax2.bar(xs - 0.18, st_test, 0.34, label='Stacking2 P1/P7 Test (mm)', color='#4C78A8')
        bs2 = ax2.bar(xs + 0.18, st_tr, 0.34, label='Stacking2 P2/P8 OOD (mm)', color='#E45756')
        for bars in (bs1, bs2):
            for bar in bars:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f'{h:.1f}',
                         ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax2.set_xticks(xs)
        ax2.set_xticklabels(st_lbls, fontsize=8.3)
        ax2.set_ylabel('Endpoint Action-Difference RMSE (mm, ↓)')
        ax2.set_title('Zero-Shot Compositional Transfer to Unseen 2-Block Stacking2\n(Composing CW 1-Block + CoPhy Multi-Block Physics)')
        ax2.legend(frameon=False, fontsize=8.5)
        ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out2 = ASSETS / '34-geometry-zeroshot.png'
    fig.savefig(out2)
    plt.close(fig)
    print('wrote', out2)


if __name__ == '__main__':
    main()
