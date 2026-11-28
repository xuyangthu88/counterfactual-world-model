"""Stage 35: Publication-quality figures comparing Stage 34 single-OBB vs Stage 35 Multi-Convex SAT + Torque + Pearl Exogenous Abduction."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'reports/stages/assets'


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    s35_sum = json.loads((ROOT / 'results/multiconvex/summary.json').read_text())
    s35_met = json.loads((ROOT / 'results/multiconvex/metrics.json').read_text())
    s35_diag = json.loads((ROOT / 'results/multiconvex/diagnostics.json').read_text())
    s34_met = json.loads((ROOT / 'results/geometry/metrics.json').read_text())
    s34_diag = json.loads((ROOT / 'results/geometry/diagnostics.json').read_text())
    s33_met = json.loads((ROOT / 'results/benchmarks/metrics.json').read_text())
    s32_met = json.loads((ROOT / 'results/deltagraph/metrics.json').read_text())

    # Helper to get best original 9D baseline per domain/split
    def orig9d_base(dom, sp):
        if dom in ('ballscf', 'stacking2'):
            tbl = s33_met[dom][sp]
        else:
            tbl = s32_met[dom][sp]
        cp = tbl['cophynet']['delta_rmse']
        pt = tbl['pets']['delta_rmse']
        return min(cp, pt)

    domains = ['cophy', 'ballscf', 'causalworld', 'stacking2', 'pusht']
    labels = ['CoPhy\n5-blk', 'BallsCF\n5–6 balls', 'CW Push\nP2/P8', 'CW Stack2\nP2/P8', 'Push-T\ndamp=.25']
    tr_sp = {'cophy': 'cumulativetransfer', 'ballscf': 'transfer', 'causalworld': 'cumulativetransfer',
             'stacking2': 'transfer', 'pusht': 'cumulativetransfer'}

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=180)

    # Panel A: 5-Domain Single Shared Checkpoint Reduction (%) — Stage 34 vs Stage 35 (Test & OOD)
    ax = axes[0]
    x = np.arange(len(domains))
    w = 0.20
    s34_ood = [100.0 * (s35_sum[d][tr_sp[d]]['best_mc_baseline_delta_rmse'] - s34_met[d][tr_sp[d]]['geo_shared5_hop3-calibrated']['delta_rmse']) / s35_sum[d][tr_sp[d]]['best_mc_baseline_delta_rmse'] for d in domains]
    s35_test = [s35_sum[d]['test']['mc_shared5_hop3-calibrated']['reduction_pct'] for d in domains]
    s35_ood = [s35_sum[d][tr_sp[d]]['mc_shared5_hop3-calibrated']['reduction_pct'] for d in domains]
    s35_ood_9d = [100.0 * (orig9d_base(d, tr_sp[d]) - s35_met[d][tr_sp[d]]['mc_shared5_hop3-calibrated']['delta_rmse']) / orig9d_base(d, tr_sp[d]) for d in domains]

    b0 = ax.bar(x - 1.5 * w, s34_ood, w, label='S34 1-OBB OOD (vs fair)', color='#94a3b8')
    b1 = ax.bar(x - 0.5 * w, s35_test, w, label='S35 Multi-Convex Test (vs fair)', color='#2563eb')
    b2 = ax.bar(x + 0.5 * w, s35_ood, w, label='S35 Multi-Convex OOD (vs fair)', color='#dc2626')
    b3 = ax.bar(x + 1.5 * w, s35_ood_9d, w, label='S35 Multi-Convex OOD (vs orig 9D)', color='#f59e0b')
    ax.axhline(30.0, color='#16a34a', ls='--', lw=1.6, label='30% Target')
    for b_idx, bars in enumerate((b1, b2)):
        for rect in bars:
            h = rect.get_height()
            dy = 1.0 if b_idx == 0 else (2.4 if h < 25 else 1.0)
            ax.text(rect.get_x() + rect.get_width() / 2, h + dy, f'{h:+.1f}%',
                    ha='center', va='bottom', fontsize=6.9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Delta RMSE Reduction vs Better Baseline (%)', fontsize=10)
    ax.set_title('Stage 35 Single Shared Checkpoint (mc_shared5_hop3)\n5-Domain Performance vs Stage 34 & Fair/9D Baselines', fontsize=10.5)
    ax.set_ylim(0, max(max(s35_test), max(s35_ood)) + 20)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=7.6, loc='upper left', ncol=2, frameon=True)

    # Panel B: Push-T Physical Mechanism Breakthrough (1-OBB -> 2-Part Multi-Convex SAT + Torque)
    ax2 = axes[1]
    stages = ['S30\n(9D)', 'S32\nDelta (9D)', 'S34\n1-OBB (5d)', 'S35\nMC (5d)', 'S35\nMC (3d)']
    wall_ap = [0.363, 0.454, s34_diag['geo_shared5_hop3-calibrated']['pusht_wall_ap'],
               s35_diag['mc_shared5_hop3-calibrated']['pusht_wall_ap'],
               s35_diag['mc_shared3_hop3-calibrated']['pusht_wall_ap']]
    push_ap = [0.562, 0.645, s34_diag['geo_shared5_hop3-calibrated']['pusht_pusher_ap'],
               s35_diag['mc_shared5_hop3-calibrated']['pusht_pusher_ap'],
               s35_diag['mc_shared3_hop3-calibrated']['pusht_pusher_ap']]
    pinch_rmse = [147.88, 147.39, s34_diag['geo_shared5_hop3-calibrated']['pusht_pinch_rmse'],
                  s35_diag['mc_shared5_hop3-calibrated']['pusht_pinch_rmse'],
                  s35_diag['mc_shared3_hop3-calibrated']['pusht_pinch_rmse']]
    xs = np.arange(len(stages))
    ax2.bar(xs - 0.18, wall_ap, 0.34, label='Wall Contact AP (left ↑)', color='#16a34a', alpha=0.85)
    ax2.bar(xs + 0.18, push_ap, 0.34, label='Pusher Contact AP (left ↑)', color='#2563eb', alpha=0.85)
    for xi, (wa, pa) in enumerate(zip(wall_ap, push_ap)):
        ax2.text(xi - 0.18, wa + 0.015, f'{wa:.2f}', ha='center', va='bottom', fontsize=8)
        ax2.text(xi + 0.18, pa + 0.015, f'{pa:.2f}', ha='center', va='bottom', fontsize=8)
    ax2.set_ylim(0.0, 1.15)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(stages, fontsize=9)
    ax2.set_ylabel('Push-T OOD (.25) Native Contact AP (↑)', fontsize=10)
    ax2_r = ax2.twinx()
    ax2_r.plot(xs, pinch_rmse, '-o', color='#dc2626', lw=2.3, ms=6.5, label='Pinch-Pair RMSE px (right ↓)')
    for xi, pr in enumerate(pinch_rmse):
        ax2_r.text(xi, pr + 4.0, f'{pr:.0f}px', ha='center', va='bottom', color='#b91c1c', fontsize=8.2, fontweight='bold')
    ax2_r.set_ylabel('Push-T OOD (.25) Wall-Pinch RMSE (px, ↓)', fontsize=10, color='#b91c1c')
    ax2_r.set_ylim(42, 166)
    ax2.set_title('Push-T OOD (.25): 2-Part Multi-Convex SAT + Contact Torque', fontsize=10.5)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='lower left', fontsize=8, frameon=True)
    ax2.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(ASSETS / '35-multiconvex-summary.png')
    plt.close(fig)

    # Figure 2: Ablation (No-Exo vs Full Exo on CoPhy 5-blk & BallsCF) + Double Zero-Shot Transfer
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), dpi=180)
    ax = axes[0]
    abl_models = [
        ('S34 1-OBB\n(geo_shared5_hop3)', s34_met['cophy']['cumulativetransfer']['geo_shared5_hop3-calibrated']['delta_rmse'],
         s34_met['cophy']['test']['geo_shared5_hop3-calibrated']['delta_rmse'],
         s34_met['pusht']['cumulativetransfer']['geo_shared5_hop3-calibrated']['delta_rmse'] / 100.0),
        ('S35 MC+Torque\nNo-Exo (ablation)', s35_met['cophy']['cumulativetransfer']['mc_shared5_noexo-calibrated']['delta_rmse'],
         s35_met['cophy']['test']['mc_shared5_noexo-calibrated']['delta_rmse'],
         s35_met['pusht']['cumulativetransfer']['mc_shared5_noexo-calibrated']['delta_rmse'] / 100.0),
        ('S35 MC+Torque\n+Exo (5-dom full)', s35_met['cophy']['cumulativetransfer']['mc_shared5_hop3-calibrated']['delta_rmse'],
         s35_met['cophy']['test']['mc_shared5_hop3-calibrated']['delta_rmse'],
         s35_met['pusht']['cumulativetransfer']['mc_shared5_hop3-calibrated']['delta_rmse'] / 100.0),
        ('S35 MC+Torque\n+Exo (3-dom)', s35_met['cophy']['cumulativetransfer']['mc_shared3_hop3-calibrated']['delta_rmse'],
         s35_met['cophy']['test']['mc_shared3_hop3-calibrated']['delta_rmse'],
         s35_met['pusht']['cumulativetransfer']['mc_shared3_hop3-calibrated']['delta_rmse'] / 100.0),
    ]
    xa = np.arange(len(abl_models))
    wa = 0.25
    v_c5 = [r[1] for r in abl_models]
    v_c3 = [r[2] for r in abl_models]
    v_pt = [r[3] for r in abl_models]
    for off, vals, lbl, col in [(-wa, v_c3, 'CoPhy 3–4 blk Test', '#2563eb'),
                                (0.0, v_c5, 'CoPhy 5-blk OOD', '#dc2626'),
                                (wa, v_pt, 'Push-T OOD (×100px)', '#16a34a')]:
        bars = ax.bar(xa + off, vals, wa, label=lbl, color=col, alpha=0.88)
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xticks(xa)
    ax.set_xticklabels([r[0] for r in abl_models], fontsize=8.3)
    ax.set_ylabel('Endpoint Delta RMSE (↓ lower is better)', fontsize=9.5)
    ax.set_title('Stage 35 Component Ablation (CoPhy & Push-T)', fontsize=10.5)
    ax.set_ylim(0, max(v_c5) + 0.35)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')

    # Panel B: Zero-Shot Unseen Push-T Contact AP & RMSE Progression
    ax = axes[1]
    zs_names = ['NCII-MC\n(Push-T trained)', 'S32 9D\n(0 Push-T)', 'S34 1-OBB\n(0 Push-T)', 'S35 2-Part MC\n(0 Push-T)']
    ap_t = [s35_met['pusht']['test']['ncii_mc']['graph']['contact']['ap'], 0.230,
            s34_met['pusht']['test']['geo_unseen_pusht4-calibrated']['graph']['contact']['ap'],
            s35_met['pusht']['test']['mc_unseen_pusht4-calibrated']['graph']['contact']['ap']]
    ap_o = [s35_met['pusht']['cumulativetransfer']['ncii_mc']['graph']['contact']['ap'], 0.176,
            s34_met['pusht']['cumulativetransfer']['geo_unseen_pusht4-calibrated']['graph']['contact']['ap'],
            s35_met['pusht']['cumulativetransfer']['mc_unseen_pusht4-calibrated']['graph']['contact']['ap']]
    xz = np.arange(len(zs_names))
    b_t = ax.bar(xz - 0.18, ap_t, 0.34, label='Push-T Test Contact AP (↑)', color='#2563eb')
    b_o = ax.bar(xz + 0.18, ap_o, 0.34, label='Push-T OOD .25 Contact AP (↑)', color='#dc2626')
    for bars in (b_t, b_o):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.015, f'{h:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(xz)
    ax.set_xticklabels(zs_names, fontsize=8.8)
    ax.set_ylabel('Push-T Contact Graph AP (↑)', fontsize=9.5)
    ax.set_ylim(0, max(ap_t) + 0.15)
    ax.set_title('Zero-Shot Cross-Simulator Push-T Contact Discovery\n(0 Push-T Training Frames vs Push-T-Trained NCII-MC)', fontsize=10.2)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=8.2, loc='upper left')

    plt.tight_layout()
    plt.savefig(ASSETS / '35-multiconvex-ablation-zeroshot.png')
    plt.close(fig)
    print('Saved Stage 35 figures to', ASSETS)


if __name__ == '__main__':
    main()
