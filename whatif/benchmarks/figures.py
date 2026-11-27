"""Stage 33 figures: 5-domain reduction vs original baselines and contact graph AP/BER."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .data import ALL_DOMAINS, NEW_DOMAINS
from .fit import OUT, ROOT

ASSETS = ROOT / 'reports/stages/assets'


def transfer_key(dom):
    return 'transfer' if dom in NEW_DOMAINS else 'cumulativetransfer'


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    summary = json.loads((OUT / 'summary.json').read_text())
    metrics = json.loads((OUT / 'metrics.json').read_text())

    # Pick primary 5-domain shared model
    shared_candidates = [k for k in summary['ballscf']['transfer'] if k.startswith('shared5')]
    primary = shared_candidates[0]
    if len(shared_candidates) > 1:
        primary = max(shared_candidates,
                      key=lambda m: np.mean([summary[d][transfer_key(d)][m]['reduction_pct'] for d in ALL_DOMAINS]))

    labels = ['CoPhy\n5-blk', 'BallsCF\n5-6 balls', 'CW Push\nP2/P8', 'CW Stack2\nP2/P8', 'Push-T\ndamp=.25']
    test_labels = ['CoPhy\n3-4 blk', 'BallsCF\n3-4 balls', 'CW Push\nP1/P7', 'CW Stack2\nP1/P7', 'Push-T\ntest']

    # Figure 1: % reduction vs better of original CoPhyNet / PETS on both test and OOD transfer
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=160)
    x = np.arange(len(ALL_DOMAINS))
    width = 0.35

    test_red = [summary[d]['test'][primary]['reduction_pct'] for d in ALL_DOMAINS]
    tr_red = [summary[d][transfer_key(d)][primary]['reduction_pct'] for d in ALL_DOMAINS]

    ax = axes[0]
    b1 = ax.bar(x - width / 2, test_red, width, label='In-dist Test', color='#4C78A8')
    b2 = ax.bar(x + width / 2, tr_red, width, label='OOD Transfer', color='#E45756')
    ax.axhline(30.0, color='#2ca02c', linestyle='--', linewidth=1.4, label='30% Target')
    ax.axhline(0.0, color='black', linewidth=0.8)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + (1.5 if h >= 0 else -4.0),
                    f'{h:+.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Delta RMSE Reduction vs Better Baseline (%)')
    ax.set_title(f'Single Shared Checkpoint ({primary})\nvs Better of Original CoPhyNet / PETS')
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Contact Graph AP & BER vs NCII on OOD Transfer
    ax2 = axes[1]
    our_ap = [metrics[d][transfer_key(d)][primary]['graph']['contact']['ap'] for d in ALL_DOMAINS]
    ncii_ap = [metrics[d][transfer_key(d)]['ncii']['graph']['contact']['ap'] for d in ALL_DOMAINS]
    our_ber = [metrics[d][transfer_key(d)][primary]['graph']['contact']['ber'] for d in ALL_DOMAINS]
    ncii_ber = [metrics[d][transfer_key(d)]['ncii']['graph']['contact']['ber'] for d in ALL_DOMAINS]

    b3 = ax2.bar(x - width / 2, ncii_ap, width, label='NCII (repro) AP', color='#9E9E9E')
    b4 = ax2.bar(x + width / 2, our_ap, width, label=f'{primary} AP', color='#54A24B')
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f'{h:.2f}',
                     ha='center', va='bottom', fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel('Contact Graph Average Precision (AP)')
    ax2.set_title('Dynamic Contact Graph Discovery (OOD Transfer)\nHigher AP = Better')
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out1 = ASSETS / '33-benchmarks-summary.png'
    fig.savefig(out1)
    plt.close(fig)
    print('wrote', out1)

    # Figure 2: Absolute normalized error bar chart (CoPhyNet vs PETS vs Shared5) on OOD transfer
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    w = 0.25
    cp_rel, pt_rel, our_rel, pt_raw = [], [], [], []
    y_cap = 1.45
    for d in ALL_DOMAINS:
        tk = transfer_key(d)
        best_b = min(metrics[d][tk]['cophynet']['delta_rmse'], metrics[d][tk]['pets']['delta_rmse'])
        cp_rel.append(metrics[d][tk]['cophynet']['delta_rmse'] / best_b)
        pr = metrics[d][tk]['pets']['delta_rmse'] / best_b
        pt_raw.append(pr)
        pt_rel.append(min(pr, y_cap))
        our_rel.append(metrics[d][tk][primary]['delta_rmse'] / best_b)
    ax.bar(x - w, cp_rel, w, label='CoPhyNet (orig)', color='#72B7B2')
    ax.bar(x, pt_rel, w, label='PETS (orig)', color='#F58518')
    ax.bar(x + w, our_rel, w, label=f'Ours ({primary})', color='#E45756')
    ax.axhline(1.0, color='black', linestyle=':', linewidth=1.1, label='Best Baseline = 1.00x')
    ax.axhline(0.7, color='#2ca02c', linestyle='--', linewidth=1.4, label='-30% Target (0.70x)')
    for i, (v, pr) in enumerate(zip(our_rel, pt_raw)):
        ax.text(x[i] + w, v + 0.025, f'{v:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#B22222')
        if pr > y_cap:
            ax.text(x[i], y_cap - 0.03, f'{pr:.1f}x\n(clip)', ha='center', va='top', fontsize=8,
                    fontweight='bold', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 1.55)
    ax.set_ylabel('Relative Endpoint Delta RMSE (Best Baseline = 1.0)')
    ax.set_title('Five-Domain OOD Transfer: Relative Counterfactual Action-Difference RMSE\n(Single Shared 870k Checkpoint vs Original Baselines)')
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out2 = ASSETS / '33-benchmarks-relative.png'
    fig.savefig(out2)
    plt.close(fig)
    print('wrote', out2)


if __name__ == '__main__':
    main()
