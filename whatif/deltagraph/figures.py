"""Generate Stage 32 figures comparing baselines, Stage 30, Stage 31, and Stage 32 DeltaGraph variants."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from whatif.effects.collect import DOMAINS, ROOT
from whatif.shortstep.collect import PROTOCOL as SHORT
from .fit import OUT, PROTOCOL

NAMES = ('CoPhy (5-block)', 'CausalWorld (P2/P8)', 'Push-T (damp .25)')
UNITS = ('block lengths', 'mm', 'px')
ASSETS = ROOT / 'reports/stages/assets'


def plot_endpoints(metrics, choice):
    fresh = PROTOCOL['test_split']
    selected = choice['selected']
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 5.4), layout='constrained')
    for dom, title, unit, col in zip(DOMAINS, NAMES, UNITS, axes.T):
        panel = metrics[dom][fresh]
        ref = panel[selected]['comparison']['prediction_reference']
        keys = [ref, 'continued-calibrated', 'fix_only-calibrated', 'layout_hop2-calibrated',
                'deltagraph_hop2-calibrated', 'deltagraph_hop3-calibrated']
        keys = [k for k in keys if k in panel]
        lmap = {
            'cophynet': 'CoPhyNet', 'pets': 'PETS',
            'continued-calibrated': 'Stage 30',
            'fix_only-calibrated': 'Stage 31',
            'layout_hop2-calibrated': 'S32 Layout',
            'deltagraph_hop2-calibrated': 'S32 Delta-2',
            'deltagraph_hop3-calibrated': 'S32 Delta-3',
        }
        cmap = {
            'cophynet': '#9aa3ad', 'pets': '#9aa3ad',
            'continued-calibrated': '#c3cbd3',
            'fix_only-calibrated': '#87a8bc',
            'layout_hop2-calibrated': '#45849c',
            'deltagraph_hop2-calibrated': '#1c6172',
            'deltagraph_hop3-calibrated': '#0b3642',
        }
        for ax, key, caption in zip(col, ('delta_rmse', 'largest_effect_rmse'), ('All pairs', 'Top 5% tail')):
            vals = [panel[k][key] for k in keys]
            bars = ax.bar(np.arange(len(keys)), vals, width=0.68, color=[cmap[k] for k in keys])
            ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)
            ax.set_ylim(0, 1.24 * max(vals))
            ax.set_ylabel(f'{caption} RMSE ({unit})')
            ax.spines[['top', 'right']].set_visible(False)
            if key == 'delta_rmse':
                ax.set_title(title, fontweight='semibold')
                ax.set_xticks([])
                ax.axhline(0.7 * vals[0], ls='--', lw=1.0, color='#b03a3a', label='-30% target')
            else:
                ax.set_xticks(np.arange(len(keys)), [lmap[k] for k in keys], rotation=24, ha='right', fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8, loc='upper right')
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / '32-deltagraph-endpoints.png', dpi=240)
    plt.close(fig)


def plot_diagnostics(diag):
    keys = [k for k in ('continued-calibrated', 'fix_only-calibrated', 'layout_hop2-calibrated',
                        'deltagraph_hop2-calibrated', 'deltagraph_hop3-calibrated') if k in diag]
    lmap = {'continued-calibrated': 'S30', 'fix_only-calibrated': 'S31',
            'layout_hop2-calibrated': 'Layout', 'deltagraph_hop2-calibrated': 'Delta-2',
            'deltagraph_hop3-calibrated': 'Delta-3'}
    cmap = {'continued-calibrated': '#c3cbd3', 'fix_only-calibrated': '#87a8bc',
            'layout_hop2-calibrated': '#45849c', 'deltagraph_hop2-calibrated': '#1c6172',
            'deltagraph_hop3-calibrated': '#0b3642'}

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), layout='constrained')
    x = np.arange(len(keys))
    w = 0.36
    # Panel 1: Push-T Wall AP & Pusher AP
    b1 = axes[0].bar(x - w / 2, [diag[k]['pusht_wall_ap'] for k in keys], w, label='Wall AP', color='#c45b43')
    b2 = axes[0].bar(x + w / 2, [diag[k]['pusht_pusher_ap'] for k in keys], w, label='Pusher AP', color='#2a6f82')
    axes[0].bar_label(b1, fmt='%.2f', fontsize=6.5, padding=1)
    axes[0].bar_label(b2, fmt='%.2f', fontsize=6.5, padding=1)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('Push-T native contact AP', fontweight='semibold')
    axes[0].set_xticks(x, [lmap[k] for k in keys], fontsize=8)
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[0].spines[['top', 'right']].set_visible(False)

    # Panel 2: Push-T Pinch vs No-Pinch & Actor RMSE
    w3 = 0.26
    axes[1].bar(x - w3, [diag[k]['pusht_pinch_rmse'] for k in keys], w3, label='Pinch', color='#b84a39')
    axes[1].bar(x, [diag[k]['pusht_nopinch_rmse'] for k in keys], w3, label='No-pinch', color='#2f7f6f')
    axes[1].bar(x + w3, [diag[k]['pusht_actor_rmse_end'] for k in keys], w3, label='Pusher', color='#4e6fa8')
    axes[1].set_title('Push-T error breakdown', fontweight='semibold')
    axes[1].set_ylabel('RMSE (px)')
    axes[1].set_xticks(x, [lmap[k] for k in keys], fontsize=8)
    axes[1].legend(frameon=False, fontsize=7.5)
    axes[1].spines[['top', 'right']].set_visible(False)

    # Panel 3: CoPhy 5-block per-layer RMSE
    layers = np.arange(5)
    for k in keys:
        axes[2].plot(layers, diag[k]['cophy_5blk_by_layer'], marker='o', ms=4, color=cmap[k], label=lmap[k], lw=1.8)
    axes[2].set_title('CoPhy 5-block by height', fontweight='semibold')
    axes[2].set_xlabel('Block layer (0=base, 4=top)')
    axes[2].set_ylabel('Action-diff RMSE (blocks)')
    axes[2].set_xticks(layers)
    axes[2].legend(frameon=False, fontsize=7.5)
    axes[2].spines[['top', 'right']].set_visible(False)

    fig.savefig(ASSETS / '32-deltagraph-diagnostics.png', dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    metrics = json.loads((OUT / 'metrics.json').read_text())
    choice = json.loads((OUT / 'selection.json').read_text())
    diag = json.loads((OUT / 'diagnostics.json').read_text())
    plot_endpoints(metrics, choice)
    plot_diagnostics(diag)
