"""Generate Stage 31 figures comparing baselines, Stage 30, fix_only, and inertial_multihop variants."""
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


def plot_main_comparison(metrics, choice):
    fresh = PROTOCOL['test_split']
    selected = choice['selected']
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.4), layout='constrained')
    for dom, title, unit, col in zip(DOMAINS, NAMES, UNITS, axes.T):
        panel = metrics[dom][fresh]
        ref = panel[selected]['comparison']['prediction_reference']
        keys = [ref, 'continued-calibrated', 'fix_only-calibrated', 'inertial_hop2-calibrated', 'inertial_hop3-calibrated']
        keys = [k for k in keys if k in panel]
        label_map = {
            'cophynet': 'CoPhyNet',
            'pets': 'PETS',
            'continued-calibrated': 'Stage 30',
            'fix_only-calibrated': 'S31 Fix-only',
            'inertial_hop2-calibrated': 'S31 Hop-2',
            'inertial_hop3-calibrated': 'S31 Hop-3',
        }
        color_map = {
            'cophynet': '#9aa3ad',
            'pets': '#9aa3ad',
            'continued-calibrated': '#b7c2cc',
            'fix_only-calibrated': '#6290a8',
            'inertial_hop2-calibrated': '#1c6172',
            'inertial_hop3-calibrated': '#0e3c4a',
        }
        labels = [label_map[k] for k in keys]
        colors = [color_map[k] for k in keys]
        for ax, key, caption in zip(col, ('delta_rmse', 'largest_effect_rmse'), ('All pairs', 'Top 5% tail')):
            vals = [panel[k][key] for k in keys]
            bars = ax.bar(np.arange(len(keys)), vals, width=0.68, color=colors)
            ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)
            ax.set_ylim(0, 1.24 * max(vals))
            ax.set_ylabel(f'{caption} RMSE ({unit})')
            ax.spines[['top', 'right']].set_visible(False)
            if key == 'delta_rmse':
                ax.set_title(title, fontweight='semibold')
                ax.set_xticks([])
                ax.axhline(0.7 * vals[0], ls='--', lw=1.0, color='#b03a3a', label='-30% target')
            else:
                ax.set_xticks(np.arange(len(keys)), labels, rotation=24, ha='right', fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8, loc='upper right')
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / '31-stabilized-endpoints.png', dpi=240)
    plt.close(fig)


def plot_diagnostics(diag, choice):
    keys = [k for k in ('continued-calibrated', 'fix_only-calibrated', 'inertial_hop2-calibrated', 'inertial_hop3-calibrated')
            if k in diag]
    labels = {'continued-calibrated': 'Stage 30', 'fix_only-calibrated': 'Fix-only',
              'inertial_hop2-calibrated': 'Hop-2', 'inertial_hop3-calibrated': 'Hop-3'}
    colors = {'continued-calibrated': '#b7c2cc', 'fix_only-calibrated': '#6290a8',
              'inertial_hop2-calibrated': '#1c6172', 'inertial_hop3-calibrated': '#0e3c4a'}

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.3), layout='constrained')

    # Panel 1: Terminal state norm (Push-T & CoPhy)
    x = np.arange(len(keys))
    w = 0.36
    cp_s = [diag[k]['state_norm_end']['cophy'] for k in keys]
    pt_s = [diag[k]['state_norm_end']['pusht'] for k in keys]
    b1 = axes[0].bar(x - w / 2, cp_s, w, label='CoPhy', color='#3e7b8f')
    b2 = axes[0].bar(x + w / 2, pt_s, w, label='Push-T', color='#d27d46')
    axes[0].bar_label(b1, fmt='%.1f', fontsize=6.5, padding=1)
    axes[0].bar_label(b2, fmt='%.1f', fontsize=6.5, padding=1)
    axes[0].set_title('Terminal state norm', fontweight='semibold')
    axes[0].set_xticks(x, [labels[k] for k in keys], rotation=20, ha='right', fontsize=8)
    axes[0].set_ylabel('Mean L2 norm')
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[0].spines[['top', 'right']].set_visible(False)

    # Panel 2: Push-T damping probe accuracy (m+u)
    accs = [100 * diag[k]['pusht_damping_acc_moved'] for k in keys]
    bars = axes[1].bar(x, accs, width=0.62, color=[colors[k] for k in keys])
    axes[1].bar_label(bars, fmt='%.1f%%', fontsize=7, padding=2)
    axes[1].axhline(79.2, ls='--', lw=0.9, color='#2c7a4b', label='Raw-fact RF (79.2%)')
    axes[1].axhline(33.3, ls=':', lw=0.8, color='#888888', label='Chance (33.3%)')
    axes[1].set_ylim(0, 100)
    axes[1].set_title('Push-T damping abduction', fontweight='semibold')
    axes[1].set_xticks(x, [labels[k] for k in keys], rotation=20, ha='right', fontsize=8)
    axes[1].set_ylabel('3-way linear probe (%)')
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].spines[['top', 'right']].set_visible(False)

    # Panel 3: Push-T pinch vs no-pinch & actor RMSE
    pinch = [diag[k]['pusht_pinch_rmse'] for k in keys]
    nopinch = [diag[k]['pusht_nopinch_rmse'] for k in keys]
    actor = [diag[k]['pusht_actor_rmse_end'] for k in keys]
    w3 = 0.26
    axes[2].bar(x - w3, pinch, w3, label='Pinch pairs', color='#b84a39')
    axes[2].bar(x, nopinch, w3, label='No-pinch pairs', color='#2f7f6f')
    axes[2].bar(x + w3, actor, w3, label='Pusher RMSE', color='#4e6fa8')
    axes[2].set_title('Push-T error decomposition', fontweight='semibold')
    axes[2].set_xticks(x, [labels[k] for k in keys], rotation=20, ha='right', fontsize=8)
    axes[2].set_ylabel('RMSE (px)')
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].spines[['top', 'right']].set_visible(False)

    # Panel 4: CoPhy 5-block RMSE by layer (0..4)
    layers = np.arange(5)
    for k in keys:
        axes[3].plot(layers, diag[k]['cophy_5blk_by_layer'], marker='o', ms=4,
                     color=colors[k], label=labels[k], lw=1.8)
    axes[3].set_title('CoPhy 5-block by height', fontweight='semibold')
    axes[3].set_xlabel('Block layer (0=base, 4=top)')
    axes[3].set_ylabel('Action-diff RMSE (blocks)')
    axes[3].set_xticks(layers)
    axes[3].legend(frameon=False, fontsize=7.5)
    axes[3].spines[['top', 'right']].set_visible(False)

    fig.savefig(ASSETS / '31-stabilized-diagnostics.png', dpi=240)
    plt.close(fig)


def plot_time_curves(metrics):
    fresh = PROTOCOL['test_split']
    keys = [k for k in ('continued-calibrated', 'fix_only-calibrated', 'inertial_hop2-calibrated', 'inertial_hop3-calibrated')
            if k in metrics['cophy'][fresh]]
    labels = {'continued-calibrated': 'Stage 30', 'fix_only-calibrated': 'S31 Fix-only',
              'inertial_hop2-calibrated': 'S31 Hop-2', 'inertial_hop3-calibrated': 'S31 Hop-3'}
    colors = {'continued-calibrated': '#a9b4bf', 'fix_only-calibrated': '#6290a8',
              'inertial_hop2-calibrated': '#1c6172', 'inertial_hop3-calibrated': '#0e3c4a'}
    styles = {'continued-calibrated': '--', 'fix_only-calibrated': '-.',
              'inertial_hop2-calibrated': '-', 'inertial_hop3-calibrated': '-'}

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), layout='constrained')
    for dom, title, unit, ax in zip(DOMAINS, NAMES, UNITS, axes):
        panel = metrics[dom][fresh]
        t = SHORT['cophy_clock' if dom == 'cophy' else 'push_clock']
        for k in keys:
            vals = panel[k]['native']['native_delta_rmse']
            ax.plot(t, vals, marker='o', ms=2.5, color=colors[k], ls=styles[k], label=labels[k], lw=1.6)
        ax.set_title(title, fontweight='semibold')
        ax.set_xlabel('Simulator step')
        ax.set_ylabel(f'Action diff. RMSE ({unit})')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.savefig(ASSETS / '31-stabilized-time.png', dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    metrics = json.loads((OUT / 'metrics.json').read_text())
    choice = json.loads((OUT / 'selection.json').read_text())
    diag = json.loads((OUT / 'diagnostics.json').read_text())
    plot_main_comparison(metrics, choice)
    plot_diagnostics(diag, choice)
    plot_time_curves(metrics)
