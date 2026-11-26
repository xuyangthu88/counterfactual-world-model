"""Read the new-layout results for the local-motion graph experiment."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .collect import DOMAINS, MOTION_PROTOCOL, OUT
from .figures import ASSETS, TITLES, UNITS


NAMES = ('shared-calibrated','motion_once-calibrated','motion_twice-calibrated')
LABELS = ('Stage 27','Predict motion, 1 update','Predict motion, 2 updates')
COLORS = ('#9ba9bc','#65a5b5','#1d5778')


def window_errors(records,split):
    fig,axes=plt.subplots(1,3,figsize=(10.4,3.15),layout='constrained')
    for ax,domain in zip(axes,DOMAINS):
        panel=records[domain][split]
        reference=panel[NAMES[-1]]['comparison']['prediction_reference']
        t=(6,12,18,24,30) if domain=='cophy' else (1,2,4,6,8)
        for method,label,color in zip(NAMES,LABELS,COLORS):
            ax.plot(t, panel[method]['delta_rmse_by_window'], marker='o',ms=3,color=color,label=label)
        label='CoPhyNet' if reference=='cophynet' else 'PETS'
        ax.plot(t,panel[reference]['delta_rmse_by_window'],ls='--',color='#5c524c',label=label)
        ax.set_title(TITLES[domain],fontweight='semibold')
        ax.set_xlabel('Simulator step')
        ax.set_ylabel(f'Action difference RMSE ({UNITS[domain]})')
        ax.set_xticks(t)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=7)
    fig.savefig(ASSETS/'28-prediction-over-time.png',dpi=220)


def edit_effect(records,split):
    panel=records['cophy'][split]
    fig,axes=plt.subplots(1,3,figsize=(9.6,3.2),layout='constrained')
    specs=[('remote_delta_rmse','Distant action effect','RMSE (block edges)',1.),
           ('downstream_trajectory_rmse','Downstream edge-removal effect','RMSE (block edges)',1.),
           ('downstream_effect_recall','Detected downstream effects','Recall (%)',100.)]
    for ax,(key,title,unit,scale) in zip(axes,specs):
        vals=[panel[m][key]*scale for m in NAMES]
        b=ax.bar(np.arange(len(NAMES)),vals,color=COLORS,width=.65)
        ax.bar_label(b,fmt='%.1f' if scale>1 else '%.2f',padding=3,fontsize=9)
        ax.set_ylim(0,max(vals)*1.20)
        ax.set_xticks(range(3),['Stage 27','1 update','2 updates'])
        ax.set_title(title,fontsize=10,fontweight='semibold')
        ax.set_ylabel(unit)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(ASSETS/'28-downstream-effects.png',dpi=220)


if __name__=='__main__':
    results=json.loads((OUT/'metrics-motion.json').read_text())
    split=MOTION_PROTOCOL['test_split']
    window_errors(results,split)
    edit_effect(results,split)
