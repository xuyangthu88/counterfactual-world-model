"""Plot endpoint effects and measured changes after cutting the same interaction."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from whatif.effects.collect import DOMAINS, ROOT
from .collect import OUT, PROTOCOL

ASSETS=ROOT/'reports/stages/assets'
NAMES=('CoPhy','CausalWorld','Push-T')
UNITS=('block lengths','mm','px')
COLORS=('#9aa6b4','#738441','#c4cad1','#226679')
PARENT='motion_once-calibrated'


def terminal(panel, selected):
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.1),layout='constrained')
    for domain,name,unit,ax in zip(DOMAINS,NAMES,UNITS,axes):
        stat=panel[domain]
        ref=stat[selected]['comparison']['prediction_reference']
        methods=(ref,'shared-calibrated',PARENT,selected)
        values=[stat[m]['delta_rmse'] for m in methods]
        bars=ax.bar(range(4),values,color=COLORS,width=.62)
        ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
        ax.hlines(.7*values[0],-.38,3.38,ls='--',lw=1,color='#a83b3b')
        ax.set_xticks(range(4),['CoPhyNet' if ref=='cophynet' else 'PETS','Stage 27','Stage 28','Stage 29'],fontsize=8)
        ax.set_title(name,fontweight='semibold')
        ax.set_ylabel(f'Endpoint RMSE ({unit})')
        ax.set_ylim(0,max(values)*1.18)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(ASSETS/'29-endpoint-effects.png',dpi=240)


def edit_effect(panel,selected):
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.1),layout='constrained')
    for domain,name,unit,ax in zip(DOMAINS,NAMES,UNITS,axes):
        stat=panel[domain]
        keys=(PARENT,'short_coarse-calibrated','shortstep-calibrated','short_rotated-calibrated')
        def measure(key):
            row=stat[key] if domain=='cophy' else stat[key]['edit']
            return row['paired_edit_rmse']
        values=[measure(key) for key in keys]
        bars=ax.bar(range(4),values,color=['#226679' if k==selected else '#a6b1bd' for k in keys],width=.64)
        ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=2)
        ax.set_xticks(range(4),['Stage 28','Coarse','Short steps','+ Rotation'],fontsize=8)
        ax.set_title(name,fontweight='semibold')
        ax.set_ylabel(f'Paired edit RMSE ({unit})')
        ax.set_ylim(0,max(values)*1.18)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(ASSETS/'29-paired-edits.png',dpi=240)


def native_time(panel):
    fig,axes=plt.subplots(1,3,figsize=(10.2,3),layout='constrained')
    stages=('short_coarse-calibrated','shortstep-calibrated','short_rotated-calibrated')
    for domain,name,unit,ax in zip(DOMAINS,NAMES,UNITS,axes):
        t=PROTOCOL['cophy_clock' if domain=='cophy' else 'push_clock']
        stat=panel[domain]
        for key,label,color in zip(stages,('Coarse labels','Short labels','Short + rotation'),
                                   ('#8f9db2','#256984','#738441')):
            ax.plot(t,stat[key]['native']['native_delta_rmse'],marker='o',ms=3,color=color,label=label)
        ax.set_xticks(t if len(t)<=8 else [1,3,6,12,18,24,30])
        ax.set_title(name,fontweight='semibold')
        ax.set_xlabel('Simulator step')
        ax.set_ylabel(f'Action difference RMSE ({unit})')
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(ASSETS/'29-short-time-effects.png',dpi=240)


if __name__=='__main__':
    scores=json.loads((OUT/'metrics.json').read_text())
    choice=json.loads((OUT/'selection.json').read_text())['selected']+'-calibrated'
    panel={d:scores[d][PROTOCOL['test_split']] for d in DOMAINS}
    terminal(panel,choice)
    edit_effect(panel,choice)
    native_time(panel)
