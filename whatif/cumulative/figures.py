"""Plot the endpoint learning comparison, including the largest effects."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from whatif.effects.collect import DOMAINS, ROOT
from whatif.shortstep.collect import PROTOCOL as SHORT
from .fit import OUT, PARENT, PROTOCOL

NAMES=('CoPhy','CausalWorld','Push-T')
UNITS=('block lengths','mm','px')
MODELS=(PARENT,'continued-calibrated','cumulative-calibrated')
ASSETS=ROOT/'reports/stages/assets'
LABELS=('Stage 29','More training','Cumulative')
PALETTE=('#c3cbd3','#879eaa','#1c6172')


def endpoints(stats,selected):
    fig,axes=plt.subplots(2,3,figsize=(10.2,5),layout='constrained')
    for dom,title,unit,col in zip(DOMAINS,NAMES,UNITS,axes.T):
        row=stats[dom]
        ref=row[selected]['comparison']['prediction_reference']
        keys=(ref,'shared-calibrated',*MODELS)
        labels=('CoPhyNet' if ref=='cophynet' else 'PETS','Stage 27',*LABELS)
        colors=('#a0a8b2','#788c55',*PALETTE)
        for ax,key,caption in zip(col,('delta_rmse','largest_effect_rmse'),('All pairs','Top 5%')):
            values=[row[name][key] for name in keys]
            bars=ax.bar(np.arange(len(keys)),values,width=.72,color=colors)
            ax.bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
            ax.set_ylim(0,1.22*max(values))
            ax.set_ylabel(f'{caption} RMSE ({unit})')
            ax.spines[['top','right']].set_visible(False)
            if key=='delta_rmse':
                ax.set_title(title,fontweight='semibold')
                ax.set_xticks([])
                ax.axhline(.7*values[0],ls='--',lw=.85,color='#a33d3d')
            else:
                ax.set_xticks(np.arange(len(keys)),labels,rotation=24,ha='right',fontsize=8)
    fig.savefig(ASSETS/'30-cumulative-effects.png',dpi=240)


def time_curves(stats):
    fig,axes=plt.subplots(1,3,figsize=(10.3,2.8),layout='constrained')
    for dom,title,unit,ax in zip(DOMAINS,NAMES,UNITS,axes):
        row=stats[dom]
        for key,color,label,style in zip(MODELS,PALETTE,LABELS,('-','--','-')):
            values=row[key]['native']['native_delta_rmse']
            t=SHORT['cophy_clock' if dom=='cophy' else 'push_clock']
            ax.plot(t,values,marker='o',ms=2.5,color=color,label=label,ls=style,
                    zorder=3 if style=='--' else 2)
        ax.set_title(title,fontweight='semibold')
        ax.set_xlabel('Simulator step')
        ax.set_ylabel(f'Action diff. RMSE ({unit})')
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(ASSETS/'30-cumulative-time.png',dpi=240)


if __name__=='__main__':
    metrics=json.loads((OUT/'metrics.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    panel={d:metrics[d][PROTOCOL['test_split']] for d in DOMAINS}
    endpoints(panel,selection['selected'])
    time_curves(panel)
