"""Plots for the Stage 27 report."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .collect import DOMAINS, OUT, ROOT
from .data import load, pair_ids, subset
from .evaluate import restore_world

ASSETS = ROOT/'reports/stages/assets'
TITLES = {'cophy':'CoPhy', 'causalworld':'CausalWorld', 'pusht':'Push-T'}
UNITS = {'cophy':'block edges', 'causalworld':'mm', 'pusht':'px'}
COLORS = ('#8894a5', '#c7ccd4', '#1e6680')
SHARED = 'shared-calibrated'


def plot_effects(data):
    fig, axes=plt.subplots(1,3,figsize=(10.2,3.6),layout='constrained')
    methods=('cophynet','pets',SHARED)
    labels=('CoPhyNet','PETS','Shared graph')
    for ax,domain in zip(axes,DOMAINS):
        for i,(method,label,color) in enumerate(zip(methods,labels,COLORS)):
            nums=[data[domain][split][method]['delta_rmse'] for split in ('test','transfer')]
            bars=ax.bar(np.arange(2)+(i-1)*.25,nums,.23,label=label,color=color)
            ax.bar_label(bars,fmt='%.2f' if domain=='cophy' else '%.1f',fontsize=8,padding=2)
        for i,split in enumerate(('test','transfer')):
            v=data[domain][split]
            goal=.7*min(v[m]['delta_rmse'] for m in methods[:2])
            ax.plot([i-.38,i+.38],[goal,goal],color='#ab413b',ls='--',lw=1)
        ax.set_xticks(range(2),['New layouts','Held-out setting'])
        ax.set_title(TITLES[domain],fontweight='semibold')
        ax.set_ylabel(f'Paired endpoint RMSE ({UNITS[domain]})')
        ax.set_ylim(0,ax.get_ylim()[1]*1.12)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(ASSETS/'27-action-effects.png',dpi=220)


@torch.no_grad()
def plot_excision():
    torch.set_num_threads(2)
    d=load('cophy','transfer')
    u,v=pair_ids(d['x'].size(1),'cpu')
    receiver=torch.arange(d['x'].size(1))
    nonlocal_receiver=(receiver!=u[:,None])&(receiver!=v[:,None])
    present=d['role'].sum(-1).bool()
    valid=(present[:,u]&present[:,v])[:,None,:,None]&present[:,None,None,:]&nonlocal_receiver
    amount=d['off'][:,:,:,-1].norm(dim=-1)
    mask=valid&(amount>.10)
    ids=mask.nonzero()
    order=torch.argsort(amount[mask])
    ix,action,pair,node=map(int,ids[order[len(ids)//2]])
    one=subset(d,torch.tensor([ix]),torch.tensor([action]))
    reference=one['off'][0,0,pair,:,node].numpy()
    fig,axes=plt.subplots(1,3,figsize=(9.8,2.9),layout='constrained',sharex=True,sharey=True)
    series={}
    for name in (SHARED,'no_edge_intervention-calibrated'):
        net=restore_world(name,'cpu')
        series[name]=net(one,torch.tensor([[[pair]]]))['edge_effect'][0,0,0,:,node].numpy()
    for i,ax in enumerate(axes):
        step=np.arange(1,6)*6
        ax.axhline(0,color='#d9dce0',lw=.8)
        ax.plot(step,reference[:,i],marker='o',color='#2c3140',label='Simulator pair')
        ax.plot(step,series[SHARED][:,i],marker='s',color=COLORS[-1],label='Trained on edge edits')
        ax.plot(step,series['no_edge_intervention-calibrated'][:,i],ls='--',color=COLORS[0],label='Without edit loss')
        ax.set_title(['X displacement','Y displacement','Z displacement'][i])
        ax.set_xticks(step)
        ax.set_xlabel('Bullet step')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Normal − disabled (block edges)')
    axes[2].legend(frameon=False,fontsize=8)
    fig.savefig(ASSETS/'27-edge-edit.png',dpi=220)
    record=dict(seed=int(one['seed'][0]),action=action,pair=[int(u[pair]),int(v[pair])],receiver=node,
                windows=[6,12,18,24,30], simulator=reference.tolist(),
                **{k:v.tolist() for k,v in series.items()})
    (OUT/'excision-example.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    plot_effects(json.loads((OUT/'metrics.json').read_text()))
    plot_excision()
