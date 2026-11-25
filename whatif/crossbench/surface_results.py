"""Summarize the matched precision-control comparison by layout and damping."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path('results/crossbench')


def main():
    with (OUT/'tracked-control.csv').open() as f:rows=list(csv.DictReader(f))
    rng=np.random.default_rng(211)
    all_seeds=sorted({int(r['scene']) for r in rows})
    ix=rng.integers(len(all_seeds),size=(4000,len(all_seeds)))
    ys={};result={}
    for policy in ('graph','pets','physics'):
        group=sorted((r for r in rows if r['policy']==policy),key=lambda r:int(r['scene']))
        assert [int(r['scene']) for r in group]==all_seeds
        y=np.array([[int(r['success']),float(r['coverage'])] for r in group])
        ys[policy]=y
        result[policy]={'n':len(y),'successes':int(y[:,0].sum()),'coverage':float(y[:,1].mean()),'strata':{}}
        for pr in ('0','1'):
            mask=np.array([r['protocol']==pr for r in group])
            result[policy]['strata'][pr]={'n':int(mask.sum()),'successes':int(y[mask,0].sum()),'coverage':float(y[mask,1].mean())}
    delta=ys['graph']-ys['pets']
    result['graph_minus_pets']={'difference':delta.mean(0).tolist(),'ci95':np.quantile(delta[ix].mean(1),[.025,.975],axis=0).T.tolist()}
    (OUT/'surface-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(8,3.1),layout='constrained')
    for j,(ax,label) in enumerate(zip(axes,['Native success (%)','Final goal coverage (%)'])):
        for n,(p,c) in enumerate(zip(('graph','pets','physics'),('#146463','#b57833','#657486'))):
            v=ys[p][:,j]*100;lo,hi=np.quantile(v[ix].mean(1),[.025,.975])
            ax.bar(n,v.mean(),color=c,width=.65)
            ax.vlines(n,lo,hi,color='#293841',lw=1.3)
            ax.annotate(f'{v.mean():.1f}',(n,hi),xytext=(0,3),textcoords='offset points',ha='center')
        ax.set_title(label);ax.set_xticks(range(3),['WhatIf','PETS (H=1)','Physics']);ax.set_ylim(0,105)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.savefig('reports/stages/assets/07-pusht-precision.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    main()
