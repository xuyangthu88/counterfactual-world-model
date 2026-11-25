"""Aggregate at the episode level; plot the three paper questions in one figure."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('results/crossbench')
ASSETS = Path('reports/stages/assets')
TASKS = {'causalworld': ('CausalWorld',65.,'mm'), 'pusht': ('PushT',100.,'px')}


def bootstrap(values, index):
    return np.quantile(values[index].mean(1),[.025,.975]).tolist()


def graph_boot(y, pred, threshold, index):
    hit=pred>=threshold
    pos=y.sum((1,2)); neg=(~y).sum((1,2))
    fn=(~hit & y).sum((1,2)); fp=(hit & ~y).sum((1,2))
    return .5*(fn[index].sum(1)/pos[index].sum(1)+fp[index].sum(1)/neg[index].sum(1))


def episodes(files):
    traces={}
    for name in files:
        with (OUT/name).open() as f:
            for r in csv.DictReader(f):
                key=r['task'],r['policy'],int(r['scene'])
                traces.setdefault(key,[]).append(r)
    return traces


def finish():
    metric=json.loads((OUT/'metrics.json').read_text())
    config=json.loads((OUT/'calibration.json').read_text())
    select=json.loads((OUT/'value_selection.json').read_text())['policy']
    test=episodes(['rollouts.csv','value-test.csv'])
    val=episodes(['value-validation.csv'])
    raw=np.load(OUT/'test_arrays.npz')
    rng=np.random.default_rng(211)
    report={}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    traj,curve=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    colors=['#146463','#b57833']
    for row,(task,(label,scale,unit)) in enumerate(TASKS.items()):
        y=raw[task+'_labels']
        ix=rng.integers(len(y),size=(4000,len(y)))
        graph={}
        for p in ('ours','null_only','ncii'):
            graph[p]=graph_boot(y,raw[f'{task}_{p}_edges'],config[p]['threshold'],ix)
        gp=(graph['ours']-graph['ncii'])*100.
        err={p:raw[f'{task}_{p}_error']*scale for p in ('ours','cophy')}
        by_protocol={}
        strata=raw[task+'_stratum']
        for g in np.unique(strata):
            by_protocol[int(g)]={p:float(v[strata==g].mean()) for p,v in err.items()}
        seeds=sorted(i for t,p,i in test if t==task and p==select)
        pix=rng.integers(len(seeds),size=(4000,len(seeds)))
        control={}; validation={}; scores={}
        for data,dest in [(test,control),(val,validation)]:
            for policy in sorted({p for t,p,i in data if t==task}):
                group=[data[k] for k in sorted(data) if k[:2]==(task,policy)]
                last=[e[-1] for e in group]
                success=np.array([int(r['success']) for r in last])
                cover=np.array([float(r['coverage']) for r in last])
                length=np.array([sum(int(r['steps']) for r in seq) for seq in group])
                dest[policy]={'n':len(group),'successes':int(success.sum()),'coverage':float(cover.mean()),
                              'native_steps':float(length.mean()),'by_protocol':{}}
                for g in sorted({r['protocol'] for r in last}):
                    sub=np.array([r['protocol']==g for r in last])
                    dest[policy]['by_protocol'][g]={'n':int(sub.sum()),'successes':int(success[sub].sum()),'coverage':float(cover[sub].mean())}
                if data is test:
                    scores[policy]=(success,cover)
        def control_delta(policy):
            return {k:{'delta':float((a-b).mean()),'ci95':bootstrap(a-b,pix)}
                    for k,a,b in zip(['success','coverage'],scores[select],scores[policy])}
        report[task]={'graph_delta_pp':float((metric[task]['ours']['graph']['balanced_error']-metric[task]['ncii']['graph']['balanced_error'])*100),
                      'graph_delta_ci':np.quantile(gp,[.025,.975]).tolist(),
                      'xy_delta':float((err['ours']-err['cophy']).mean()),
                      'xy_delta_ci':bootstrap(err['ours']-err['cophy'],ix),
                      'xy_by_protocol':by_protocol,'control':control,'validation':validation,
                      'control_delta':{p:control_delta(p) for p in scores}}
        h = 10 if task=='causalworld' else 20
        for p,name,color in [(select,'WhatIf',colors[0]),('pets','PETS',colors[1])]:
            group=[test[task,p,i] for i in seeds]
            series=np.array([[float(e[min(t,len(e)-1)]['coverage']) for t in range(h)] for e in group])
            lo,hi=np.quantile(series[pix].mean(1),[.025,.975],axis=0)
            axis=curve[row]
            axis.plot(np.arange(1,h+1),series.mean(0),label=name,color=color,lw=2)
            axis.fill_between(np.arange(1,h+1),lo,hi,color=color,alpha=.12)
        curve[row].set(title=label,xlabel='Executed pushes',ylabel='Mean goal coverage',ylim=(0,1.04))
        curve[row].legend(frameon=False)
        fields=[(['WhatIf','NCII'],[100*metric[task][p]['graph']['balanced_error'] for p in ('ours','ncii')],
                 [np.quantile(100*graph[p],[.025,.975]) for p in ('ours','ncii')], 'Edge balanced error (%)'),
                (['WhatIf','CoPhyNet'],[v.mean() for v in err.values()],[bootstrap(v,ix) for v in err.values()],
                 f'Action-effect XY error ({unit})'),
                (['WhatIf','PETS'],[100*scores[p][0].mean() for p in (select,'pets')],
                 [bootstrap(scores[p][0]*100,pix) for p in (select,'pets')],'Closed-loop success (%)')]
        for col,(names,means,interval,title) in enumerate(fields):
            ax=axes[row,col]
            ax.bar([0,1],means,color=colors,width=.58)
            ax.vlines([0,1],[v[0] for v in interval],[v[1] for v in interval],color='#24343b',lw=1.4)
            for j,m in enumerate(means):
                ax.annotate(f'{m:.1f}',(j,interval[j][1]),xytext=(0,3),textcoords='offset points',ha='center')
            ax.set_xticks([0,1],names)
            ax.set_title(title,fontsize=10.5)
            ax.set_ylim(0,max(5,max(v[1] for v in interval)*1.25))
            ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
            if col==0:ax.set_ylabel(label,weight='bold')
    ASSETS.mkdir(parents=True,exist_ok=True)
    fig.savefig(ASSETS/'07-three-questions.png',dpi=170)
    traj.savefig(ASSETS/'07-control.png',dpi=170)
    plt.close('all')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    finish()
