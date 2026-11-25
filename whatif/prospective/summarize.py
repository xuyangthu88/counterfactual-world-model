import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .collect import OUT,DATA


def unify(prefix,name):
    files=[p for p in sorted(OUT.glob(prefix+'-*.csv')) if p.name.endswith('-steps.csv')==('steps' in name)]
    rows=[]
    for p in files or [OUT/name]:
        with p.open() as f:rows.extend(csv.DictReader(f))
    if files:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return rows


def main():
    unify('v','validation.csv');rows=unify('t','control.csv');unify('t','steps.csv')
    frozen=json.loads((OUT/'freeze.json').read_text());primary=frozen['model']
    raw=np.load(OUT/'paired-test.npz');rng=np.random.default_rng(11);result={}
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.4),layout='constrained')
    colors=['#788688','#666da1','#ad6d28','#8b9fbc','#116961']
    for t,unit in [('causalworld',65),('pusht',100)]:
        res={};scores={};pairs={};xy={}
        for name in sorted({r['policy'] for r in rows if r['task']==t}):
            group=sorted((r for r in rows if (r['task'],r['policy'])==(t,name)),key=lambda r:int(r['seed']))
            assert [int(r['seed']) for r in group]==list(range(351000,351160))
            a=np.array([[int(r['success']),float(r['coverage']),int(r['steps']),int(r['protocol'])] for r in group])
            scores[name]=a
            info=dict(n=len(a),success=int(a[:,0].sum()),coverage=a[:,1].mean(),steps=a[:,2].mean(),strata={})
            for g in np.unique(a[:,3]):
                z=a[a[:,3]==g]
                info['strata'][str(int(g))]=dict(n=len(z),success=int(z[:,0].sum()),coverage=z[:,1].mean(),steps=z[:,2].mean())
            res[name]=info
        def paired(v):
            ix=rng.integers(len(v),size=(6000,len(v)))
            return dict(mean=v.mean(0).tolist(),ci95=np.quantile(v[ix].mean(1),[.025,.975],axis=0).T.tolist())
        for n,a in scores.items():
            d=scores[primary][:,:3]-a[:,:3]
            pairs[n]=paired(d)
            if t=='pusht':pairs[n]['new_damping']=paired(d[a[:,3]==1])
        orig=raw[t+'_whatif_error']*unit
        for n in ('uniform','legacy','pets','cophy'):xy[n]=paired(orig-raw[t+'_'+n+'_error']*unit)
        result[t]=dict(control=res,paired_control=pairs,paired_position=xy)
        for ax,g,title in ([(axes[0],None,'CausalWorld · all layouts')] if t=='causalworld' else
                           [(axes[1],0,'PushT · native damping'),(axes[2],1,'PushT · new damping 0.25')]):
            names=['physics','cophy',frozen[t]['pets'],'uniform',primary]
            for j,n in enumerate(names):
                a=scores[n];v=a[:,0] if g is None else a[a[:,3]==g,0]
                ix=rng.integers(len(v),size=(6000,len(v)));lo,hi=100*np.quantile(v[ix].mean(1),[.025,.975])
                ax.bar(j,100*v.mean(),color=colors[j],width=.68);ax.vlines(j,lo,hi,lw=1.4,color='#2c4242')
                ax.annotate(f'{int(v.sum())}/{len(v)}',(j,hi),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xticks(range(5),['Physics','CoPhyNet','PETS','Uniform','WhatIf'],rotation=24,ha='right')
            ax.set_ylim(0,112);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
            ax.spines[['right','top']].set_visible(False)
    axes[0].set_ylabel('Native success (%)')
    fig.savefig('reports/stages/assets/09-future-control.png',dpi=170);plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest={}
    for path in sorted(DATA.glob('*.npz')):
        with np.load(path) as z:manifest[path.name]=dict(n=len(z['seed']),seed_min=int(z['seed'].min()),seed_max=int(z['seed'].max()),
                                              keys=sorted(z.files),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
