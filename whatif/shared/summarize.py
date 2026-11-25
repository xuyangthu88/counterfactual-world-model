"""Paired layout statistics, data provenance, and one stage's two figures."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .actions import OUT,DATA

ASSETS=Path('reports/stages/assets')
NAMES={'graph':'WhatIf','pets':'PETS','cophy':'CoPhyNet','physics':'Physics','previous':'Stage 07','prior':'Uniform','poseonly':'Pose only'}
COLORS={'graph':'#116860','pets':'#bc742c','cophy':'#5d69a1','physics':'#7b858b','previous':'#a4aaba','prior':'#799c99','poseonly':'#a5bdb5'}


def interval(x,ix):
    return np.quantile(np.asarray(x)[ix].mean(1),[.025,.975],axis=0).tolist()


def load_rows(files):
    rows=[]
    for f in files:
        with f.open() as handle:rows.extend(csv.DictReader(handle))
    return rows


def collect():
    files=[p for p in sorted(OUT.glob('test-*.csv')) if not p.name.endswith('-steps.csv')]
    rows=load_rows(files or [OUT/'control.csv'])
    for r in load_rows([OUT/'previous.csv']):
        rows.append(dict(task='pusht',policy='previous',seed=r['scene'],success=r['success'],coverage=r['coverage'],
                         steps=r['steps'],protocol=r['protocol'],initial_success='0'))
    metrics=json.loads((OUT/'metrics.json').read_text())
    raw=np.load(OUT/'paired.npz')
    report={}
    np_rng=np.random.default_rng(11)
    pct=['graph','pets','cophy','physics','previous','prior','poseonly']
    full={}
    for task,unit in [('causalworld',65.),('pusht',100.)]:
        pools={};control={};diff={}
        for name in pct:
            grp=sorted([r for r in rows if r['task']==task and r['policy']==name],key=lambda r:int(r['seed']))
            seeds=[int(r['seed']) for r in grp]
            assert seeds==list(range(261000,261128)),(name,task,len(grp))
            a=np.array([[int(r['success']),float(r['coverage']),int(r['steps']),int(r['protocol'])] for r in grp])
            pools[name]=a
            sub={}
            for key in sorted(set(a[:,3])):
                v=a[a[:,3]==key]
                sub[str(int(key))]=dict(n=len(v),success=int(v[:,0].sum()),coverage=v[:,1].mean(),steps=v[:,2].mean())
            control[name]=dict(n=len(a),success=int(a[:,0].sum()),coverage=a[:,1].mean(),steps=a[:,2].mean(),by_protocol=sub)
        full[task]=pools
        ix=np_rng.integers(0,128,(6000,128))
        for n in pct:
            delta=pools['graph'][:,:3]-pools[n][:,:3]
            diff[n]=dict(mean=delta.mean(0).tolist(),ci95=interval(delta,ix))
            if task=='pusht':
                for g in [0,1]:
                    x=delta[pools['graph'][:,3]==g]
                    j=np_rng.integers(len(x),size=(6000,len(x)))
                    diff[n]['damping_'+str(g)]=dict(mean=x.mean(0).tolist(),ci95=interval(x,j))
        xy={}
        a=raw[task+'_graph_error']*unit
        ix=np_rng.integers(len(a),size=(6000,len(a)))
        for n in ['cophy','pets','uniform','pre_probe']:
            dx=a-raw[task+'_'+n+'_error']*unit
            xy[n]={'mean':float(dx.mean()),'ci95':interval(dx,ix)}
        report[task]=dict(control=control,paired_control=diff,paired_xy=xy)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.7),layout='constrained')
    for ax,(task,protocol,title) in zip(axes,[('causalworld',None,'CausalWorld · all layouts'),('pusht',0,'PushT · native damping'),('pusht',1,'PushT · new damping 0.25')]):
        names=['previous','physics','cophy','pets','graph']
        for j,n in enumerate(names):
            z=full[task][n]
            a=z[:,0] if protocol is None else z[z[:,3]==protocol,0]
            ix=np_rng.integers(len(a),size=(6000,len(a)));lo,hi=interval(a*100,ix)
            ax.bar(j,100*a.mean(),color=COLORS[n],width=.68);ax.vlines(j,lo,hi,lw=1.2,color='#273037')
            ax.annotate(f'{int(a.sum())}/{len(a)}',(j,hi),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(len(names)),[NAMES[n] for n in names],rotation=25,ha='right')
        ax.set_ylim(0,112);ax.set_title(title);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    axes[0].set_ylabel('Native success (%)')
    fig.savefig(ASSETS/'08-control.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10.5,5.5),layout='constrained')
    for row,(task,unit,label) in enumerate([('causalworld',65,'mm'),('pusht',100,'px')]):
        v=metrics[task]
        fields=[(['graph','ncii','source_only'],[100*v[k]['edges']['balanced_error'] for k in ['graph','ncii','source_only']],'Balanced edge error (%)'),
                (['graph','pets','cophy'],[unit*v[k]['prediction']['xy'] for k in ['graph','pets','cophy']],f'Action-effect position error ({label})'),
                (['graph','pets','cophy'],[unit*v[k]['prediction']['energy'] for k in ['graph','pets','cophy']],f'Position energy score ({label})')]
        for col,(names,values,title) in enumerate(fields):
            ax=axes[row,col]
            ax.bar(range(len(names)),values,color=[COLORS.get(k,'#83929a') for k in names],width=.65)
            for j,val in enumerate(values):ax.annotate(f'{val:.1f}',(j,val),xytext=(0,2),textcoords='offset points',ha='center',fontsize=9)
            ax.set_xticks(range(len(names)),[NAMES.get(k,{'ncii':'NCII','source_only':'Source ID'}.get(k,k)) for k in names])
            ax.set_ylim(0,max(values)*1.25+.05);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
            if not col:ax.set_ylabel('CausalWorld' if row==0 else 'PushT')
    fig.savefig(ASSETS/'08-prediction.png',dpi=170);plt.close(fig)
    manifest={}
    for path in sorted(DATA.glob('*.npz')):
        with np.load(path) as z:
            manifest[path.name]=dict(keys=sorted(z.files),n=len(z['seed']),seed_min=int(z['seed'].min()),seed_max=int(z['seed'].max()),
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest['inherited_cw_training']={'path':'data/crossbench/causalworld-train.npz','snapshots':1200,'sha256':hashlib.sha256(Path('data/crossbench/causalworld-train.npz').read_bytes()).hexdigest()}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':collect()
