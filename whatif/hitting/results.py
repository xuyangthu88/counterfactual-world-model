"""Paired native outcomes, fixed before comparison, and the figures for Stage 25."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .native import OUT,ROOT,DATA,DOMAINS,CONTROL

ASSETS=ROOT/'reports/stages/assets'
NAMES={'pusht':'Push-T','causalworld':'CausalWorld'}
def read(name):
    return json.loads((OUT/f'{name}.json').read_text())

def compare(x,y,indices):
    diff=x.astype(float)-y.astype(float)
    lo,hi=np.quantile(diff[indices].mean(1)*100,[.025,.975])
    return dict(rescued=int((diff>0).sum()),broken=int((diff<0).sum()),points=100*diff.mean(),interval=[lo,hi])

def summary():
    native=read('episodes')
    refs=read('references')
    choices=read('second')
    base,n=CONTROL
    indices=np.random.default_rng(271).integers(n,size=(5000,n))
    records={}
    for dom in DOMAINS:
        with np.load(DATA/f'{dom}-initial.npz') as f:
            early_alive=f['alive'].astype(bool)
        with np.load(DATA/f'{dom}-live.npz') as f:
            late_alive={(int(s),int(a)):bool(live) for s,a,live in zip(f['seed'],f['first'],f['mid_alive'])}
        vals={m:np.array(v) for m,v in native[dom].items()}
        for key in ('stage21','pets'):
            rows=sorted((r for r in refs if r['domain']==dom and r['policy']==key),key=lambda r:r['seed'])
            if [r['seed'] for r in rows]!=list(range(base,base+n)):
                raise ValueError('reference cohort mismatch')
            vals[key]=np.array([[r['success'],r['peak'],r['terminal']] for r in rows])
        arms={m:dict(n=n,success=int(q[:,0].sum()),peak=q[:,1].mean(),terminal=q[:,2].mean(),
                     vs_actor=compare(q[:,0],vals['actor'][:,0],indices)) for m,q in vals.items()}
        diffs={}
        def contrast(a,b):
            v=compare(vals[a][:,0],vals[b][:,0],indices)
            if a in choices[dom] and b in choices[dom]:
                left,right=[np.asarray(choices[dom][m]) for m in (a,b)]
                first=(left[:,0]!=right[:,0])&early_alive
                active=np.array([late_alive[base+i,int(a)] for i,a in enumerate(left[:,0])])
                second=(left[:,0]==right[:,0])&(left[:,1]!=right[:,1])&active
                v.update(first_changed=int(first.sum()),path_changed=int((first|second).sum()))
            return v
        for ref in ('actor','fixed','stage24','stage21','pets'):
            diffs['selected-'+ref]=contrast('selected',ref)
        for prefix in ('first_','late_','both_'):
            for a,b in [('direct','unpaired'),('means','direct'),('independent','means'),('shape','independent'),('shape','means'),('kl','shape')]:
                diffs[prefix+a+'-'+b]=contrast(prefix+a,prefix+b)
        paths={tuple([i,*v]) for seq in choices[dom].values() for i,v in enumerate(seq)}
        records[dom]=dict(arms=arms,contrasts=diffs,unique_native_paths=len(paths))
    (OUT/'summary.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    return records

def chart_style():
    local=Path.home()/'.local/share/fonts/NotoSansCJKsc-Regular.otf'
    if local.is_file():
        font_manager.fontManager.addfont(str(local))
    plt.rcParams.update({'font.family':['Noto Sans CJK SC','DejaVu Sans'],'font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})

def figures(records):
    ASSETS.mkdir(exist_ok=True,parents=True)
    chart_style()
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.1),layout='constrained')
    arms=['actor','fixed','stage21','stage24','both_unpaired','both_direct','both_means','both_independent','both_shape','both_kl','selected']
    names=['生成器','固定策略','Stage 21','Stage 24','无新增配对损失，读成功率','配对训练，读成功率','读均值','读独立边际形状','读联合形状','联合形状与 KL','验证集预选策略']
    for ax,dom in zip(axes,DOMAINS):
        a=records[dom]['arms']
        count=np.array([a[m]['success'] for m in arms])
        x=np.arange(len(names))
        ax.barh(x,100*count/CONTROL[1],color=['#9daab3']*4+['#41848a']*6+['#ba6b36'],height=.72)
        ax.set_yticks(x,names)
        ax.invert_yaxis()
        ax.set_title(NAMES[dom])
        ax.set_xlabel('闭环成功率（%）')
        ax.set_xlim(0,max(count)*100/CONTROL[1]*1.17)
        for i,v in enumerate(count):
            ax.text(v*100/CONTROL[1]+.4,i,str(v),va='center',fontsize=9)
    fig.savefig(ASSETS/'25-control.png',dpi=175)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9.4,5.9),layout='constrained')
    for row,dom in enumerate(DOMAINS):
        for col,phase in enumerate(('early','late')):
            ax=axes[row,col]
            v=read(phase+'-forecast')[dom]
            labels=[2,5,9,17,36] if phase=='early' else [10,13,17,25,36]
            for name,arr,color in [('无新增配对损失',v['marginal']['independent']['crps_xyz'],'#898b93'),
                      ('配对训练，独立读出',v['joint']['independent']['crps_xyz'],'#bb743a'),
                      ('配对训练，联合读出',v['joint']['paired']['crps_xyz'],'#25797c')]:
                ax.plot(labels,arr,marker='o',ms=4,label=name,color=color)
            ax.set_xticks(labels)
            ax.set_title(NAMES[dom]+(' · 第 1 步后预测' if col==0 else ' · 第 9 步后预测'))
            ax.set_ylabel('位移差 CRPS（'+('px' if dom=='pusht' else 'mm')+'）')
            ax.set_xlabel('预测对应的宏动作步')
            ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(ASSETS/'25-effects.png',dpi=175)
    plt.close(fig)

if __name__=='__main__':
    figures(summary())
