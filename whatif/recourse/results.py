"""Reported success counts and paired differences use the underlying independent reset layout."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .native import OUT,ROOT,DOMAINS,START
from .anchor_native import BASE,COUNT

def compare(a,b):
    diff=a[:,0]-b[:,0]
    rng=np.random.default_rng(271)
    idx=rng.integers(0,len(diff),(5000,len(diff)))
    return dict(rescued=int((diff>0).sum()),broken=int((diff<0).sum()),gain=float(diff.mean()),
                interval=np.quantile(diff[idx].mean(1),[.025,.975]).tolist())

def summarize(anchor):
    tag='anchor-' if anchor else ''
    refs=json.loads((OUT/(tag+'references.json')).read_text())
    plan=json.loads((OUT/(tag+'second.json')).read_text())
    first=json.loads((OUT/(tag+'first.json')).read_text())
    result={}
    base,n=(BASE,COUNT) if anchor else START['control']
    for dom in DOMAINS:
        raw=(json.loads((OUT/f'anchor-{dom}-episodes.json').read_text()) if anchor else
             json.loads((OUT/'episodes.json').read_text())[dom])
        runs={k:np.array(v) for k,v in raw.items()}
        for name in ('stage21','pets'):
            rows=sorted([r for r in refs if r['domain']==dom and r['policy']==name],key=lambda r:r['seed'])
            if not np.array_equal([r['seed'] for r in rows],np.arange(base,base+n)):
                raise ValueError('frozen control seeds differ')
            runs[name]=np.array([[r['success'],r['peak'],r['terminal']] for r in rows])
        records={}
        for name,values in runs.items():
            records[name]=dict(n=len(values),success=int(values[:,0].sum()),terminal=float(values[:,2].mean()),
                               peak=float(values[:,1].mean()),G=float((values@[1,.5,.25]).mean()),
                               versus={ref:compare(values,v) for ref,v in runs.items()})
            if name in first[dom]:
                if anchor:
                    pp=np.asarray(first[dom][name])
                    records[name]['committed']=int((pp[:,1]==0).sum())
                path=np.array(plan[dom][name])
                records[name]['different_second_policy']=int((path[:,0]!=path[:,1]).sum())
                records[name]['original_path']=int((path==0).all(1).sum())
        menu={k:np.asarray(v) for k,v in plan[dom].items()}
        distinct=sum(len({tuple(seq[i]) for seq in menu.values()}) for i in range(n))
        count={}
        comparisons=([('anchor_kl','anchor_law'),('anchor_law','anchor_mean'),('anchor_mean','anchor_marginal'),('anchor_both_marginal','anchor_marginal'),('kl','law')] if anchor else
                     [('kl','law'),('law','mean'),('pooled','mean'),('backup_kl','backup_law'),
                      ('backup_independent','backup_law'),('backup_mean','backup_marginal')])
        for a,b in comparisons:
            aa=np.asarray(first[dom][a])
            bb=np.asarray(first[dom][b])
            firstdiff=(aa!=bb).any(1) if anchor else aa!=bb
            count[a+' vs '+b]=dict(first=int(firstdiff.sum()),paths=int((menu[a]!=menu[b]).any(1).sum()))
        result[dom]=dict(arms=records,choice_changes=count,executions=distinct)
    return result

def figures(results):
    assets=ROOT/'reports/stages/assets'
    fig,ax=plt.subplots(2,2,figsize=(10,7))
    subsets=[('first',['actor','fixed','stage23','marginal','pooled','mean','law','kl','backup_mean','backup_law']),
             ('anchored',['actor','fixed','stage23','pooled','backup_mean','anchor_marginal','anchor_mean','anchor_law','anchor_kl'])]
    labels=dict(actor='Generator',fixed='Fixed policy',stage23='Stage 23',marginal='Marginal',pooled='Pooled mean',
                mean='Conditional mean',law='Conditional CE',kl='CE + information KL',backup_mean='Fitted policy mean',
                backup_law='Fitted policy CE',anchor_marginal='Anchored marginal',anchor_mean='Anchored pair mean',
                anchor_law='Anchored pair CE',anchor_kl='Anchored CE + KL')
    for row,(tag,arms) in enumerate(subsets):
        for col,d in enumerate(DOMAINS):
            a=ax[row,col]
            records=results[tag][d]['arms']
            v=[100*records[k]['success']/records[k]['n'] for k in arms]
            y=np.arange(len(v))[::-1]
            a.barh(y,v,color=['#a6aaa9' if k in ('actor','fixed','stage23') else '#287c80' for k in arms])
            a.set_yticks(y,[labels[k] for k in arms] if col==0 else [])
            a.set_xlim(0,100)
            for yi,val in zip(y,v):
                a.text(val+1,yi,f'{val:.1f}',va='center',fontsize=8)
            a.set_title(('Push-T' if col==0 else 'CausalWorld')+f" | n={records['actor']['n']}")
            a.spines[['top','right']].set_visible(False)
            if row:
                a.set_xlabel('Success (%)')
    fig.tight_layout()
    fig.savefig(assets/'24-native-control.png',dpi=165)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,d in zip(axes,DOMAINS):
        group=results['anchored'][d]['arms']
        keys=['marginal','anchor_marginal','anchor_mean','anchor_law','anchor_kl']
        label=['Direct marginal','Anchored marginal','Anchored pair mean','Anchored CE','Anchored CE + KL']
        idx=np.arange(len(keys))
        ax.barh(idx-.16,[group[k]['versus']['actor']['rescued'] for k in keys],.32,color='#227b80',label='Rescued')
        ax.barh(idx+.16,[group[k]['versus']['actor']['broken'] for k in keys],.32,color='#bb6852',label='Previously successful, now failed')
        ax.set_yticks(idx,label)
        ax.invert_yaxis()
        ax.set_xlabel('Layouts / 512')
        ax.set_title('Push-T' if d=='pusht' else 'CausalWorld')
        ax.spines[['top','right']].set_visible(False)
    axes[-1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(assets/'24-rescues-and-breaks.png',dpi=165)
    plt.close(fig)
    pred=json.loads((OUT/'long.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(8,3.4))
    for a,d in zip(axes,DOMAINS):
        fields=['position','position_delta']
        for j,k in enumerate(['before','observed']):
            a.bar(np.arange(2)+.32*j,[pred[d][k][f] for f in fields],.32,label=['Before observation','After observation'][j],
                  color=['#9eb6c6','#227b80'][j])
        a.set_xticks([.16,1.16],['Future position','Two-policy difference'])
        a.set_ylabel('RMSE ('+('px' if d=='pusht' else 'mm')+')')
        a.set_title('Push-T' if d=='pusht' else 'CausalWorld')
        a.spines[['top','right']].set_visible(False)
        a.legend(fontsize=8,frameon=False)
    fig.tight_layout()
    fig.savefig(assets/'24-midpoint-prediction.png',dpi=165)
    plt.close(fig)

if __name__=='__main__':
    result={k:summarize(a) for k,a in [('first',False),('anchored',True)]}
    (OUT/'summary.json').write_text(json.dumps(result,indent=1,allow_nan=False))
    figures(result)
