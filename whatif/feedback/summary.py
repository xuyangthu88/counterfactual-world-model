"""Layout-matched control counts, paired differences, and readable stage figures."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .native import DATA,OUT,DOMAINS,ROOT

ASSETS=ROOT/'reports/stages/assets'
ORDER=('actor','population','marginal','mean','law','kl','stage21','pets')
NAMES=('Generator','Fixed learned','Marginal','Pair mean','Pair law','Law + KL','Stage 21','PETS')

def compare(a,b):
    diff=a[:,0]-b[:,0]
    rng=np.random.default_rng(271)
    idx=rng.integers(0,len(diff),(5000,len(diff)))
    return dict(rescues=int((diff>0).sum()),regressions=int((diff<0).sum()),
                gain=diff.mean(),interval=np.quantile(diff[idx].mean(1),[.025,.975]).tolist())

def main():
    refs=json.loads((OUT/'references.json').read_text())
    result={}
    for prefix in ('','history-'):
        tag='late' if prefix else 'early'
        all_runs=json.loads((OUT/(prefix+'episodes.json')).read_text())
        choices=json.loads((OUT/(prefix+'choices.json')).read_text())
        result[tag]={}
        for d in DOMAINS:
            with np.load(DATA/(f'history-{d}-control.npz' if prefix else f'{d}-observe.npz')) as obs:
                alive=obs['alive'].astype(bool)
                seeds=obs['seed']
            arms={k:np.array(v) for k,v in all_runs[d].items()}
            for name in ('stage21','pets'):
                rows=sorted((r for r in refs if r['domain']==d and r['policy']==name),key=lambda r:r['seed'])
                if not np.array_equal([r['seed'] for r in rows],seeds):
                    raise ValueError('reference layout mismatch')
                arms[name]=np.array([[r['success'],r['peak'],r['terminal']] for r in rows])
            if prefix and not np.allclose(arms['actor'],np.array(json.loads((OUT/'episodes.json').read_text())[d]['actor'])):
                raise ValueError('generator replay mismatch')
            arms_summary={}
            for name in ORDER:
                z=arms[name]
                desc=dict(success=int(z[:,0].sum()),n=len(z),peak=float(z[:,1].mean()),
                          terminal=float(z[:,2].mean()),G=float((z@[1,.5,.25]).mean()))
                for reference in ('actor','population','marginal','mean','law','stage21'):
                    desc['vs_'+reference]=compare(z,arms[reference])
                if name in choices[d]:
                    choice=np.asarray(choices[d][name])
                    desc['counts']=np.bincount(choice[alive],minlength=7).tolist()
                arms_summary[name]=desc
            selections=choices[d]
            decisions=sum(len({v[i] for v in selections.values()}) for i in range(len(seeds)))
            disagreements={k:int(((np.array(selections[k])!=selections['kl'])&alive).sum())
                           for k in ('law','mean','marginal')}
            result[tag][d]=dict(arms=arms_summary,distinct_native_episodes=decisions,
                                kl_disagreements=disagreements,live=int(alive.sum()))
    # Both test/control use the same official reset protocol and maximum of 36 macros.
    # The full-branch cohort evaluates the selected policy from independently executed rollouts.
    with np.load(DATA/'expanded-pusht-test.npz') as panel:
        offline_base=int(panel['outcome'][:,0,-1,0].sum())
        n=len(panel['seed'])
    off=json.loads((OUT/'long.json').read_text())['pusht']['rescue_counts']['mean']
    live=result['early']['pusht']['arms']['mean']['vs_actor']
    rng=np.random.default_rng(271)
    sums=[]
    for size,rescue,broken in [(192,live['rescues'],live['regressions']),(n,off['rescues'],off['breaks'])]:
        delta=np.r_[np.ones(rescue),-np.ones(broken),np.zeros(size-rescue-broken)]
        sums.append(delta[rng.integers(0,size,(5000,size))].sum(1))
    result['early_pusht_across_cohorts']=dict(n=192+n,
        generator=offline_base+result['early']['pusht']['arms']['actor']['success'],
        paired_mean=offline_base+off['rescues']-off['breaks']+result['early']['pusht']['arms']['mean']['success'],
        rescued=live['rescues']+off['rescues'],broken=live['regressions']+off['breaks'],
        interval=np.quantile(sum(sums)/(192+n),[.025,.975]).tolist())
    (OUT/'summary.json').write_text(json.dumps(result,indent=1,allow_nan=False))
    figure(result)

def figure(summary):
    ASSETS.mkdir(exist_ok=True)
    fig,axs=plt.subplots(2,2,figsize=(10,7.4),sharex=True)
    for row,tag in enumerate(('early','late')):
        for col,d in enumerate(DOMAINS):
            ax=axs[row,col]
            scores=summary[tag][d]['arms']
            vals=[scores[k]['success'] for k in ORDER]
            y=np.arange(len(vals))[::-1]
            ax.barh(y,vals,color=['#a1a9ac','#64847f','#85a8bb','#19718c','#21817e','#69588c','#aaaaaa','#c5c5c5'])
            ax.set_yticks(y,NAMES if col==0 else [])
            for yi,v in zip(y,vals):
                ax.text(v+2,yi,str(v),va='center',fontsize=9)
            ax.set_xlim(0,192)
            ax.set_title(('Push-T' if d=='pusht' else 'CausalWorld')+' | '+('1' if row==0 else '18')+' observed macro-step'+('' if row==0 else 's'))
            if row:
                ax.set_xlabel('Successful layouts / 192')
            ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS/'23-feedback-control.png',dpi=165)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.7))
    for ax,d in zip(axs,DOMAINS):
        for prefix,style,title in [('', 'o-', 'Early intervention'),('history-', 's--', 'Midpoint intervention')]:
            records=json.loads((OUT/(prefix+'long.json')).read_text())[d]['horizons']
            horizon=list(map(int,records))
            ax.plot(horizon,[v['difference_rmse'] for v in records.values()],style,label=title)
        ax.set_xlabel('Macro-steps after intervention')
        ax.set_ylabel('Paired displacement RMSE ('+('px' if d=='pusht' else 'mm')+')')
        ax.set_title('Push-T' if d=='pusht' else 'CausalWorld')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS/'23-feedback-effects.png',dpi=165)
    plt.close(fig)

if __name__=='__main__':
    main()
