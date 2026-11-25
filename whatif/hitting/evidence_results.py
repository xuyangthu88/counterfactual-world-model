"""Layout-paired performance of the independently registered observation-update experiment."""
import json
import numpy as np
from .native import OUT,DATA,DOMAINS
from .evidence_native import N
from .results import read,compare,ASSETS,NAMES,plt,chart_style

def run():
    episodes=read('evidence-episodes')
    first_episodes=read('episodes')
    plans=read('evidence-second')
    arms=('stage25','prior','moment','posterior','law','kl','selected','actor')
    ix=np.random.default_rng(271).integers(N,size=(5000,N))
    result={}
    for dom in DOMAINS:
        with np.load(DATA/f'evidence-mid-{dom}.npz') as f:
            active=f['mid_alive'].astype(bool)
            start=f['alive'].astype(bool)
        vals={m:np.array(episodes[dom][m]) for m in arms}
        results={m:dict(success=int(x[:,0].sum()),peak=float(x[:,1].mean()),terminal=float(x[:,2].mean())) for m,x in vals.items()}
        diff={}
        for a,b in [('selected','stage25'),('selected','actor'),('posterior','prior'),('posterior','moment'),('law','posterior'),('kl','law')]:
            r=compare(vals[a][:,0],vals[b][:,0],ix)
            left,right=[np.array(plans[dom][m]) for m in (a,b)]
            first=left[:,0]!=right[:,0]
            r['path_changed']=int(((first&start)|(~first&active&(left[:,1]!=right[:,1]))).sum())
            diff[a+'-'+b]=r
        primary=first_episodes[dom]
        before=np.array(primary['selected'])[:,0]-np.array(primary['actor'])[:,0]
        after=vals['stage25'][:,0]-vals['actor'][:,0]
        rng=np.random.default_rng(271)
        sums=sum(d[rng.integers(len(d),size=(5000,len(d)))].sum(1) for d in (before,after))
        total=len(before)+len(after)
        pool=dict(n=total,stage25=int(np.array(primary['selected'])[:,0].sum()+vals['stage25'][:,0].sum()),
                  actor=int(np.array(primary['actor'])[:,0].sum()+vals['actor'][:,0].sum()),
                  points=100*(before.sum()+after.sum())/total,interval=np.quantile(100*sums/total,[.025,.975]).tolist())
        result[dom]=dict(n=N,arms=results,contrasts=diff,same_stage25_vs_actor=pool)
    (OUT/'belief-summary.json').write_text(json.dumps(result,indent=2))
    methods=['prior','moment','posterior','law','kl']
    chart_style()
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.15),layout='constrained')
    for ax,dom in zip(axes,DOMAINS):
        vals=np.array(episodes[dom]['stage25'])[:,0]
        q=[compare(np.array(episodes[dom][m])[:,0],vals,ix) for m in methods]
        means=np.array([r['points'] for r in q])
        intervals=np.array([r['interval'] for r in q])
        ax.errorbar(means,np.arange(5),xerr=[means-intervals[:,0],intervals[:,1]-means],fmt='o',capsize=3,color='#25797c')
        ax.axvline(0,color='#676974',ls=':',lw=1)
        ax.set_yticks(np.arange(5),['先验','均值残差更新','后验成功概率','后验动作差分布','动作差分布与 KL'])
        ax.invert_yaxis()
        ax.set_title(NAMES[dom]+' · 1,024 个布局')
        ax.set_xlabel('相对原 Stage 25 的成功率差（百分点）')
    fig.savefig(ASSETS/'25-observation.png',dpi=175)
    plt.close(fig)
if __name__=='__main__':
    run()
