"""Paired control and matched eight-program forecast of the complete follow-up checkpoint."""
import json
import numpy as np
import torch
from whatif.hitting.results import chart_style,compare,plt
from .native import OUT,ROOT,DOMAINS,FOLLOWUP
from .backup import records,menu,world
from .forecast import evaluate

def run():
    torch.set_num_threads(2)
    n=FOLLOWUP[1]
    ix=np.random.default_rng(271).integers(n,size=(5000,n))
    episode=json.loads((OUT/'backup-episodes.json').read_text())
    primary=json.loads((OUT/'episodes.json').read_text())
    plans=json.loads((OUT/'backup-second.json').read_text())
    name='backup_'+json.loads((OUT/'recourse-selection.json').read_text())['mode']
    net=world('cuda')
    table={}
    for dom in DOMAINS:
        values={m:np.array(q) for m,q in episode[dom].items()}
        arms={m:dict(success=int(v[:,0].sum()),peak=float(v[:,1].mean()),last=float(v[:,2].mean())) for m,v in values.items()}
        def contrast(a,b):
            result=compare(values[a][:,0],values[b][:,0],ix)
            left,right=[np.array(plans[dom][m]) for m in (a,b)]
            result['different_plans']=int((left!=right).any(1).sum())
            return result
        comparisons={name+'-'+m:contrast(name,m) for m in ('both_kl','stage25','stage25belief','late_law','stage24','actor','fixed_policy')}
        for a,b in [('means','direct'),('independent','means'),('law','independent'),('law','means'),('kl','law')]:
            comparisons['backup_'+a+'-'+b]=contrast('backup_'+a,'backup_'+b)
        d=menu(records(dom,'test','cuda'))
        forecast=evaluate(net,d,'recourse',100 if dom=='pusht' else 65)
        cohorts=[np.array(source[dom]['late_law'])[:,0]-np.array(source[dom]['stage25'])[:,0] for source in (primary,episode)]
        rng=np.random.default_rng(271)
        draws=sum(v[rng.integers(len(v),size=(5000,len(v)))].sum(1) for v in cohorts)
        size=sum(map(len,cohorts))
        pooled=dict(n=size,points=100*sum(v.sum() for v in cohorts)/size,
                    interval=np.quantile(100*draws/size,[.025,.975]).tolist())
        table[dom]=dict(arms=arms,paired=comparisons,forecast=forecast,same_first_pooled=pooled)
    (OUT/'backup-summary.json').write_text(json.dumps(table,indent=1))
    chart_style()
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.8),layout='constrained')
    methods=['actor','stage25','both_kl','late_law','backup_direct','backup_means','backup_law','backup_kl']
    names=['生成器','冻结 Stage 25','未拟合改选后果','Stage 25 首选＋新后段','拟合改选后果，读成功率','拟合后，读均值','拟合后，读效果分布','拟合后，分布与 KL']
    for ax,dom in zip(axes,DOMAINS):
        v=[table[dom]['arms'][m]['success'] for m in methods]
        ax.barh(np.arange(len(v)),100*np.array(v)/n,color=['#a2acb6']*2+['#69829a']*2+['#af6339']+['#19797b']*3)
        ax.invert_yaxis()
        ax.set_yticks(np.arange(len(v)),names)
        ax.set_xlim(0,max(v)*117/n)
        ax.set(xlabel='闭环成功率（%）',title='Push-T' if dom=='pusht' else 'CausalWorld')
        for y,value in enumerate(v):
            ax.text(100*value/n+.3,y,str(value),va='center',fontsize=9)
    fig.savefig(ROOT/'reports/stages/assets/26-recourse.png',dpi=170)
    plt.close(fig)
if __name__=='__main__':
    run()
