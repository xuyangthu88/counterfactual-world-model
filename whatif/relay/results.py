"""Paired native controls and Stage 26 figures."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from whatif.hitting.results import chart_style,compare
from .native import OUT,ROOT,DOMAINS,CONTROL
from .policy import MODES
ASSETS=ROOT/'reports/stages/assets'
NAMES={'pusht':'Push-T','causalworld':'CausalWorld'}

def load(name):
    return json.loads((OUT/f'{name}.json').read_text())

def summarize():
    n=CONTROL[1]
    ix=np.random.default_rng(271).integers(n,size=(5000,n))
    records={}
    plans=load('second')
    references=load('references')
    selection=load('selection')
    for dom,arms in load('episodes').items():
        vals={m:np.asarray(v) for m,v in arms.items()}
        for m in ('pets','stage21'):
            rows=sorted((v for v in references if v['domain']==dom and v['policy']==m),key=lambda v:v['seed'])
            if [r['seed'] for r in rows]!=list(range(CONTROL[0],sum(CONTROL))):
                raise ValueError('reference layout mismatch')
            vals[m]=np.array([[r['success'],r['peak'],r['terminal']] for r in rows])
        primary=selection['primary']
        def contrast(a,b):
            c=compare(vals[a][:,0],vals[b][:,0],ix)
            if a in plans[dom] and b in plans[dom]:
                v,w=np.asarray(plans[dom][a]),np.asarray(plans[dom][b])
                c['different_plans']=int((v!=w).any(1).sum())
            return c
        comp={primary+'-'+m:contrast(primary,m) for m in ('actor','fixed_policy','stage25','stage25belief','stage24','stage21','pets')}
        for prefix in ('both_','hold_','late_'):
            for a,b in [('means','direct'),('independent','means'),('law','independent'),('law','means'),('kl','law')]:
                comp[prefix+a+'-'+b]=contrast(prefix+a,prefix+b)
            if prefix!='late_':
                for base in ('fixed','unpaired'):
                    comp[prefix+'direct-'+base]=contrast(prefix+'direct',prefix+base)
        table={k:dict(success=int(v[:,0].sum()),peak=float(v[:,1].mean()),last=float(v[:,2].mean()),
                      versus_generator=contrast(k,'actor')) for k,v in vals.items()}
        records[dom]=dict(n=n,arms=table,paired=comp)
    (OUT/'summary.json').write_text(json.dumps(records,indent=1))
    return records

def plots(results):
    chart_style()
    ASSETS.mkdir(exist_ok=True,parents=True)
    ex=load('example')
    t=np.arange(5)
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    wall=np.asarray(ex['x'])[ex['wall'],:2]*100+256
    initial=np.asarray(ex['x'])[0,:2]*100+256
    colors=['#126c70','#b9592f']
    if abs(wall[0]-256)>abs(wall[1]-256):
        ax[0].axvline(wall[0],color='#555',lw=4,label='墙')
    else:
        ax[0].axhline(wall[1],color='#555',lw=4,label='墙')
    for j,arm in enumerate(ex['arms']):
        with np.load(OUT/'example-paths.npz') as archive:
            path=archive[str(arm)]
        path=path[path[:,0]>1]
        pos=path[:,1:3]*100+256
        ax[0].plot(pos[:,0],pos[:,1],lw=1.3,color=colors[j],label=f'策略 {arm}')
        hit=path[:,5]>0
        if hit.any():
            ax[0].scatter(pos[hit,0],pos[hit,1],s=19,marker='s',color=colors[j],label='碰墙时的块中心')
        ax[1].plot(t,ex['prob'][j],'-o',ms=4,color=colors[j],label=f'策略 {arm} 预测概率')
        ax[1].scatter(t,np.array(ex['truth'][j])+(j-.5)*.025,marker=('s','x')[j],s=32,color=colors[j],label=f'策略 {arm} 实测接触',zorder=4)
    ax[0].scatter(*initial,color='black',marker='*',s=70,label='起点')
    ax[0].set(xlabel='横向位置（px）',ylabel='纵向位置（px）',title='同一布局的 T 块中心轨迹',aspect='equal')
    ax[0].invert_yaxis()
    ax[0].legend(fontsize=8)
    ax[1].axhline(.5,ls=':',color='#777')
    ax[1].set(xticks=t,xticklabels=ex['window_ends'],ylim=(-.08,1.08),xlabel='时段结束的宏动作步',ylabel='接触概率 / 实测标签',title='同一面墙到 T 块的边')
    ax[1].legend(fontsize=8,loc='upper left')
    fig.savefig(ASSETS/'26-wall.png',dpi=170)
    plt.close(fig)
    temporal=load('temporal')
    fig,axes=plt.subplots(1,2,figsize=(9.0,3.9),layout='constrained')
    for ax,dom in zip(axes,DOMAINS):
        for j,(kind,name,color) in enumerate([('fixed','首轮定图','#a6afb8'),('dynamic','逐轮建图','#19797b')]):
            v=[temporal[kind][dom+'-'+p]['contact']['birth']['ap'] for p in ('short','early','late')]
            ax.bar(np.arange(3)+(j-.5)*.34,v,.33,label=name,color=color)
        ax.set(xticks=range(3),xticklabels=['一个宏动作内','第 1 步后规划','第 9 步后规划'],ylim=(0,1),title=NAMES[dom],ylabel='新接触边 AP')
        ax.legend(fontsize=9)
    fig.savefig(ASSETS/'26-births.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.8),layout='constrained')
    methods=['actor','fixed_policy','stage25','both_direct','both_means','both_independent','both_law','both_kl']
    names=['生成器','固定反馈策略','冻结 Stage 25','直接读成功概率','读均值','读独立边际','读联合分布','联合分布与 KL']
    for ax,dom in zip(axes,DOMAINS):
        vals=np.array([results[dom]['arms'][m]['success'] for m in methods])
        ax.barh(np.arange(len(vals)),100*vals/CONTROL[1],color=['#99a7ae']*3+['#19797b']*4+['#ad6038'])
        ax.set_yticks(np.arange(len(vals)),names)
        ax.invert_yaxis()
        ax.set_xlim(0,max(vals)*100/CONTROL[1]*1.17)
        ax.set(xlabel='成功率（%）',title=NAMES[dom])
        for y,v in enumerate(vals):
            ax.text(100*v/CONTROL[1]+.35,y,str(v),va='center',fontsize=9)
    fig.savefig(ASSETS/'26-control.png',dpi=170)
    plt.close(fig)

if __name__=='__main__':
    plots(summarize())
