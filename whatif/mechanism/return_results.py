import json
import numpy as np
import torch
from .figures import plt,bars,save,control_summary
from .return_model import DEST,BINS,RANGE,backbone,load_head,panel


@torch.no_grad()
def main():
    torch.set_num_threads(3)
    rows=json.loads((DEST/'control.json').read_text())
    summary=control_summary(rows)
    order=['actor','previous','value','delta','delta_kl','pets']
    labels=['生成器','单步配对','六步边际','六步配对','配对+KL','PETS+KL']
    fig,axes=plt.subplots(2,2,figsize=(10,6.4))
    comparisons={}
    for row,(dom,values) in enumerate(summary.items()):
        bars(axes[row,0],labels,[values[k]['success'] for k in order],dom+' · 128 个布局的成功数','.0f')
        bars(axes[row,1],labels,[values[k]['peak'] for k in order],dom+' · 回合最高原生分')
        comparisons[dom]={}
        def paired(name):
            sample=sorted([r for r in rows if r['domain']==dom and r['policy']==name],key=lambda r:r['seed'])
            return np.array([r['seed'] for r in sample]),np.array([r['success'] for r in sample])
        for a,b in [('delta','actor'),('delta','value'),('delta_kl','delta'),('delta_kl','pets'),('delta','previous')]:
            sa,va=paired(a)
            sb,vb=paired(b)
            assert np.array_equal(sa,sb)
            diff=va-vb
            bounds=np.quantile(np.random.default_rng(71).choice(diff,(4000,len(diff))).mean(-1),[.025,.975])
            comparisons[dom][a+'-'+b]=dict(improved=int((diff>0).sum()),lost=int((diff<0).sum()),
                                           difference=float(diff.mean()),interval=bounds.tolist())
    save(fig,'21-six-step-control.png')
    (DEST/'summary.json').write_text(json.dumps(dict(control=summary,paired=comparisons),indent=2))
    figure,axes=plt.subplots(1,2,figsize=(9,3.4))
    x,g,_=panel(backbone('graph'),'test')
    model=load_head('graph')
    i,j=torch.triu_indices(4,4,1)
    for ax,dom,ix in zip(axes,['CausalWorld','Push-T'],torch.arange(len(x)).chunk(2)):
        _,probs=model(x[ix],i,j)
        p=probs.exp().mean((0,1))
        effect=g[ix][:,i]-g[ix][:,j]
        empirical=(1-(effect[...,None]-model.bins).abs()/(2*RANGE/(BINS-1))).clamp_min(0).mean((0,1))
        ax.bar(model.bins.numpy(),empirical.numpy(),width=.095,color='#b7cbcf',label='实测频率')
        ax.plot(model.bins.numpy(),p.numpy(),'o-',color='#944c33',markersize=3,label='模型平均概率')
        ax.set_xlabel('换动作带来的六步回报差')
        ax.set_ylabel('概率')
        ax.set_title(dom)
        ax.legend()
    save(figure,'21-return-distributions.png')


if __name__=='__main__':
    main()
