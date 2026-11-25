"""Stage 22 figures and matched-layout comparisons calculated from native episode records."""
import json
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from whatif.mechanism.figures import plt,save
from .data import OUT
from .fit import world
from .decision import decision_head,panel,ModalAction,targets,statistics
from .option import choose
from .evaluate import laplace_crps


def read(name):
    return json.loads((OUT/name).read_text())


def summaries(rows):
    out={}
    for domain in ('causalworld','pusht'):
        methods={}
        for metric in sorted({v['policy'] for v in rows}):
            trials=sorted([r for r in rows if r['domain']==domain and r['policy']==metric],key=lambda r:r['seed'])
            if not trials:
                continue
            result=dict(n=len(trials),success=sum(r['success'] for r in trials),
                        terminal=np.mean([r['terminal'] for r in trials]),peak=np.mean([r['peak'] for r in trials]))
            for key in ('decisions','switches','kl_changes','options','retrieved','locked_steps','teacher_switches','kl_disagreements'):
                if key in trials[0]:
                    result[key]=sum(r[key] or 0 for r in trials)
            methods[metric]=result
        out[domain]=methods
    return out


def paired(rows):
    out={}
    for dom in ('causalworld','pusht'):
        by={}
        for r in rows:
            if r['domain']==dom:
                by.setdefault(r['policy'],{})[r['seed']]=r['success']
        parts={}
        for a in by:
            for b in by:
                if a==b:
                    continue
                if by[a].keys()!=by[b].keys():
                    raise ValueError('incomplete paired panel')
                z=np.array([by[a][k]-by[b][k] for k in sorted(by[a])])
                ci=np.quantile(np.random.default_rng(271).choice(z,(4000,len(z))).mean(1),[.025,.975])
                parts[f'{a}-{b}']=dict(gain=int((z>0).sum()),loss=int((z<0).sum()),ci=ci.tolist())
        out[dom]=parts
    return out


@torch.no_grad()
def segment_metrics():
    net,head=world(),decision_head(True)
    x,y,pr=panel(net,ModalAction(),'test','cpu',True)
    config=read('segment-selection.json')['config']
    result={}
    for dom,ix in zip(['causalworld','pusht'],torch.arange(len(x)).chunk(2)):
        xx,yy,prior=x[ix],y[ix],pr[ix]
        s,aux=targets(yy)
        i,j=torch.triu_indices(4,4,1)
        o=head(xx,i,j)
        truth=(s+aux)[:,i]-(s+aux)[:,j]
        v=statistics(o)
        info=dict(effect_rmse=float((v['mean']-truth).square().mean().sqrt()),
                  crps=float(laplace_crps(o,truth).mean()))
        i=torch.arange(4)
        o=head(xx,i,torch.zeros_like(i))
        for k,index in [('rescue',2),('break',1)]:
            truth=s[:,1:]>s[:,:1] if k=='rescue' else s[:,1:]<s[:,:1]
            probs=o['mass'][:,1:,index]
            info[k]=dict(ap=float(average_precision_score(truth.flatten(),probs.flatten())),
                         rate=float(truth.float().mean()),brier=float((probs-truth.float()).square().mean()))
        info['any_retrieval_rescues']=float(((s[:,1:]>s[:,:1]).any(-1)).float().mean())
        info['reference_success']=float(s[:,0].mean())
        for policy in ['marginal','pair_mean','pair_law','pair_kl']:
            k=choose(o,prior,policy,config,batched=True)
            info[policy]=dict(actual_success_gain=float((s.gather(1,k[:,None])-s[:,:1]).mean()),
                              switch_rate=float((k!=0).float().mean()))
        result[dom]=info
    return result


def graph_figure(offline):
    data=offline['cophy']
    fig,axes=plt.subplots(1,2,figsize=(8.7,3.5))
    for ax,key,title,labels in [(axes[0],'ap','识别删边后会移动的接收块',['stage20','contact_path','transport','ncii_excision']),
                                 (axes[1],'distant_effect_rmse','相隔至少两层的位移差',['stage20','contact_path','transport'])]:
        values=[data[m]['excision']['ap'] if key=='ap' else data[m][key] for m in labels]
        bars=ax.bar(range(len(values)),values,color=['#8596ae','#e2a563','#227778','#986187'][:len(values)])
        ax.set_xticks(range(len(values)),['旧图','仅接触通路','有向通路','NCII'][:len(values)])
        ax.set_ylim(0,max(values)*1.2)
        ax.set_title(title)
        ax.set_ylabel('AP ↑' if key=='ap' else 'RMSE（积木边长）↓')
        ax.bar_label(bars,fmt='%.3f',padding=3)
    save(fig,'22-graph-transport.png')


def control_figure(table):
    keys=['actor','stage21','marginal','pair_mean','pair_law','pair_kl','combined_marginal','combined_law','combined_kl']
    labels=['演示生成器','Stage 21','片段 · 边际','片段 · 配对均值','片段 · 风险约束','片段 · 加 KL',
            '两级 · 边际','两级 · 风险约束','两级 · 加 KL']
    fig,axes=plt.subplots(1,2,figsize=(10,4.9),sharey=True)
    for ax,dom,label in zip(axes,['causalworld','pusht'],['CausalWorld','Push-T']):
        values=[table[dom][k]['success'] for k in keys]
        bar=ax.barh(np.arange(len(keys)),values,color=['#a7afb8','#6d829c','#dcab66','#83ada8','#4b9591','#216b6e','#b49abb','#9c79a3','#74547f'])
        ax.set_yticks(np.arange(len(keys)),labels)
        ax.bar_label(bar,padding=3)
        ax.set_xlim(0,132)
        ax.set_title(label)
        ax.set_xlabel('达到官方成功阈值的布局数（共 128）')
    axes[0].invert_yaxis()
    save(fig,'22-closed-loop.png')


def main():
    torch.set_num_threads(3)
    immediate=(read('control-previous.json')+read('control-new.json') if (OUT/'control-new.json').exists() and (OUT/'control-previous.json').exists()
               else read('control-all.json'))
    rows={'step':immediate,
          'commit':read('option-test.json'),'segment':read('segment-control.json')}
    result=dict(control={k:summaries(v) for k,v in rows.items()},paired={k:paired(v) for k,v in rows.items()},
                segment_offline=segment_metrics())
    graph_figure(read('offline.json'))
    control_figure(result['control']['segment'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
