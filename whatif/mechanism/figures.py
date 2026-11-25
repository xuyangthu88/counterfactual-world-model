"""Render stage figures from saved metrics and complete native episode records."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font = Path.home()/'.local/share/fonts/NotoSansCJKsc-Regular.otf'
if font.is_file():
    font_manager.fontManager.addfont(str(font))
from .train import OUT,ROOT

FIG=ROOT/'reports/stages/assets'
colors=['#176D72','#DB9553','#6874A8','#BC6482','#74827D','#477B9E']
plt.rcParams.update({'font.family':['Noto Sans CJK SC','DejaVu Sans'],'axes.spines.top':False,
                     'axes.spines.right':False,'font.size':10,'axes.unicode_minus':False})


def bars(ax,labels,values,title,fmt='.2f'):
    bars=ax.bar(range(len(values)),values,color=colors[:len(values)],width=.67)
    ax.set_xticks(range(len(values)),labels,rotation=12)
    ax.set_title(title)
    ax.bar_label(bars,labels=[format(x,fmt) for x in values],padding=3,fontsize=9)
    ax.set_ylim(0,max(.01,max(values)*1.22))
    ax.grid(axis='y',alpha=.14)
    ax.set_axisbelow(True)


def save(fig,file):
    fig.tight_layout()
    fig.savefig(FIG/file,dpi=155,bbox_inches='tight')
    plt.close(fig)


def control_summary(rows):
    summary={}
    for d in ['causalworld','pusht']:
        sub={}
        for name in sorted({row['policy'] for row in rows}):
            part=sorted([r for r in rows if r['domain']==d and r['policy']==name],key=lambda r:r['seed'])
            if not part:
                continue
            sub[name]={key:float(np.mean([r[key] for r in part])) for key in ('terminal','peak','moves')}
            sub[name].update(n=len(part),success=sum(r['success'] for r in part))
        summary[d]=sub
    return summary


def controls(rows,file):
    summary=control_summary(rows)
    policies=['value','delta','delta_kl','pets','cophynet']
    if 'actor' in summary['pusht']:
        policies=policies+['actor']
    names={'value':'边际','delta':'配对','delta_kl':'配对+KL','pets':'PETS','cophynet':'CoPhyNet','actor':'生成器'}
    fig,axes=plt.subplots(2,2,figsize=(9,5.7))
    for side,(d,table) in enumerate(summary.items()):
        dom={'causalworld':'CausalWorld','pusht':'Push-T'}[d]
        n=next(iter(table.values()))['n']
        bars(axes[side,0],[names[p] for p in policies],[table[p]['success'] for p in policies],f'{dom}  达标布局数（共 {n} 个）','.0f')
        bars(axes[side,1],[names[p] for p in policies],[table[p]['peak'] for p in policies],f'{dom}  回合最高任务分')
    save(fig,file)
    return summary


def trajectories(rows):
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,domain in zip(axes,['causalworld','pusht']):
        for idx,(policy,label) in enumerate([('actor','生成器'),('value','边际'),('delta_kl','配对+KL')]):
            row=next(r for r in rows if r['domain']==domain and r['policy']==policy and 'trace' in r)
            trace=row['trace']
            x=np.asarray(trace['poses'])
            ax.plot(x[:,0],x[:,1],'.-',color=colors[idx],label=label,alpha=.75,markersize=3)
            ax.scatter(x[-1,0],x[-1,1],marker='s',color=colors[idx],s=25)
        ax.scatter(*trace['goal'][:2],s=130,c='#222222',marker='*',label='目标中心')
        ax.set_title(domain+' · 首个测试布局')
        ax.set_aspect('equal',adjustable='datalim')
        ax.set_xlabel('x（物体长度）')
        ax.set_ylabel('y（物体长度）')
        ax.legend(fontsize=8)
    save(fig,'19-native-paths.png')


def main():
    graph=json.loads((OUT/'edges.json').read_text())
    pred=json.loads((OUT/'prediction.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(8.5,3))
    labels=['共享图','NCII','NCII+接触']
    for ax,edge,title in zip(axes,['contact','excision'],['接触过的积木对','切断作用后接收块发生位移']):
        bars(ax,labels,[graph['cophy'][m][edge]['ap'] for m in ['mechanism','ncii','ncii_contact']],title+' · AP')
    save(fig,'15-mechanism.png')
    fig,axes=plt.subplots(2,2,figsize=(9,5.6))
    for idx,(d,unit) in enumerate([('causalworld','mm'),('pusht','px')]):
        bars(axes[idx,0],labels,[graph[d][m]['contact']['ap'] for m in ['mechanism','ncii','ncii_contact']],d+'  入边 AP')
        bars(axes[idx,1],['共享图','CoPhyNet','PETS'],[pred[d][m]['effect_position_rmse'] for m in
             ('mechanism','cophynet','pets')],d+f'  动作位移差 RMSE（{unit}）','.1f')
    save(fig,'16-mechanism.png')
    if (OUT/'control.json').exists():
        s=controls(json.loads((OUT/'control.json').read_text()),'17-mechanism.png')
        (OUT/'control-summary.json').write_text(json.dumps(s,indent=2))


if __name__=='__main__':
    main()
