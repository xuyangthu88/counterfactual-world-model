import argparse
import json
import torch
from .evaluate import metrics,graph
from .figures import controls, trajectories
from .model import labeled
from .train import OUT,read,build,goal_training


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--proposal',action='store_true')
    args=parser.parse_args()
    folder=OUT/('demonstrations' if args.proposal else 'fine')
    torch.set_num_threads(3)
    metrics_by_task={}
    for dom in ['causalworld','pusht']:
        val,test=[goal_training(read(s,'cpu',[dom],fine=not args.proposal,proposal=args.proposal),False) for s in ('val','test')]
        scores={}
        for name in ['mechanism','cophynet','pets']:
            model=build(name,'cpu')
            model.load_state_dict(torch.load(folder/f'{name}.pt',map_location='cpu'))
            model.eval()
            result=metrics(name,model,test,dom)
            if name=='mechanism':
                threshold=None
                for d in [val,test]:
                    o=model(d)
                    mask=(labeled(d['role'])[:,None]*d['role'][:,None,None,:,0]).bool().expand_as(d['e'])
                    edges=graph(d['e'][mask].numpy(),o['edge'].sigmoid()[mask].numpy(),threshold)
                    threshold=edges['threshold']
                result['contact']=edges
            scores[name]=result
        metrics_by_task[dom]=scores
    (folder/'prediction.json').write_text(json.dumps(metrics_by_task,indent=2))
    rows=json.loads((folder/'control.json').read_text())
    summary=controls(rows,'19-learned-actions.png' if args.proposal else '18-fine-control.png')
    if args.proposal:
        trajectories(rows)
    (folder/'control-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=1))


if __name__=='__main__':
    main()

