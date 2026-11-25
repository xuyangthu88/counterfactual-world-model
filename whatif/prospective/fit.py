import argparse
import json
import torch
from .filter import OUT,fit_data,calibration_score
from whatif.shared.planning import Models


@torch.no_grad()
def fit(device):
    m=Models(device);tasks=['causalworld','pusht']
    batches=[fit_data(m,t,'fit') for t in tasks]
    records=[]
    for sig in (.08,.16,.30):
        for strength in (.25,.60,.95):
            for g in (0.,.5,1.):
                cfg=dict(sigma=sig,strength=strength,gain=g)
                vals=[calibration_score(b,cfg) for b in batches]
                records.append(dict(cfg=cfg,values=vals))
    ref=[calibration_score(b,dict(sigma=.16,strength=0,gain=0)) for b in batches]
    def objective(r):
        return sum(.55*x[0]/max(y[0],.01)+.45*x[1]/max(y[1],.01) for x,y in zip(r['values'],ref))/2
    for r in records:r['score']=objective(r)
    records.sort(key=lambda r:r['score'])
    # Validate both a learned-strength mass filter and the transported contact correction.
    chosen={label:min((r for r in records if (r['cfg']['gain']>0)==use),key=objective)['cfg']
            for label,use in [('mass',False),('gain',True)]}
    (OUT/'fit.json').write_text(json.dumps(dict(candidates=records,shortlist=chosen),indent=2)+'\n')
    print(chosen)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();torch.set_num_threads(2);fit(a.device)
