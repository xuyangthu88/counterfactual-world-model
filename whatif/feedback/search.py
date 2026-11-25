"""Search a low-dimensional, pose-conditioned policy output layer on training layouts."""
import json
import numpy as np
from .native import DOMAINS, OUT, episode

def measure(pool, params, base, count):
    items = [(d,s,'train',p,'score') for p in params for d in DOMAINS for s in range(base,base+count)]
    y = np.stack(pool.map(episode,items,chunksize=3)).reshape(len(params),2,count,3)
    return (y @ [1.,.5,.25]).mean(2), y[...,0].mean(2)

def expand(pool):
    rng = np.random.default_rng(271)
    initial = np.array(json.loads((OUT/'bank.json').read_text())['bank'])
    archive = list(np.pad(initial, ((0,0),(0,5))))
    history = []
    # Tradeoff weights maintain candidate diversity; every candidate is deployed in both domains.
    for line, mix in enumerate(([.85,.15],[.15,.85])):
        center = archive[0 if line==0 else 3].copy()
        std = np.array([.34,.35,.15,.15,.12,.3,.38,.38,.24,.19,.19])
        for gen in range(4):
            noise = rng.normal(size=(8,len(center)))
            weights = np.clip(center + np.r_[noise,-noise]*std, -1.3,1.3)
            score, succ = measure(pool, weights, 4000000+1000*(4*line+gen), 40)
            order = np.argsort(-(score @ mix))
            elite = weights[order[:4]]
            center = elite.mean(0)
            std = np.maximum(.65*std+.35*elite.std(0),.045)
            archive.extend([weights[order[0]],center.copy()])
            history.append(dict(line=line,generation=gen,weights=weights.tolist(),score=score.tolist(),success=succ.tolist()))
            print('response search',line,gen,score[order[0]],flush=True)
    pts = np.stack(archive)
    score, succ = measure(pool,pts,4100000,144)
    ids = [0]
    for rank in range(len(pts)):
        for mix in ([.9,.1],[.1,.9],[.5,.5]):
            ix = int(np.argsort(-(score@mix))[rank])
            if ix not in ids:
                ids.append(ix)
            if len(ids)==7:
                result = dict(bank=pts[ids].tolist(),ids=ids,archive=pts.tolist(),screen_score=score.tolist(),
                              screen_success=succ.tolist(),history=history)
                (OUT/'expanded-bank.json').write_text(json.dumps(result,indent=1))
                print('shared menu',ids,succ[ids],flush=True)
                return
