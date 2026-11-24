"""Describe observed contact paths and reachable goals on each held-out split."""
import json
from pathlib import Path
import numpy as np


def describe(path):
    d=np.load(path)
    n=int(d['count'][0])
    phase=d['edges'][:,0]  # scene, action, phase, sender, receiver
    hand=phase[:,:,:,0,:n]
    chain=[hand[:,:,:,0]]+[phase[:,:,:,i,i] for i in range(1,n)]
    first=[np.where(edge.any(axis=-1),edge.argmax(axis=-1),99) for edge in chain]
    ordered=np.logical_and.reduce([x<99 for x in first])
    for x,y in zip(first,first[1:]):
        ordered &= x<=y
    ordered &= ~hand[:,:,:,1:].any(axis=(-1,-2))
    if n==3:
        ordered &= ~phase[:,:,:,1,2].any(axis=-1)
    endpoint=d['states'][:,1,:,-1,n-1,:2]
    dist=np.linalg.norm(endpoint-d['goal'][:,None],axis=-1)
    dx=d['states'][:,0,:,-1,n-1,:2]-d['states'][:,0,:,0,n-1,:2]
    changes=dx-dx[:,1:2]
    return dict(scenes=len(d['seed']),branches=int(np.prod(d['states'].shape[:3])),
                ordered_path_branch_rate=float(ordered.mean()),
                ordered_path_any_action=float(ordered.any(axis=-1).mean()),
                ordered_path_switch=float((ordered.any(axis=-1)&(~ordered).any(axis=-1)).mean()),
                target_effect_mm=float(np.linalg.norm(changes[:,np.arange(dx.shape[1])!=1],axis=-1).mean()*1000),
                second_oracle_distance_mm=float(dist.min(axis=-1).mean()*1000),
                second_oracle_success=float((dist.min(axis=-1)<=.012).mean()))


if __name__=="__main__":
    result={name:describe(Path("data/chains")/(name+".npz")) for name in ("two","three","physical")}
    target=Path("results/chains/scene_summary.json")
    target.parent.mkdir(exist_ok=True,parents=True)
    target.write_text(json.dumps(result,indent=2)+"\n")
    print(target.read_text())
