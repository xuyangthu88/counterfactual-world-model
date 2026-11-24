"""Run three frozen controllers in native Pushing under official P1/P2 mass interventions."""
import argparse
import csv
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from .chain_model import Branches
from .pilot_eval import Planner
from .pilot_models import OUT, DATA


def as_panel(batch, indices, probe, device, follow=False):
    n = len(indices)
    buf = io.BytesIO()
    # Protocol truth is kept in the simulator output, outside the planning features.
    states = batch['states'][indices].copy()
    tips = batch['tips'][indices].copy()
    goal = batch['goals'][indices].copy()
    frame = batch['angle']
    if follow:
        frame = 0.
        for i, angles in enumerate(batch['frames'][indices]):
            for t, angle in enumerate(angles):
                c, s = np.cos(angle), np.sin(angle)
                rotation = np.array([[c, -s], [s, c]])
                states[i,t,...,:2] = states[i,t,...,:2] @ rotation
                states[i,t,...,3:5] = states[i,t,...,3:5] @ rotation
                yaw = states[i,t,...,2] - angle
                states[i,t,...,2] = np.arctan2(np.sin(yaw), np.cos(yaw))
                tips[i,t,:2] = tips[i,t,:2] @ rotation
            goal[i] = goal[i] @ rotation  # second action is scored in the current frame
    np.savez(buf, states=states, tips=tips,
             goal=goal, count=np.ones(n, dtype=int), seed=indices,
             frame=frame, probe=probe, actions=batch['actions'],
             physics=np.full((n, 2), np.nan), noise=np.full((n, 2), np.nan))
    buf.seek(0)
    return Branches(buf, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-python', required=True)
    ap.add_argument('--layouts', type=int, default=32)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--replay', action='store_true', help='Use previously saved native branch panels')
    ap.add_argument('--mode', choices=['original','fixed','reach'], default='original')
    args = ap.parse_args()
    follow = args.mode == 'reach'
    suffix = '' if args.mode == 'original' else '_'+args.mode
    offset = 0 if args.mode == 'original' else 6000
    torch.set_num_threads(2)
    planner = Planner('adapt', args.device)
    families = ('corrected', 'dense', 'transport')
    tasks = [(p,s+offset+i) for p,s in [('P1',47000),('P2',48000)] for i in range(args.layouts)]
    histories = {(model,p,s): [5] for model in families for p,s in tasks}
    last_pose = {}
    rows = []
    path = DATA/('native'+suffix)
    stopped = set()
    path.mkdir(parents=True, exist_ok=True)
    for step in range(2, 6):
        keys = sorted({(p,s,tuple(hist)) for (m,p,s),hist in histories.items() if (m,p,s) not in stopped})
        if not keys:
            break
        requested = [[p,s,list(h)]+(['reach'] if follow else []) for p,s,h in keys]
        jobs = path/f'step{step}_jobs.json'
        file = path/f'step{step}.npz'
        if args.replay:
            assert json.loads(jobs.read_text()) == requested
        else:
            jobs.write_text(json.dumps(requested)+'\n')
            subprocess.run([args.sim_python, '-m', 'whatif.native_branches', str(jobs), str(file),
                            '--workers', str(args.workers)], check=True,
                           env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        with np.load(file) as raw:
            batch = {k: raw[k] for k in raw.files}
        by_key = {k:i for i,k in enumerate(keys)}
        actions = {model: np.empty(len(keys), dtype=int) for model in families}
        for probe in sorted({k[2][-1] for k in keys}):
            which = np.array([i for i,k in enumerate(keys) if k[2][-1] == probe])
            d = as_panel(batch, which, probe, args.device, follow)
            predicted = planner.evaluate(d)
            for model in families:
                actions[model][which] = predicted[model][1][:,-1].astype(int)
        for (model,p,s), hist in histories.items():
            key = (model,p,s)
            if key in stopped:
                continue
            i = by_key[p,s,tuple(hist)]
            start = batch['states'][i,1,0,0,0,:2]
            if key in last_pose:
                np.testing.assert_allclose(last_pose[key], start, atol=2e-5, rtol=0)
            a = int(actions[model][i])
            xy = batch['states'][i,1,a,-1,0,:2]
            goal = batch['goals'][i]
            if step == 2:
                rows.append(dict(model=model, protocol=p, scene=s, step=1, action=5, mass=batch['masses'][i],
                                 x=start[0], y=start[1], distance_mm=np.linalg.norm(start-goal)*1000,
                                 fractional_success=batch['start_scores'][i]))
            rows.append(dict(model=model, protocol=p, scene=s, step=step, action=a, mass=batch['masses'][i],
                             x=xy[0], y=xy[1], distance_mm=np.linalg.norm(xy-goal)*1000,
                             fractional_success=batch['scores'][i,a]))
            if follow and batch['scores'][i,a] > .9:
                stopped.add(key)
            last_pose[key] = xy
            hist.append(a)
        print('completed push',step,'unique prefixes',len(keys),flush=True)
    with (OUT/('native'+suffix+'_per_step.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    result={}
    for p in ('P1','P2'):
        for model in families:
            trace = [r for r in rows if r['model']==model and r['protocol']==p]
            group = list({r['scene']: r for r in trace}.values())
            result[p+'/'+model]={k:float(np.mean([r[k] for r in group])) for k in ('distance_mm','fractional_success')}
            result[p+'/'+model]['layouts']=len(group)
            if suffix:
                result[p+'/'+model]['mean_pushes']=float(np.mean([r['step'] for r in group]))
                result[p+'/'+model]['success']=float(np.mean([r['fractional_success']>.9 for r in group]))
    (OUT/('native'+suffix+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
