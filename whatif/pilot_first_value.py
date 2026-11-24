"""Learn the value difference of alternative first pushes under a frozen second-push policy."""
import argparse
import csv
import json

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from .chain_model import Branches, LENGTH
from .pilot_eval import Planner
from .pilot_models import DATA, OUT, SEED, simple_panel

FAMILIES = ('corrected','dense','transport')
LENGTHS = (32, 41, 54, 69)


def group_data(group, planner, device):
    outcomes, costs = {}, {}
    for length in LENGTHS:
        d = Branches(DATA/'value'/f'value_{group}_{length}.npz', device)
        p = simple_panel(d, 0)
        if not outcomes:
            scenes = d.ids
            start = d.end[:,0].clone()
            # Source-time geometry and known target only.
            geometry = torch.cat((p.features[:,0,:-2], p.goal), dim=1).cpu().numpy()
            goal = d.goal.clone()
        else:
            np.testing.assert_array_equal(d.ids, scenes)
            torch.testing.assert_close(start, d.end[:,0])
            torch.testing.assert_close(goal, d.goal)
        values = planner.evaluate(d)
        for model in FAMILIES:
            mid, metrics = values[model]
            costs.setdefault(model, []).append(mid[:, d.probe]*1000*LENGTH)
            outcomes.setdefault(model, []).append(metrics)
    result = {}
    for model in FAMILIES:
        heuristic = np.stack(costs[model], axis=1)
        heuristic = heuristic - heuristic[:, 1:2]
        result[model] = (np.column_stack((geometry, heuristic)), heuristic,
                         np.stack(outcomes[model],axis=1))
    return scenes, result


def select(estimate, heuristic, alpha):
    return (alpha*estimate+(1-alpha)*heuristic).argmin(axis=1)


def fit_select(train, val):
    x, _, records = train
    target = records[:,:,2]-records[:,1:2,2]
    vx, vheuristic, actual = val
    best = np.inf
    for leaf in (4, 12, 24):
        forest = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=leaf,
                                     random_state=SEED, n_jobs=2).fit(x, target)
        estimate = forest.predict(vx)
        for alpha in (0., .25, .5, .75, 1.):
            chosen = select(estimate,vheuristic,alpha)
            loss = actual[np.arange(len(vx)),chosen,2].mean()
            if loss < best:
                best, selected = loss, (forest,alpha)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device',default='cpu')
    options = ap.add_mutually_exclusive_group()
    options.add_argument('--replay',action='store_true')
    options.add_argument('--fit-only',action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    planner = Planner('adapt',args.device)
    if not args.replay:
        _, train = group_data('fit', planner, args.device)
        _, val = group_data('val', planner, args.device)
        for model in FAMILIES:
            forest,alpha = fit_select(train[model],val[model])
            joblib.dump((forest,alpha),OUT/f'value-{model}.joblib',compress=3)
    if args.fit_only:
        return
    ids,test = group_data('test', planner,args.device)
    ix = np.arange(len(ids))
    boot = np.random.default_rng(11).integers(len(ix),size=(3000,len(ix)))
    rows, results, ends = [], {}, {}
    for model in FAMILIES:
        forest,alpha = joblib.load(OUT/f'value-{model}.joblib')
        features,heuristic,actual = test[model]
        choice = select(forest.predict(features),heuristic,alpha)
        mid = heuristic.argmin(axis=1)
        for name, first in [('midpoint',mid),('value',choice)]:
            chosen = actual[ix,first]
            ends[model,name]=chosen[:,2]
            results[model+'/'+name] = dict(distance_mm=float(chosen[:,2].mean()),success=float(chosen[:,3].mean()),
                second_regret_mm=float(chosen[:,4].mean()),alpha=alpha,leaf=forest.min_samples_leaf)
            for i, row in enumerate(chosen):
                rows.append(dict(model=model,rule=name,scene=int(ids[i]),first_mm=LENGTHS[first[i]],
                                 second=int(row[-1]),end_mm=float(row[2]),success=int(row[3])))
    for model in FAMILIES:
        delta = ends[model,'value']-ends[model,'midpoint']
        results[model+'/change_mm'] = [float(delta.mean()),*np.quantile(delta[boot].mean(axis=1),[.025,.975]).tolist()]
    for other in ('dense','transport'):
        delta = ends['corrected','value']-ends[other,'value']
        results['corrected minus '+other] = [float(delta.mean()),*np.quantile(delta[boot].mean(axis=1),[.025,.975]).tolist()]
    (OUT/'value_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    with (OUT/'value_decisions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__=='__main__':
    main()
