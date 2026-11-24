"""Train on complete scenes; compute paired prediction and second-push control metrics."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .model import POSE_SCALE, features, pose_diff, train_model, wrap


PARTICLES = np.array(list(itertools.product((.025, .075, .14), (.18, .5, .85), (-.0025, 0., .0025))))


def predict_samples(model, data, ids, step, device):
    x = features(data["states"][ids, step, None, None], data["actions"][:, None, :],
                 PARTICLES[:, :2], PARTICLES[:, 2])
    with torch.no_grad():
        arr = torch.tensor(x, device=device)
        pred = model(arr).cpu().numpy() * POSE_SCALE
        if model.use_graph:
            gates = model.gate(arr).sigmoid().squeeze(-1).cpu().numpy()
        else:
            gates = None
    return pred, gates


def posterior(samples, observed, scale):
    error = pose_diff(samples, observed[:, None]) / scale
    logw = -.5 * np.sum(error**2, axis=-1)
    weight = np.exp(logw - logw.max(axis=1, keepdims=True))
    return weight / weight.sum(axis=1, keepdims=True)


def signed_js(success, ref):
    def entropy(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -p * np.log(p) - (1-p) * np.log1p(-p)
    return np.sign(success - ref) * (entropy((success + ref)/2) -
                                      (entropy(success) + entropy(ref))/2)


def plan(samples, weights, pos, goal, bonus=.15):
    xy = pos[:, None, None, :2] + samples[..., :2]
    dist = np.linalg.norm(xy - goal[:, None, None], axis=-1)
    near = np.sum(weights[:, None] * (dist < .012), axis=-1)
    score = -np.sum(weights[:, None] * dist, axis=-1) / .065
    return np.argmax(score + bonus * signed_js(near, near[:, 1:2]), axis=1)


def run_eval(model, data, ids, valid, seed, device):
    # Calibrate likelihood widths on separate scenes at their known P,u.
    valpred, _ = predict_samples(model, data, valid, 0, device)
    true_ksi = np.column_stack([data["physics"][valid], data["noise"][valid, 0]])
    matches = np.all(true_ksi[:, None] == PARTICLES[None], axis=-1)
    assert np.all(matches.sum(axis=1) == 1)
    true_index = matches.argmax(axis=1)
    true_delta = pose_diff(data["ends"][valid, 0], data["states"][valid, 0, None, :3])
    cal_error = pose_diff(valpred[np.arange(len(valid)), :, true_index], true_delta)
    scale = np.maximum(np.sqrt((cal_error**2).mean(axis=0)), [.002, .002, .045])
    pred0, gates = predict_samples(model, data, ids, 0, device)
    pred1, _ = predict_samples(model, data, ids, 1, device)
    true0 = data["ends"][ids, 0]
    ref_obs = pose_diff(true0[:, 1], data["states"][ids, 0, :3])
    post_ref = posterior(pred0[:, 1], ref_obs, scale[1])
    probe = int(data["probe"])
    seen = pose_diff(true0[:, probe], data["states"][ids, 0, :3])
    post_probe = posterior(pred0[:, probe], seen, scale[probe])
    # Between real attempts only P persists. Marginalise old u and draw a fresh u.
    wphys = post_probe.reshape(len(ids), 9, 3).sum(axis=2)
    next_weights = np.repeat(wphys / 3, 3, axis=1)
    prior = np.full_like(post_ref, 1/len(PARTICLES))
    goal = data["goal"][ids]
    current = data["states"][ids, 1]
    dist = np.linalg.norm(data["ends"][ids, 1, :, :2] - goal[:, None], axis=-1)
    pre_distance = np.linalg.norm(current[:, :2] - goal, axis=-1)
    rows = []
    for mode, w0, w1, bonus in [("prior", prior, prior, .15),
                                ("reflect", post_ref, next_weights, .15),
                                ("reflect_mean", post_ref, next_weights, 0.)]:
        choice = plan(pred1, w1, current, goal, bonus)
        errors = {}
        for label, alt in [("miss", 0), ("offset", 2)]:
            delta = pose_diff(pred0[:, alt], pred0[:, 1])
            truth = pose_diff(true0[:, alt], true0[:, 1])
            mean_xy = np.sum(w0[:, :, None] * delta[..., :2], axis=1)
            mean_yaw = np.arctan2(np.sum(w0*np.sin(delta[..., 2]), axis=1),
                                  np.sum(w0*np.cos(delta[..., 2]), axis=1))
            errors[label+"_mm"] = 1000*np.linalg.norm(mean_xy-truth[:, :2], axis=1)
            errors[label+"_deg"] = np.rad2deg(np.abs(wrap(mean_yaw - truth[:, 2])))
            # Energy score for the joint (x,y,yaw) action effect in fixed units.
            def embed(a):
                return np.stack((a[..., 0]/.06, a[..., 1]/.025,
                                 np.sin(a[..., 2])/.7, np.cos(a[..., 2])/.7), axis=-1)
            v = embed(delta)
            first = np.linalg.norm(v-embed(truth)[:, None], axis=-1)
            between = np.linalg.norm(v[:, :, None]-v[:, None, :], axis=-1)
            errors[label+"_es"] = (w0*first).sum(axis=-1)-.5*(w0[:, :, None]*w0[:, None]*between).sum(axis=(1, 2))
        for i, ix in enumerate(ids):
            a = choice[i]
            rows.append(dict(seed=seed, scene=int(data["scene_id"][ix]),
                             model="graph" if model.use_graph else "dense", mode=mode, choice=int(a),
                             start_mm=pre_distance[i]*1000, end_mm=dist[i, a]*1000,
                             oracle_mm=dist[i].min()*1000, overlap=data["overlap"][ix, 1, a],
                             success=float(data["overlap"][ix, 1, a] > .9),
                             **{k: v[i] for k, v in errors.items()}))
    metrics = {"likelihood_scale_probe": scale[probe].tolist()}
    if gates is not None:
        # Marginal probability by averaging physics particles before the action.
        g = gates.mean(axis=-1) > .5
        y = data["contact"][ids, 0].astype(bool)
        metrics["contact_balanced_accuracy"] = (g[y].mean() + (~g[~y]).mean())/2
    return rows, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pairs.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    data = dict(np.load(args.data))
    if len(data["scene_id"]) != 640:
        raise ValueError("Round 1 uses 400 training, 80 validation and 160 test scenes.")
    ids, val, test = np.arange(400), np.arange(400, 480), np.arange(480, 640)
    all_rows, meta = [], {}
    for seed in args.seeds:
        for graph in (True, False):
            model, validation_loss = train_model(data, ids, val, seed, graph, args.device)
            rows, stats = run_eval(model, data, test, val, seed, args.device)
            all_rows.extend(rows)
            name = ("graph" if graph else "dense") + "_" + str(seed)
            meta[name] = dict(validation_loss=validation_loss, **stats)
            torch.save(model.cpu().state_dict(), out/(name+".pt"))
            print(name, {m: round(np.mean([r["end_mm"] for r in rows if r["mode"] == m]), 3)
                         for m in ("prior", "reflect")}, flush=True)
    with (out/"per_scene.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    (out/"fit.json").write_text(json.dumps(meta, indent=2)+"\n")


if __name__ == "__main__":
    main()
