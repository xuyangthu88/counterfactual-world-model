"""Evaluate frozen models with goal-overlap planning on a fresh set of scenes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import PARTICLES, plan, posterior, predict_samples, signed_js
from .model import World, pose_diff


def planar_overlap(pose, goal):
    # CausalWorld uses world-aligned bounding boxes; approximate the z overlap as one.
    radius = .0325
    extent = radius * (np.abs(np.cos(pose[..., 2])) + np.abs(np.sin(pose[..., 2])))
    relative = pose[..., :2] - goal[:, None, None, :]
    upper = np.minimum(relative + extent[..., None], radius)
    lower = np.maximum(relative - extent[..., None], -radius)
    return (np.maximum(upper-lower, 0)/(2*radius)).prod(axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fresh.npz")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    data = dict(np.load(args.data))
    ids = np.arange(len(data["scene_id"]))
    assert (data["scene_id"] >= 20000).all()
    torch.set_num_threads(2)
    fit = json.loads((args.out/"fit.json").read_text())
    results = []
    for name, details in fit.items():
        model_name, seed = name.split("_")
        model = World(model_name=="graph")
        model.load_state_dict(torch.load(args.out/(name+".pt"), map_location="cpu", weights_only=True))
        model.eval()
        probe = int(data["probe"])
        first = predict_samples(model, data, ids, 0, "cpu")[0][:, probe]
        factual = pose_diff(data["ends"][:, 0, probe], data["states"][:, 0, :3])
        post = posterior(first, factual, np.array(details["likelihood_scale_probe"]))
        post = np.repeat(post.reshape(len(ids), 9, 3).sum(axis=-1)/3, 3, axis=-1)
        pred = predict_samples(model, data, ids, 1, "cpu")[0]
        current = data["states"][:, 1, :3]
        pose = current[:, None, None, :] + pred
        overlap = planar_overlap(pose, data["goal"])
        dist = np.linalg.norm(pose[..., :2] - data["goal"][:, None, None], axis=-1)
        actual_dist = np.linalg.norm(data["ends"][:, 1, :, :2] - data["goal"][:, None], axis=-1)
        actual_overlap = data["overlap"][:, 1]
        for mode,w in [("prior", np.full_like(post, 1/len(PARTICLES))), ("reflect", post)]:
            weights = w[:, None, :]
            chance = ((overlap > .9)*weights).sum(axis=-1)
            expected_overlap = (weights*overlap).sum(axis=-1)
            expected_dist = (weights*dist).sum(axis=-1)
            score = expected_overlap - .05*expected_dist/.065
            for policy, action in [
                ("distance", plan(pred, w, current, data["goal"])),
                ("overlap_mean", score.argmax(axis=1)),
                ("overlap_js", (score + .15*signed_js(chance, chance[:, 1:2])).argmax(axis=1))]:
                chosen = actual_overlap[ids, action]
                for i in ids:
                    results.append(dict(model=model_name, seed=int(seed), scene=int(data["scene_id"][i]),
                                      mode=mode, policy=policy, choice=int(action[i]),
                                      end_mm=actual_dist[i, action[i]]*1000,
                                      overlap=chosen[i], success=int(chosen[i]>.9),
                                      oracle_success=int(actual_overlap[i].max()>.9)))
    with (args.out/"fresh_per_scene.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    aggregate(results, args.out)


def aggregate(rows, out):
    groups = {}
    for r in rows:
        groups.setdefault("/".join(r[k] for k in ("model","mode","policy")), []).append(r)
    metrics = ("end_mm","overlap","success","oracle_success")
    means, matrices = {}, {}
    for name, group in groups.items():
        group.sort(key=lambda r:(r["seed"],r["scene"]))
        size = len(set(r["scene"] for r in group))
        mat = np.array([[r[k] for k in metrics] for r in group]).reshape(-1,size,len(metrics))
        matrices[name] = mat.mean(axis=0)
        means[name] = {k:{"mean":float(mat[:,:,i].mean()), "seeds":mat[:,:,i].mean(axis=1).tolist()}
                       for i,k in enumerate(metrics)}
    rng = np.random.default_rng(20261124)
    sample = rng.integers(size, size=(5000,size))
    contrast = {}
    key = "graph/reflect/overlap_js"
    for ref in ["graph/prior/overlap_js", "graph/reflect/distance", "graph/reflect/overlap_mean",
                "dense/reflect/overlap_js"]:
        delta = matrices[key]-matrices[ref]
        bands = np.quantile(delta[sample].mean(axis=1),[.025,.975],axis=0)
        contrast[key+" minus "+ref] = {k:[float(delta[:,i].mean()), *bands[:,i].tolist()]
                                        for i,k in enumerate(metrics)}
    (out/"fresh_summary.json").write_text(json.dumps({"methods":means,"contrasts":contrast}, indent=2)+"\n")


if __name__ == "__main__":
    main()
