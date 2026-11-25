"""Matched five-stroke sequences for checking persistence of physics inference."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from .collect import DATA, OUT


def sample(job):
    from whatif.shared.actions import menu
    from whatif.shared.sim import Scene
    from whatif.shared.returns import physics

    task, seed = job
    scene = Scene(task, seed, "control")
    physical = scene.env.mass
    group = scene.env.protocol
    states = [scene.state()]
    actions = []
    observations = []
    for step in range(5):
        cmd = menu(states[-1])
        a = 1 if step == 0 else int(np.argmax(physics(states[-1], cmd)))
        nxt, _, _, _ = scene.push(a)
        actions.append(cmd[a])
        states.append(nxt)
        observations.append(len(scene.latest_free))
    scene.close()
    return dict(
        seed=seed, state=states, cmd=actions, free=observations, physical=physical, protocol=group
    )


def collect(workers):
    for task in ("causalworld", "pusht"):
        with ProcessPoolExecutor(workers) as p:
            rows = list(p.map(sample, [(task, i) for i in range(561000, 561096)]))
        np.savez_compressed(
            DATA / f"{task}-history.npz", **{k: [r[k] for r in rows] for k in rows[0]}
        )


def analyze(device):
    import torch
    from whatif.shared.model import grid, posterior
    from .model import FactualGraph, evidence

    torch.set_num_threads(2)
    model = FactualGraph().to(device)
    model.load_state_dict(torch.load(OUT / "graph.pt", weights_only=True, map_location=device))
    model.eval()
    result = {}
    raw = {}
    with torch.no_grad():
        for t in ["pusht", "causalworld"]:
            with np.load(DATA / f"{t}-history.npz") as f:
                d = dict(f)
            z = torch.as_tensor(d["state"], device=device)
            a = torch.as_tensor(d["cmd"], device=device)
            b = len(z)
            old = z.new_full((b, 5), 0.2)
            log = {"legacy": [], "ours": []}
            for j in range(5):
                st, obs = z[:, j], z[:, j + 1]
                cmd = a[:, j]
                mass = model.belief(evidence(st, cmd, obs))
                p = grid(model, st, cmd[:, None])[0]
                old = (
                    posterior(
                        st, obs, p, torch.zeros(b, dtype=torch.long, device=device), 0.055, old
                    )
                    .reshape(b, 5, 3)
                    .sum(-1)
                )
                log["legacy"].append(old.cpu().numpy())
                log["ours"].append(mass.cpu().numpy())
            res = {}
            for name, values in log.items():
                w = np.stack(values, 1)
                tv = np.abs(np.diff(w, axis=1)).sum(2) * 0.5
                item = dict(
                    mean_repeat_tv=float(tv.mean()), neff=(1 / (w * w).sum(2)).mean(0).tolist()
                )
                if t == "pusht":
                    anchors = np.array(
                        [0, 0.5 * 0.08**0.1, 0.08**0.1, (0.08**0.1 + 0.55**0.1) / 2, 0.55**0.1]
                    )
                    rho = w @ anchors
                    true = d["physical"] ** 0.1
                    item["rho_mae_by_damping"] = {
                        str(g): np.abs(rho[d["protocol"] == g] - true[d["protocol"] == g, None])
                        .mean(0)
                        .tolist()
                        for g in [0, 1]
                    }
                    stale = (d["free"][:, 1:] == 0) & (d["state"][:, 2:, 18] > 0)
                    item["reused_rho_steps"] = int(stale.sum())
                    item["tv_no_new_rho"] = float(tv[stale].mean()) if stale.any() else None
                res[name] = item
                raw[t + "_" + name] = w
            result[t] = res
    (OUT / "history.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "history-weights.npz", **raw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "analyze"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.mode == "collect":
        collect(args.workers)
    else:
        analyze(args.device)
