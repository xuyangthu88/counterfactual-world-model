import argparse
import csv
from contextlib import ExitStack
import json
import numpy as np
import torch
from whatif.prospective.planner import cem, enumerate_cw
from whatif.shared.actions import menu
from whatif.shared.control import NativeBatch
from whatif.shared.model import grid, posterior
from whatif.shared.planning import Models, policy_scores
from .collect import OUT, START


@torch.no_grad()
def run(args):
    torch.set_num_threads(2)
    m = Models(args.device)
    name = args.method
    task = args.task
    lock = OUT / "freeze.json"
    cost = (
        json.loads(lock.read_text())["contact_cost"]
        if name in ("ours", "uniform")
        else (0.25 if name == "cost" else 0.0)
    )
    if name in ("ours", "uniform", "cost", "direct"):
        m.graph.load_state_dict(
            torch.load(OUT / "graph.pt", map_location=args.device, weights_only=True)
        )
    seeds = list(range(START[args.split], START[args.split] + args.count))
    # P1/P2/P7/P8 are represented on fresh validation and test layouts.
    protocol = "transfer_val" if task == "pusht" and args.split == "controlval" else "control"
    cap, goal = (420, 0.90) if task == "causalworld" else (280, 0.95)

    def tensor(x):
        return torch.as_tensor(np.asarray(x), device=args.device, dtype=torch.float32)

    args.tag = f"switch-{name}-{task}"
    with ExitStack() as stack:
        sim = NativeBatch(args, stack)
        sim.start = seeds[0]
        start = sim.reset(task, seeds, protocol)
        s = tensor([start[j][0] for j in seeds])
        prev = s.clone()
        prev_cmd = tensor([menu(z)[1] for z in s.cpu().numpy()])
        mass = s.new_full((len(s), 5), 0.2)
        used = np.zeros(len(s), int)
        hit = s[:, 12].cpu().numpy() > goal
        step = 0
        trace = []
        while np.any(~hit & (used < cap)):
            ids = np.flatnonzero(~hit & (used < cap))
            ss = s[ids]
            cmd = tensor([menu(x) for x in ss.cpu().numpy()])
            if step == 0:
                sel = np.ones(len(ids), int)
            elif name in ("physics", "cophy"):
                scores = policy_scores(
                    m,
                    ss,
                    cmd,
                    mass[ids],
                    task,
                    0.25 if name == "cophy" else 0.0,
                    False,
                    name,
                    prev[ids],
                    prev_cmd[ids],
                )
                sel = scores.argmax(-1).cpu().numpy()
            elif task == "causalworld" and name != "pets":
                mm = mass[ids]
                zero = torch.zeros_like(mm)
                sel = (
                    enumerate_cw(m, ss, cmd, (mm, zero, zero), {"gain": 0.0}, contact_cost=cost)
                    .cpu()
                    .numpy()
                )
            else:
                sel = []
                for j, idx in enumerate(ids):
                    rng = 11 + seeds[idx] * 73 + step
                    if name == "pets":
                        action = cem(m, ss[j], task, rng, terminal=task == "causalworld")
                    else:
                        action = cem(
                            m,
                            ss[j],
                            task,
                            rng,
                            mass[idx],
                            torch.zeros_like(mass[idx]),
                            contact_cost=cost,
                        )
                    sel.append(action)
            actual = sim.step([(seeds[i], a, min(42, cap - used[i])) for i, a in zip(ids, sel)])
            nxt = tensor([actual[seeds[i]][0] for i in ids])
            ii = torch.arange(len(ids), device=s.device)
            chosen = cmd[ii, torch.as_tensor(sel, device=s.device)]
            if name in ("ours", "cost", "direct", "old"):
                z = grid(m.graph, ss, chosen[:, None])[0]
                w = posterior(ss, nxt, z, torch.zeros_like(ii), 0.055, mass[ids])
                mass[ids] = w.reshape(-1, 5, 3).sum(-1)
            for j, i in enumerate(ids):
                used[i] += actual[seeds[i]][1]
                hit[i] = float(nxt[j, 12]) > goal
                trace.append(
                    [task, name, seeds[i], step + 1, int(sel[j]), used[i], float(nxt[j, 12])]
                )
            prev[ids] = ss
            prev_cmd[ids] = chosen
            s[ids] = nxt
            step += 1
        tag = f"{args.split}-{task}-{name}"
        with (OUT / f"{tag}.csv").open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["task", "policy", "seed", "protocol", "success", "coverage", "steps"])
            writer.writerows(
                [
                    [task, name, x, start[x][1], int(hit[i]), float(s[i, 12]), int(used[i])]
                    for i, x in enumerate(seeds)
                ]
            )
        with (OUT / f"{tag}-steps.csv").open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["task", "policy", "seed", "decision", "action", "steps", "coverage"])
            writer.writerows(trace)
        print(task, name, int(hit.sum()), s[:, 12].mean().item(), used.mean(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["controlval", "control"])
    ap.add_argument("task", choices=["pusht", "causalworld"])
    ap.add_argument(
        "method", choices=["ours", "uniform", "direct", "cost", "old", "pets", "cophy", "physics"]
    )
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sim-python", default="/tmp/whatif-push-venv/bin/python")
    run(ap.parse_args())
