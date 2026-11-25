import argparse
from contextlib import ExitStack
import csv
import json
import numpy as np
import torch
from whatif.shared.actions import menu
from whatif.shared.control import NativeBatch
from whatif.shared.planning import policy_scores
from whatif.prospective.planner import cem, enumerate_cw
from whatif.shared.model import grid, posterior
from .collect import OUT, START
from .model import AdaptedModels, evidence
from .planning import two_step


@torch.no_grad()
def run(args):
    torch.set_num_threads(2)
    m = AdaptedModels(args.device)
    name = args.method
    task = args.task
    cfg = json.loads((OUT / "freeze.json").read_text()) if args.split == "control" else {}
    plan = cfg.get(task, {}).get("ours", "cem")
    if name in ("cem", "enumerate", "root"):
        plan = name
    cw = task == "causalworld"
    cap, goal = (420, 0.9) if cw else (280, 0.95)
    protocol = "transfer_val" if not cw and args.split == "controlval" else "control"
    seeds = list(range(START[args.split], START[args.split] + args.count))
    tensor = lambda x: torch.as_tensor(np.asarray(x), device=args.device, dtype=torch.float32)
    args.tag = f"fact-{name}-{task}"
    with ExitStack() as stack:
        sim = NativeBatch(args, stack)
        sim.start = seeds[0]
        start = sim.reset(task, seeds, protocol, fixed_coast=0.45 if name == "fixed" else None)
        s = tensor([start[i][0] for i in seeds])
        prev = s.clone()
        last = tensor([menu(z)[1] for z in s.cpu().numpy()])
        mass = s.new_full((len(s), 5), 0.2)
        if name == "static":
            avg = json.loads((OUT / "calibration.json").read_text())[task]["mean_posterior"]
            mass[:] = s.new_tensor(avg)
        hit = s[:, 12].cpu().numpy() > goal
        used = np.zeros(len(s), int)
        step = 0
        traces = []
        while np.any(~hit & (used < cap)):
            ids = np.flatnonzero(~hit & (used < cap))
            ss = s[ids].clone()
            if name in ("masked", "fixed"):
                ss[:, 17] = 0.45
                ss[:, 18] = 0.0
            cmd = tensor([menu(x) for x in ss.cpu().numpy()])
            if step == 0:
                sel = np.ones(len(ids), int)
            elif name in ("physics", "cophy", "cophy_high"):
                beta = (
                    cfg.get(task, {}).get("cophy", 0.25)
                    if name == "cophy"
                    else (0.7 if name == "cophy_high" else 0.0)
                )
                value = policy_scores(
                    m,
                    ss,
                    cmd,
                    mass[ids],
                    task,
                    beta,
                    False,
                    "physics" if name == "physics" else "cophy",
                    prev[ids],
                    last[ids],
                )
                sel = value.argmax(-1).cpu().numpy()
            elif name in ("pets", "pets_plain"):
                choices = []
                ctx = evidence(prev[ids], last[ids], ss)
                terminal = cfg.get(task, {}).get("terminal", cw) if name == "pets" else not cw
                for j, i in enumerate(ids):
                    m.context = ctx[j]
                    choices.append(
                        cem(m, ss[j], task, 11 + seeds[i] * 73 + step, terminal=terminal)
                    )
                sel = np.array(choices)
            else:
                if name in ("ours", "cem", "enumerate", "root"):
                    mass[ids] = m.graph.belief(evidence(prev[ids], last[ids], ss))
                mm = mass[ids]
                if plan in ("enumerate", "root"):
                    sel = (
                        torch.cat(
                            [
                                two_step(
                                    m,
                                    ss[j : j + 4],
                                    cmd[j : j + 4],
                                    mm[j : j + 4],
                                    task,
                                    root_only=plan == "root",
                                )
                                for j in range(0, len(ss), 4)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                elif cw:
                    zero = torch.zeros_like(mm)
                    sel = enumerate_cw(m, ss, cmd, (mm, zero, zero), {"gain": 0.0}).cpu().numpy()
                else:
                    sel = np.array(
                        [
                            cem(
                                m,
                                ss[j],
                                task,
                                11 + seeds[i] * 73 + step,
                                mm[j],
                                torch.zeros_like(mm[j]),
                            )
                            for j, i in enumerate(ids)
                        ]
                    )
            real = sim.step([(seeds[i], j, min(42, cap - used[i])) for i, j in zip(ids, sel)])
            new = tensor([real[seeds[i]][0] for i in ids])
            ii = torch.arange(len(ids), device=s.device)
            executed = cmd[ii, torch.as_tensor(sel, device=s.device)]
            if name == "legacy":
                zz = grid(m.graph, ss, executed[:, None])[0]
                mass[ids] = (
                    posterior(ss, new, zz, torch.zeros_like(ii), 0.055, mass[ids])
                    .reshape(-1, 5, 3)
                    .sum(-1)
                )
            for j, i in enumerate(ids):
                used[i] += real[seeds[i]][1]
                hit[i] = float(new[j, 12]) > goal
                traces.append(
                    [
                        task,
                        name,
                        seeds[i],
                        step + 1,
                        int(sel[j]),
                        int(used[i]),
                        float(new[j, 12]),
                        float(new[j, 17]),
                        float(mass[i].square().sum().reciprocal()),
                    ]
                )
            prev[ids] = ss
            last[ids] = executed
            s[ids] = new
            step += 1
        tag = f"{args.split}-{task}-{name}"
        for suffix, header, rows in [
            (
                "",
                ["task", "policy", "seed", "protocol", "success", "coverage", "steps"],
                [
                    [task, name, i, start[i][1], int(h), float(z), int(k)]
                    for i, h, z, k in zip(seeds, hit, s[:, 12], used)
                ],
            ),
            (
                "-steps",
                [
                    "task",
                    "policy",
                    "seed",
                    "decision",
                    "action",
                    "steps",
                    "coverage",
                    "rho",
                    "effective_k",
                ],
                traces,
            ),
        ]:
            with (OUT / f"{tag}{suffix}.csv").open("w", newline="") as f:
                wr = csv.writer(f, lineterminator="\n")
                wr.writerow(header)
                wr.writerows(rows)
        print(task, name, int(hit.sum()), float(s[:, 12].mean()), used.mean(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["controlval", "control"])
    ap.add_argument("task")
    ap.add_argument("method")
    ap.add_argument("--count", required=True, type=int)
    ap.add_argument("--sim-python", default="/tmp/whatif-push-venv/bin/python")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=6)
    run(ap.parse_args())
