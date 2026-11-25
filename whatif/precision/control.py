"""Real native control with held-out layouts. No simulator future enters action selection."""

import argparse
import csv
import json
import os
from contextlib import ExitStack
import numpy as np
import torch
from whatif.shared.actions import menu
from whatif.shared.control import NativeBatch
from whatif.shared.model import grid, task_value, unpack
from whatif.shared.planning import mechanical, policy_scores, scale
from whatif.prospective.planner import cem
from whatif.factual.planning import two_step
from whatif.factual.model import AdaptedModels, evidence
from .collect import OUT, START, OFFSET
from .geometry import overlap
from .model import install


def geometry_readout(s, z):
    out = z.clone()
    out[..., 12] = overlap(z[..., :3], s[..., 9:12])
    return out, torch.maximum(s[..., 12], out[..., 12])


def root_score(m, name, s, a, prev, last, w, endpoint_only=False):
    st = s[:, None]
    ctx = evidence(prev, last, s)
    b, n = a.shape[:2]
    if name == "graph":
        z = grid(m.graph, s, a)[0]
    elif name == "pets":
        mu, sd = m.predict(
            st.expand(-1, n, -1).reshape(-1, 19),
            a.reshape(-1, 7),
            ctx[:, None].expand(-1, n, -1).reshape(-1, 52),
        )
        gen = torch.Generator(device=s.device).manual_seed(11)
        eps = (
            torch.randn(5, 1, 1, 1, 16, generator=gen, device=s.device)
            * s.new_tensor([-1.224745, 0.0, 1.224745])[None, None, None, :, None]
        )
        dy = (
            (mu.reshape(5, b, n, 1, 16) + sd.reshape(5, b, n, 1, 16) * eps)
            .permute(1, 2, 0, 3, 4)
            .reshape(b, n, 15, 16)
        )
        z = unpack(st[:, :, None].expand(-1, n, 15, -1), dy)[0]
        w = torch.full_like(w, 0.2)
    else:
        p = m.cophy(prev, s, last, s, a)[0]
        z = st[:, :, None].expand(-1, n, 1, -1).clone()
        z[..., :3] = p[:, :, None]
    z, peak = geometry_readout(st[:, :, None], z)
    v = task_value(st[:, :, None], z, z[..., 12] if endpoint_only else peak)
    if name == "cophy":
        v = v[..., 0]
    else:
        if name == "graph":
            v = v + 0.12 * torch.tanh((v - v[:, 1:2]) / 0.30)
        v = (v * w.repeat_interleave(3, -1)[:, None] / 3).sum(-1)
    return (0.70 * scale(v) + 0.30 * scale(mechanical(st, a))).argmax(-1)


@torch.no_grad()
def run(args, *, output=OUT, start_seeds=START, models=None, selector=None, transform_context=None, tag_prefix="", output_variant=None):
    torch.set_num_threads(2)
    m = AdaptedModels(args.device) if models is None else models
    orig = args.method
    name = "graph" if orig in ("uniform", "static") else orig
    if models is None and args.variant != "stage11" and name != "physics":
        install(m, args.device, (name,))
    cw = args.task in ("p9", "causalworld")
    task = "causalworld" if cw else "pusht"
    mode = args.variant
    if mode == "selected" and selector is None:
        mode = json.loads((OUT / "freeze.json").read_text())[task][name]
    simsplit = (
        "p9"
        if args.task == "p9"
        else ("transfer_val" if not cw and args.split == "controlval" else "control")
    )
    goal = 0.90 if cw else 0.95
    cap = 420 if cw else 280
    begin = start_seeds[args.split] + OFFSET[args.task] + args.offset
    seeds = list(range(begin, begin + args.count))
    tag = f"{tag_prefix}{args.split}-{args.task}-{orig}-{output_variant or args.variant}"
    if args.offset:
        tag += f"-{args.offset}"
    args.tag = "precision-" + tag
    tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=args.device)
    traces = []
    with ExitStack() as stack:
        sim = NativeBatch(args, stack)
        sim.start = begin
        start = sim.reset(task, seeds, simsplit)
        s = tensor([start[i][0] for i in seeds])
        prev = s.clone()
        last = tensor([menu(z)[1] for z in s.cpu().numpy()])
        hit = s[:, 12].cpu().numpy() > goal
        used = np.zeros(len(s), int)
        step = 0
        while np.any(~hit & (used < cap)):
            ids = np.flatnonzero(~hit & (used < cap))
            ss = s[ids]
            a = tensor([menu(z) for z in ss.cpu().numpy()])
            ctx = evidence(prev[ids], last[ids], ss)
            if transform_context is not None:
                ctx=transform_context(ctx, task)
            if orig == "uniform":
                mass = ss.new_full((len(ss), 5), 0.2)
            elif orig == "static":
                avg = json.loads((OUT / "posterior-mean.json").read_text())[task]
                mass = ss.new_tensor(avg).expand(len(ss), -1)
            else:
                mass = m.graph.belief(ctx)
            selected = np.ones(len(ss), int)
            if step and selector is not None:
                selected = selector(m, ss, a, ctx, task, mass, step, np.asarray(seeds)[ids], cap-used[ids], prev[ids], last[ids]).cpu().numpy()
            elif step and name in ("physics", "cophy"):
                if name == "cophy" and not cw and mode not in ("raw", "stage11"):
                    selected = (
                        root_score(m, name, ss, a, prev[ids], last[ids], mass, mode == "endonly")
                        .cpu()
                        .numpy()
                    )
                else:
                    beta = 0 if name == "physics" else (0.25 if cw else 0.7)
                    selected = (
                        policy_scores(m, ss, a, mass, task, beta, False, name, prev[ids], last[ids])
                        .argmax(-1)
                        .cpu()
                        .numpy()
                    )
            elif step and cw and name == "graph":
                selected = (
                    torch.cat(
                        [
                            two_step(m, ss[j : j + 3], a[j : j + 3], mass[j : j + 3], task)
                            for j in range(0, len(ss), 3)
                        ]
                    )
                    .cpu()
                    .numpy()
                )
            elif step:
                local = (
                    (ss[:, 12] > 0.70).cpu().numpy()
                    if mode == "local" and not cw
                    else np.zeros(len(ss), bool)
                )
                if local.any():
                    selected[local] = (
                        root_score(
                            m,
                            name,
                            ss[local],
                            a[local],
                            prev[ids[local]],
                            last[ids[local]],
                            mass[local],
                        )
                        .cpu()
                        .numpy()
                    )
                for j in np.flatnonzero(~local):
                    m.context = ctx[j]
                    graph = name == "graph"
                    selected[j] = cem(
                        m,
                        ss[j],
                        task,
                        11 + seeds[ids[j]] * 73 + step,
                        mass[j] if graph else None,
                        torch.zeros_like(mass[j]) if graph else None,
                        terminal=(not graph and not cw),
                        readout=(
                            geometry_readout
                            if not cw and mode in ("exact", "local", "endonly")
                            else None
                        ),
                        retain_heads=mode == "endonly",
                    )
            real = sim.step([(seeds[i], j, min(42, cap - used[i])) for i, j in zip(ids, selected)])
            nxt = tensor([real[seeds[i]][0] for i in ids])
            for j, i in enumerate(ids):
                used[i] += real[seeds[i]][1]
                hit[i] = float(nxt[j, 12]) > goal
                traces.append(
                    [
                        seeds[i],
                        step + 1,
                        int(selected[j]),
                        int(used[i]),
                        float(nxt[j, 12]),
                        float(ss[j, 12]),
                        float(mass[j].square().sum().reciprocal()),
                    ]
                )
            prev[ids] = ss
            last[ids] = a[
                torch.arange(len(ss), device=s.device), torch.as_tensor(selected, device=s.device)
            ]
            s[ids] = nxt
            step += 1
    rows = [
        [i, start[i][1], int(h), float(z), int(k)] for i, h, z, k in zip(seeds, hit, s[:, 12], used)
    ]
    for suffix, header, table in [
        ("", ("seed", "protocol", "success", "coverage", "steps"), rows),
        (
            "-steps",
            ("seed", "decision", "action", "steps", "coverage", "start_coverage", "effective_k"),
            traces,
        ),
    ]:
        with (output / f"{tag}{suffix}.csv").open("w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(table)
    print(
        tag,
        int(hit.sum()),
        len(hit),
        round(float(s[:, 12].mean()), 5),
        round(used.mean(), 2),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("split", choices=["controlval", "control"])
    p.add_argument("task", choices=["pusht", "causalworld", "p9"])
    p.add_argument("method", choices=["graph", "pets", "cophy", "physics", "uniform", "static"])
    p.add_argument("variant", choices=["raw", "exact", "local", "endonly", "stage11", "selected"])
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument(
        "--sim-python", default=os.environ.get("SIM_PY", "/tmp/whatif-push-venv/bin/python")
    )
    run(p.parse_args())
