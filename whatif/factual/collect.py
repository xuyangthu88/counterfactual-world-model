"""Standard prospective cases and executable near-wall contact interventions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from whatif.shared.actions import SITES, corners, menu, rotate
from whatif.prospective.collect import collect

DATA = Path("data/factual")
OUT = Path("results/factual")
START = {"train": 511000, "val": 521000, "test": 531000, "controlval": 541000, "control": 551000}
COUNTS = {
    "causalworld": (384, 160, 192),
    "pusht": (768, 192, 256),
    "wall": (640, 160, 192),
    "wall_crossed": (0, 160, 192),
}


def wall_menu(s, axis):
    # axis points from the block toward the closest boundary in world coordinates.
    n_world = rotate(SITES[:, 2:], s[2])
    align = n_world @ axis
    arm = SITES[:, 0] * SITES[:, 3] - SITES[:, 1] * SITES[:, 2]
    face = np.flatnonzero(align > align.max() - 0.02)
    toward = face[np.argmin(abs(arm[face]))]
    torque = face[np.argmax(abs(arm[face]))]
    tangent = np.argmin(abs(align) + 0.1 * abs(arm))
    away = np.argmin(align + 0.03 * abs(arm))
    local_agent = rotate(s[3:5] - s[:2], -s[2]) - [0, 0.45]
    rows = []
    for site, stroke, gap in [
        (toward, 0, 0.42),
        (tangent, 0, 0.45),
        (toward, 0.07, 0),
        (toward, 0.33, 0),
        (toward, 0.74, 0),
        (torque, 0.55, 0),
        (tangent, 0.45, 0),
        (away, 0.34, 0),
    ]:
        r, n = np.split(SITES[site], 2)
        r = r - gap * n
        offset = r - 0.17 * n - local_agent
        rows.append([*r, *n, stroke, r[0] * n[1] - r[1] * n[0], np.linalg.norm(offset)])
    return np.asarray(rows, dtype=np.float32)


def wall_case(split, seed, crossed=False):
    from whatif.shared.sim import Scene

    rng = np.random.default_rng(seed)
    protocol = {"train": "train", "val": "transfer_val", "test": "test"}[split]
    orientation = seed // 2 if crossed else seed
    axis = np.zeros(2)
    dim = orientation % 2
    axis[dim] = 1 if orientation % 4 < 2 else -1
    angle = rng.uniform(-np.pi, np.pi)
    outline = rotate(corners, angle)
    gap = rng.uniform(4.0, 27.0)
    start = np.full(2, 256.0)
    start[1 - dim] += rng.uniform(-74, 74)
    start[dim] = (
        (504 - gap - outline[:, dim].max()) if axis[dim] > 0 else (8 + gap - outline[:, dim].min())
    )
    agent = start + rotate(np.array([0.0, 45.0]), angle) - axis * rng.uniform(104, 146)

    def reset():
        env = Scene("pusht", seed, protocol)
        e = env.env.env
        e.block.angle = angle
        e.block.velocity = (0, 0)
        e.block.angular_velocity = 0
        e.agent.velocity = (0, 0)
        e._set_state(np.r_[agent, start, angle])
        return env

    b = int(rng.choice([0, 2, 3, 6]))
    ex = rng.choice([-0.04, 0.0, 0.04], 2)
    env = reset()
    prev = env.state()
    prev_act = wall_menu(prev, axis)
    s = env.push(0, noise=ex[0], command=prev_act[b])[0]
    env.close()
    actions = wall_menu(s, axis)
    ends = []
    edge = []
    peak = []
    nullpose = []
    sources = []
    for j, a in enumerate(actions):
        source = int(rng.choice(3))
        sources.append(source)
        for deleted in [-1, source] if split == "train" else [-1]:
            env = reset()
            ss = env.push(0, noise=ex[0], command=prev_act[b])[0]
            if j == 0:
                np.testing.assert_allclose(ss, s, atol=1e-5)
            z, g, _, p = env.push(0, noise=ex[1], null=deleted, command=a)
            phys = env.env.physics
            group = env.env.protocol
            env.close()
            if deleted < 0:
                ends.append(z)
                edge.append(g)
                peak.append(p)
            else:
                nullpose.append(z[:3])
    return dict(
        previous=prev,
        fact=prev_act[b],
        state=s,
        cmd=actions,
        end=ends,
        edge=edge,
        peak=peak,
        physics=phys,
        protocol=group,
        seed=seed,
        ref=int(rng.integers(8)),
        source=sources,
        nullpose=nullpose,
    )


def job(args):
    task, split, seed = args
    if task in ("wall", "wall_crossed"):
        return wall_case(split, seed, crossed=task == "wall_crossed")
    sp = {"train": "fit", "val": "test" if task == "causalworld" else "val", "test": "test"}[split]
    d = collect((task, sp, seed))
    d["fact"] = d["previous_actions"][d["b"]]
    return {
        k: d[k]
        for k in (
            "previous",
            "fact",
            "state",
            "cmd",
            "end",
            "edge",
            "peak",
            "physics",
            "protocol",
            "seed",
            "ref",
        )
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=COUNTS)
    ap.add_argument("split", choices=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    n = COUNTS[a.task][["train", "val", "test"].index(a.split)]
    if not n:
        ap.error("The crossed diagnostic uses the original near-wall training data.")
    start = START[a.split] + {"wall": 4000, "wall_crossed": 6000}.get(a.task, 0)
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(job, [(a.task, a.split, i) for i in range(start, start + n)]))
    np.savez_compressed(
        DATA / f"{a.task}-{a.split}.npz", **{k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    )
    print(a.task, a.split, n, np.asarray([r["edge"] for r in rows]).mean((0, 1)), flush=True)
