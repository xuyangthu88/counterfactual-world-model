"""Generate paired trajectories on two successive real CausalWorld snapshots."""
import argparse
import concurrent.futures
import time
from pathlib import Path

import numpy as np

from .simulator import PushingScene, Push, angle

MASS = (0.025, 0.075, 0.14)
FRICTION = (0.18, 0.5, 0.85)
DISTURBANCE = (-0.0025, 0.0, 0.0025)
# First three form the miss/centre/offset triplet. Remaining actions are planner alternatives.
ACTIONS = [(0.083, 0.054), (0.0, 0.054), (0.022, 0.054),
           (-0.022, 0.054), (0.0, 0.032), (0.0, 0.041), (0.0, 0.069),
           (-0.022, 0.039), (0.022, 0.039), (-0.022, 0.069), (0.022, 0.069)]
PROBE = 5


def observation(scene):
    return np.r_[scene.state(), scene.robot.get_latest_full_state()["end_effector_positions"][3:6]]


def single_scene(seed):
    rng = np.random.default_rng(seed)
    xy = rng.uniform([0.042, -0.040], [0.065, -0.012])
    yaw = rng.uniform(-0.17, 0.17)
    mass, mu = rng.choice(MASS), rng.choice(FRICTION)
    goal = xy + rng.uniform([-0.118, -0.02], [-0.066, 0.02])
    noise = rng.choice(DISTURBANCE, size=2)
    scene = PushingScene(xy, yaw, mass, mu, goal, seed)
    states, outcomes, contacts, overlaps = [], [], [], []
    def execute(idx, step):
        offset, travel = ACTIONS[idx]
        return scene.push(Push(offset, travel + noise[step]))

    try:
        for step in range(2):
            states.append(observation(scene))
            block = []
            with scene.snapshot() as restore:
                for idx in range(len(ACTIONS)):
                    restore()
                    block.append(execute(idx, step))
            outcomes.append([r["end"] for r in block])
            contacts.append([r["contact"] for r in block])
            overlaps.append([r["overlap"] for r in block])
            if step == 0:
                factual = execute(PROBE, step)
                if not np.allclose(factual["end"], outcomes[0][PROBE], atol=1e-7, rtol=0):
                    raise RuntimeError(f"factual replay mismatch in scene {seed}")
        return (seed, np.asarray(states), np.asarray(outcomes), np.asarray(contacts),
                np.asarray(overlaps), [mass, mu], noise, goal)
    finally:
        scene.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=10000)
    parser.add_argument("--scenes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(single_scene, range(args.start, args.start + args.scenes)), 1):
            rows.append(row)
            if i % 40 == 0:
                print(f"{i}/{args.scenes} scenes, {time.monotonic()-started:.1f}s", flush=True)
    keys = ["scene_id", "states", "ends", "contact", "overlap", "physics", "noise", "goal"]
    arrays = dict(zip(keys, map(np.asarray, zip(*rows))))
    np.savez_compressed(args.output, **arrays, actions=ACTIONS, probe=PROBE)
    print(f"saved {args.output}, {args.scenes} scenes, {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
