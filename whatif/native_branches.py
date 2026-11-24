"""CausalWorld interpreter: replay an executed prefix and branch the next stroke."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pybullet as pb
from causal_world.benchmark.benchmarks import PUSHING_BENCHMARK
from causal_world.task_generators import generate_task

from .collect import ACTIONS
from .simulator import PushingScene, Push

ANGLE = -np.pi / 2


def state(scene):
    pose = scene.state()
    v, w = pb.getBaseVelocity(scene.body, physicsClientId=scene.client)
    return np.r_[pose, v[:2], w[2]]


def record(scene):
    initial = state(scene)
    ends, scores = [], []
    with scene.snapshot() as restore:
        for offset, travel in ACTIONS:
            restore()
            result = scene.push(Push(offset, travel))
            ends.append([initial, state(scene)])
            scores.append(result['overlap'])
    return np.array(ends)[:, :, None, :], scores


def aim(scene, goal):
    pos = scene.state()[:2]
    direction = np.asarray(goal) - pos
    length = np.linalg.norm(direction)
    direction = direction / length if length > 1e-8 else np.array([0., 1.])
    scene.push_angle = np.arctan2(-direction[1], -direction[0])
    approach = pos - .065*direction
    scene.active = int(np.linalg.norm(scene.rest[:, :2]-approach, axis=1).argmin())


def sample(job):
    protocol_name, seed, history, *mode = job
    follow = mode == ['reach']
    task = generate_task(task_generator_id='pushing').get_task_params()
    xy = task['tool_block_position'][:2]
    goal = task['goal_block_position'][:2]
    scene = PushingScene(xy, 0., task['tool_block_mass'], None, goal, seed)
    try:
        protocol = next(p for p in PUSHING_BENCHMARK['evaluation_protocols'] if p.get_name()==protocol_name)
        protocol.init_protocol(scene.env, None)
        np.random.seed(seed)
        intervention = protocol.get_intervention(seed, 0)
        ok, _ = scene.env.set_starting_state(intervention)
        if not ok:
            raise RuntimeError(f'invalid official intervention {protocol_name} {seed}')
        scene.rest = scene.robot.get_latest_full_state()['end_effector_positions'].reshape(3, 3).copy()
        approach = xy - np.array([0., .065])
        scene.active = int(np.linalg.norm(scene.rest[:, :2] - approach, axis=1).argmin())
        scene.push_angle = ANGLE
        for action in history[:-1]:
            if follow:
                aim(scene, goal)
            scene.push(Push(*ACTIONS[action]))
        def tip():
            return scene.robot.get_latest_full_state()['end_effector_positions'].reshape(3,3)[scene.active].copy()
        if follow:
            aim(scene, goal)
        first_frame = scene.push_angle
        before = tip()
        past, _ = record(scene)
        factual = scene.push(Push(*ACTIONS[history[-1]]))
        np.testing.assert_allclose(state(scene), past[history[-1], -1, 0], atol=2e-5, rtol=0)
        if follow:
            aim(scene, goal)
        after = tip()
        future, score = record(scene)
        return (np.stack([past, future]), [before,after], goal,
                float(intervention['tool_block']['mass']), score, float(factual['overlap']), [first_frame, scene.push_angle])
    finally:
        scene.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jobs', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    jobs = json.loads(args.jobs.read_text())
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(sample, jobs))
    states, tips, goals, masses, scores, start_scores, frames = zip(*rows)
    np.savez_compressed(args.out, states=states, tips=tips, goals=goals, masses=masses,
                        scores=scores, start_scores=start_scores, actions=ACTIONS, angle=ANGLE, frames=frames)


if __name__ == '__main__':
    main()
