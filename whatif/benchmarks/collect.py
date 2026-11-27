"""Collect Stage 33 multi-body physical benchmarks: CoPhy BallsCF and CausalWorld Stacking2.

Run with the simulator environment: /tmp/whatif-push-venv/bin/python -m whatif.benchmarks.collect ...
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from whatif.effects.collect import PUSH_WINDOWS, ROOT
from whatif.mechanism.native import A, D, N, menu, pose, roles

PROTOCOL = json.loads(Path(__file__).with_name('protocol.json').read_text())
DATA = ROOT / 'data/benchmarks'
NATIVE_DATA = DATA / 'native'

_PB_CID = None


def _pb_client():
    global _PB_CID
    if _PB_CID is None:
        import pybullet as pb
        import pybullet_data
        _PB_CID = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=_PB_CID)
    return _PB_CID


def collect_ballscf(seed, split):
    """CoPhy BallsCF (Baradel et al., ICLR 2020): elastic/inelastic multi-ball cascades.

    Train/val/test use 3-4 balls; transfer uses 5-6 balls (unseen object count & denser chains).
    Confounders per ball follow upstream dataset_balls.py:
      MASSES=[1, 2, 5], FRICTIONS=[0.1, 0.5, 1.0], RESTITUTIONS=[0.1, 0.5, 1.0].
    """
    import pybullet as pb

    cid = _pb_client()
    spec = PROTOCOL['ballscf']
    radius = float(spec['radius'])
    times = np.array(PROTOCOL['cophy_clock'], dtype=np.int64)
    ends = np.searchsorted(times, (6, 12, 18, 24, 30))
    rng = np.random.default_rng(seed)

    counts = spec['transfer_balls'] if split == 'transfer' else spec['train_balls']
    k = int(counts[seed % len(counts)])
    masses = rng.choice(spec['masses'], k)
    frictions = rng.choice(spec['frictions'], k)
    restitutions = rng.choice(spec['restitutions'], k)

    theta = rng.uniform(-np.pi, np.pi)
    dir_v = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    side_v = np.array([-dir_v[1], dir_v[0]], dtype=np.float64)

    poses = [(-dir_v * rng.uniform(1.20, 1.48) + side_v * rng.uniform(-0.12, 0.12))]
    # Place ball 1 near the impact line so factual trajectory reliably exhibits momentum transfer.
    poses.append(dir_v * rng.uniform(-0.10, 0.28) + side_v * rng.uniform(-0.22, 0.22))
    for _ in range(2, k):
        for _try in range(300):
            cand = dir_v * rng.uniform(0.10, 1.25) + side_v * rng.uniform(-0.90, 0.90)
            if all(np.linalg.norm(cand - p) >= 2.35 * radius for p in poses):
                poses.append(cand)
                break
        else:
            # Deterministic fallback ring if dense sample fails
            ang = 2 * np.pi * len(poses) / k
            poses.append(np.array([np.cos(ang), np.sin(ang)]) * 1.45)
    poses = np.asarray(poses, dtype=np.float64)
    v0 = dir_v * rng.uniform(4.6, 6.6)

    def simulate(init_xy, remove=None):
        pb.resetSimulation(physicsClientId=cid)
        pb.setGravity(0.0, 0.0, -9.81, physicsClientId=cid)
        pb.setTimeStep(1.0 / 60.0, physicsClientId=cid)
        pb.setPhysicsEngineParameter(numSubSteps=1, restitutionVelocityThreshold=0.01, physicsClientId=cid)
        plane = pb.loadURDF('plane.urdf', physicsClientId=cid)
        pb.changeDynamics(plane, -1, lateralFriction=1.0, rollingFriction=0.05, restitution=0.0,
                          physicsClientId=cid)
        col = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius, physicsClientId=cid)
        ids = []
        for i, (xy, m, fr, re) in enumerate(zip(init_xy, masses, frictions, restitutions)):
            bid = pb.createMultiBody(baseMass=float(m), baseCollisionShapeIndex=col,
                                     basePosition=[float(xy[0]), float(xy[1]), radius],
                                     physicsClientId=cid)
            pb.changeDynamics(bid, -1, lateralFriction=float(fr), rollingFriction=0.06 * float(fr),
                              spinningFriction=0.05 * float(fr), linearDamping=0.25 * float(fr),
                              angularDamping=0.20 * float(fr), restitution=float(re),
                              physicsClientId=cid)
            if i == 0:
                pb.resetBaseVelocity(bid, [float(v0[0]), float(v0[1]), 0.0], [0.0, 0.0, 0.0],
                                     physicsClientId=cid)
            ids.append(bid)
        if remove is not None:
            pb.setCollisionFilterPair(ids[remove[0]], ids[remove[1]], -1, -1, 0, physicsClientId=cid)

        def obs():
            x = np.zeros((N, D), np.float32)
            for idx, bid in enumerate(ids):
                p, q = pb.getBasePositionAndOrientation(bid, physicsClientId=cid)
                x[idx] = pose(p, q)
            return x

        start = obs()
        lookup = {bid: idx for idx, bid in enumerate(ids)}
        edge = np.zeros((N, N), np.float32)
        snaps, contacts = [], []
        tick_set = set(times.tolist())
        for tick in range(1, int(times[-1]) + 1):
            for _ in range(2):  # 2 * (1/60s) = 1/30s per tick; check contacts every substep
                pb.stepSimulation(physicsClientId=cid)
                for cp in pb.getContactPoints(physicsClientId=cid):
                    i, j = lookup.get(cp[1]), lookup.get(cp[2])
                    if i is not None and j is not None and cp[9] > 1e-4:
                        edge[i, j] = edge[j, i] = 1.0
            if tick in tick_set:
                snaps.append(obs())
                contacts.append(edge.copy())
                edge.fill(0)
        return start, np.stack(snaps), np.stack(contacts)

    x, f_native, fe_native = simulate(poses)
    fa = np.zeros((N, A), np.float32)
    pairs = list(zip(*np.tril_indices(N, -1)))

    actions, native_poses, native_edges, native_offs = [], [], [], []
    for branch in range(PROTOCOL['branches']):
        delta = np.zeros(2, dtype=np.float64)
        j = 0
        for _try in range(200):
            j = 0 if branch < 2 else int(rng.integers(1, k))
            if j == 0:
                delta = (side_v * rng.choice([-1.0, 1.0]) * rng.uniform(0.18, 0.72)
                         + dir_v * rng.uniform(-0.28, 0.28))
            else:
                delta = rng.uniform(-0.80, 0.80, size=2)
                if np.linalg.norm(delta) < 0.24:
                    continue
            alt = poses.copy()
            alt[j] = alt[j] + delta
            if all(np.linalg.norm(alt[j] - alt[m]) >= 2.25 * radius for m in range(k) if m != j):
                break
        else:
            alt = poses.copy()
            j = 0
            delta = side_v * (0.35 if branch % 2 == 0 else -0.35)
            alt[0] = alt[0] + delta

        act = np.zeros((N, A), np.float32)
        act[j, 0] = 1.0
        act[j, 1:3] = act[j, 4:6] = delta.astype(np.float32)
        _, y_nat, e_nat = simulate(alt)

        signed = np.zeros((len(pairs), len(times), N, 3), np.float32)
        for p_idx, (src, dst) in enumerate(pairs):
            if src >= k:
                break
            _, miss_nat, _ = simulate(alt, (src, dst))
            signed[p_idx] = y_nat[..., :3] - miss_nat[..., :3]

        actions.append(act)
        native_poses.append(y_nat)
        native_edges.append(e_nat)
        native_offs.append(signed)

    native_pose = np.asarray(native_poses, np.float32)
    native_edge = np.asarray(native_edges, np.float32)
    native_off = np.asarray(native_offs, np.float32)
    fact = np.concatenate([x[None], f_native[ends]], axis=0)
    traj = native_pose[:, ends]
    contact = np.stack([native_edge[:, a:b + 1].max(1)
                        for a, b in zip(np.r_[0, ends[:-1] + 1], ends)], axis=1)
    off = native_off[:, :, ends]
    return dict(
        seed=seed,
        x=x,
        a=np.asarray(actions, np.float32),
        role=roles(list(range(k)), [], []),
        goal=np.zeros(D, np.float32),
        fa=fa,
        fact=fact,
        fe=fe_native.max(0),
        traj=traj,
        contact=contact,
        off=off,
        native_pose=native_pose,
        native_edge=native_edge,
        native_off=native_off,
    )


class _Stacking2Sim:
    """Official CausalWorld STACKING2_BENCHMARK wrapper matching WhatIf's N=6, D=9, A=7 interface."""

    def __init__(self, seed, split):
        import pybullet as pb
        from causal_world.benchmark import STACKING2_BENCHMARK
        from causal_world.configs.world_constants import WorldConstants
        from causal_world.envs import CausalWorld
        from causal_world.task_generators import generate_task

        self.pb = pb
        self.floor_id = WorldConstants.FLOOR_ID
        self.robot_id = WorldConstants.ROBOT_ID
        task = generate_task(task_generator_id='stacking2')
        self.env = CausalWorld(task=task, skip_frame=4, normalize_actions=False,
                               normalize_observations=False, seed=seed, max_episode_length=500,
                               enable_visualization=False)
        self.env.set_action_mode('end_effector_positions')
        self.env.reset()
        protos = PROTOCOL['stacking2']['transfer_protocols' if split == 'transfer' else 'train_protocols']
        pname = protos[seed % len(protos)]
        proto = next(p for p in STACKING2_BENCHMARK['evaluation_protocols'] if p.get_name() == pname)
        proto.init_protocol(self.env, None)
        np.random.seed(seed)
        for _ in range(64):
            iv = proto.get_intervention(seed, 0)
            ok, _ = self.env.set_starting_state(iv)
            if ok:
                break
        else:
            raise RuntimeError(f'Stacking2 intervention failed at seed={seed}, proto={pname}')

        self.stage = self.env._stage
        self.robot = self.env._robot
        self.client = self.env._pybullet_client_full_id
        self.bodies = [
            self.stage._rigid_objects['tool_block_1']._block_ids[0],
            self.stage._rigid_objects['tool_block_2']._block_ids[0],
        ]
        self.links = np.array(self.robot._finger_tip_ids)
        self.role = roles([0, 1], [2, 3, 4], [5])
        # Settle dropped blocks (P7/P8 spawn elevated blocks) with robot held at rest
        rest = self.robot.get_latest_full_state()['end_effector_positions'].copy()
        rest = np.clip(rest, self.env.action_space.low, self.env.action_space.high)
        for _ in range(8):
            self.env.step(rest)

    def observe(self):
        x = np.zeros((N, D), np.float32)
        for idx, bname in enumerate(('tool_block_1', 'tool_block_2')):
            p = self.stage.get_object_state(bname, 'cartesian_position')
            q = self.stage.get_object_state(bname, 'orientation')
            x[idx] = pose(p / 0.065, q)
        tips = self.robot.get_latest_full_state()['end_effector_positions'].reshape(3, 3)
        for i, xyz in enumerate(tips):
            x[2 + i] = pose(xyz / 0.065)
        x[5] = pose([0.0, 0.0, 0.0])
        gp = self.stage.get_object_state('goal_block_1', 'cartesian_position')
        gq = self.stage.get_object_state('goal_block_1', 'orientation')
        goal = pose(gp / 0.065, gq).astype(np.float32)
        return x, goal

    def disable_object_contact(self, pair):
        other, movable = sorted(map(int, pair), reverse=True)
        if movable not in (0, 1) or not self.role[other].any():
            raise ValueError(f'Invalid Stacking2 contact pair {pair}')
        body = self.bodies[movable]
        if other == 1 and movable == 0:
            self.pb.setCollisionFilterPair(self.bodies[0], self.bodies[1], -1, -1, 0,
                                           physicsClientId=self.client)
        elif other == 5:
            self.pb.setCollisionFilterPair(body, self.floor_id, -1, -1, 0,
                                           physicsClientId=self.client)
        else:
            f_idx = other - 2
            lo = self.links[f_idx - 1] + 1 if f_idx > 0 else 0
            for link in range(lo, self.links[f_idx] + 1):
                self.pb.setCollisionFilterPair(self.robot_id, body, link, -1, 0,
                                               physicsClientId=self.client)

    def execute(self, action, on_tick=None):
        x0, _ = self.observe()
        node = int(action[:, 0].argmax())
        planned = x0[node, :3].copy()
        all_edges = np.zeros((N, N), np.float32)
        for d in (action[node, 1:4], action[node, 4:7]):
            waypoint = x0[node, :3] + d
            seg = planned.copy()
            for tick in range(4):
                planned = seg + (waypoint - seg) * ((tick + 1) / 4)
                cmd = x0[2:5, :3].copy() * 0.065
                cmd[node - 2] = planned * 0.065
                cmd = np.clip(cmd.ravel(), self.env.action_space.low, self.env.action_space.high)
                _, r, _, _ = self.env.step(cmd)
                edge = np.zeros((N, N), np.float32)
                for bi, bid in enumerate(self.bodies):
                    for cp in self.pb.getContactPoints(bodyA=bid, physicsClientId=self.client):
                        if cp[9] <= 0.001:
                            continue
                        other = cp[2]
                        j = None
                        if other == self.bodies[1 - bi]:
                            j = 1 - bi
                        elif other == self.floor_id:
                            j = 5
                        elif other == self.robot_id and 0 <= cp[4] <= self.links[-1]:
                            j = 2 + int(np.searchsorted(self.links, cp[4]))
                        if j is not None:
                            edge[bi, j] = edge[j, bi] = 1.0
                np.maximum(all_edges, edge, out=all_edges)
                if on_tick is not None:
                    on_tick(self.observe()[0], edge, float(r))
        return self.observe()[0], all_edges

    def close(self):
        self.env.close()


def collect_stacking2(seed, split):
    rng = np.random.default_rng(seed)
    prefix = []
    branches = []

    def record(sim, action):
        ticks = []
        sim.execute(action, on_tick=lambda y, e, r: ticks.append((y, e, r)))
        state, edge, reward = zip(*ticks)
        starts = (0, *PUSH_WINDOWS[:-1])
        coarse = (
            np.stack([state[end - 1] for end in PUSH_WINDOWS]),
            np.stack([np.maximum.reduce(edge[start:end]) for start, end in zip(starts, PUSH_WINDOWS)]),
            np.array([np.mean(reward[start:end]) for start, end in zip(starts, PUSH_WINDOWS)], np.float32),
        )
        return coarse, ticks

    for branch in range(PROTOCOL['branches']):
        sim = _Stacking2Sim(seed, split)
        try:
            if branch == 0:
                for _ in range(1 + seed % 4):
                    f0, goal = sim.observe()
                    prefix.append(menu(f0, sim.role, rng, 1, fine=True)[0])
                    (observed, factual_edge, _), _ = record(sim, prefix[-1])
                x, goal = sim.observe()
                role = sim.role.copy()
                actions = menu(x, role, rng, PROTOCOL['branches'], fine=True)
            else:
                for act in prefix:
                    sim.execute(act)
                if not np.allclose(sim.observe()[0], x, atol=1e-5):
                    raise RuntimeError(f'stacking2 {seed}: factual replay diverged')
            branches.append(record(sim, actions[branch]))
        finally:
            sim.close()

    coarse, traces = zip(*branches)
    traj, contact, rewards = map(np.stack, zip(*coarse))
    native_pose, native_edge, native_reward = zip(*(zip(*ticks) for ticks in traces))
    native_pose = np.asarray(native_pose, np.float32)
    native_edge = np.asarray(native_edge, np.float32)
    native_reward = np.asarray(native_reward, np.float32)

    # Sample one editable pair involving either movable block (0 or 1), upweighting active / action-varying edges.
    rng_pair = np.random.default_rng(seed ^ 19973)
    i, j = np.tril_indices(N, -1)
    valid = (j <= 1) & role[i].any(-1)
    touched = native_edge[:, :, i, j].any(1)
    weight = (1 + 4 * touched.any(0) + 6 * (touched.any(0) != touched.all(0))) * valid
    pair = int(rng_pair.choice(len(i), p=weight / weight.sum()))

    edited = []
    for action in actions:
        sim = _Stacking2Sim(seed, split)
        try:
            for act in prefix:
                sim.execute(act)
            sim.disable_object_contact((i[pair], j[pair]))
            points = []
            sim.execute(action, on_tick=lambda y, e, r: points.append(y[:, :3]))
            edited.append(np.stack(points))
        finally:
            sim.close()

    native_off = native_pose[..., :3] - np.asarray(edited, np.float32)
    return dict(
        seed=seed,
        x=x,
        a=actions,
        role=role,
        goal=goal,
        fa=prefix[-1],
        fact=np.concatenate([f0[None], observed], axis=0),
        fe=factual_edge.max(0),
        traj=traj,
        contact=contact,
        rewards=rewards,
        native_pose=native_pose,
        native_edge=native_edge,
        native_reward=native_reward,
        native_off=native_off,
        removed_pair=pair,
    )


def _sample(item):
    domain, seed, split = item
    return collect_ballscf(seed, split) if domain == 'ballscf' else collect_stacking2(seed, split)


def write_split(rows, domain, split):
    DATA.mkdir(parents=True, exist_ok=True)
    NATIVE_DATA.mkdir(parents=True, exist_ok=True)
    coarse_fields = ['seed', 'x', 'a', 'role', 'goal', 'fa', 'fact', 'fe', 'traj', 'contact',
                     *(['off'] if domain == 'ballscf' else ['rewards'])]
    native_fields = ['seed', 'native_pose', 'native_edge', 'native_off',
                     *([] if domain == 'ballscf' else ['native_reward', 'removed_pair'])]
    c_path = DATA / f'{domain}-{split}.npz'
    n_path = NATIVE_DATA / f'{domain}-{split}.npz'
    np.savez_compressed(c_path, **{k: np.asarray([r[k] for r in rows]) for k in coarse_fields})
    np.savez_compressed(n_path, **{k: np.asarray([r[k] for r in rows]) for k in native_fields})

    # Summary health check
    traj = np.asarray([r['traj'] for r in rows])
    role = np.asarray([r['role'] for r in rows])
    mov = role[..., 0] > 0.5
    end = traj[:, :, -1, :, :3]
    ii, jj = np.triu_indices(end.shape[1], 1)
    diff = np.linalg.norm(end[:, ii] - end[:, jj], axis=-1)
    mov_exp = mov[:, None].repeat(len(ii), axis=1)
    delta_rms = float(np.sqrt(np.mean((diff ** 2)[mov_exp])))
    edge = np.asarray([r['contact'] for r in rows])
    edge_rate = float(edge.max(2).mean())
    print(f'{c_path.relative_to(ROOT)} n={len(rows)} delta_rms={delta_rms:.4f} edge_density={edge_rate:.4f}',
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('domain', choices=PROTOCOL['domains'])
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--split', choices=list(PROTOCOL['splits']))
    p.add_argument('--count', type=int)
    a = p.parse_args()
    splits = [a.split] if a.split else list(PROTOCOL['splits'])
    with mp.Pool(a.workers, maxtasksperchild=64) as pool:
        for split in splits:
            spec = PROTOCOL['splits'][split]
            n = a.count or spec[a.domain]
            base = spec['base']
            items = [(a.domain, base + i, split) for i in range(n)]
            rows = pool.map(_sample, items, chunksize=2)
            write_split(rows, a.domain, split)


if __name__ == '__main__':
    main()
