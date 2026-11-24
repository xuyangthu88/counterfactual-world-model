"""Short planar pushes using native CausalWorld TriFinger joint-position control."""
import copy
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pybullet as pb
from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task
from causal_world.configs.world_constants import WorldConstants


def angle(x):
    return np.arctan2(np.sin(x), np.cos(x))


@dataclass(frozen=True)
class Push:
    offset: float  # metres in world y, relative to the block centre
    travel: float  # metres along negative world x, from the approach position


class PushingScene:
    def __init__(self, xy, yaw, mass, friction, goal, seed=0):
        self.goal = np.asarray(goal)
        task = generate_task(
            task_generator_id="pushing", tool_block_mass=float(mass),
            tool_block_position=np.r_[xy, 0.0325],
            tool_block_orientation=np.array(pb.getQuaternionFromEuler([0, 0, yaw])),
            goal_block_position=np.r_[goal, 0.0325],
        )
        self.env = CausalWorld(task=task, skip_frame=4, normalize_actions=False,
                               normalize_observations=False, seed=seed, max_episode_length=500)
        self.env.reset()
        self.robot = self.env.get_robot()
        self.stage = self.env.get_stage()
        self.client = self.env._pybullet_client_full_id
        self.body = self.stage.get_rigid_objects()["tool_block"].get_block_ids()[0]
        if friction is not None:
            pb.changeDynamics(WorldConstants.FLOOR_ID, -1, lateralFriction=float(friction),
                              physicsClientId=self.client)
        self.rest = self.robot.get_latest_full_state()["end_effector_positions"].reshape(3, 3).copy()
        self.active = 1
        self.push_angle = 0.
        self.neutral = self.robot.get_latest_full_state()["positions"].copy()
        for _ in range(6):
            self.env.step(self.neutral)

    def state(self):
        pos, ori = pb.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        return np.array([pos[0], pos[1], pb.getEulerFromQuaternion(ori)[2]])

    @contextmanager
    def snapshot(self):
        env = self.env
        saved = env.get_state()
        attrs = [(env, "_episode_length")]
        attrs.extend((self.robot, k) for k in vars(self.robot) if k.startswith("_last"))
        attrs.extend((env._task, k) for k in vars(env._task)
                     if k.startswith(("previous_", "_current_goal", "_desired_goal", "_achieved_goal")))
        cache = [(o, k, copy.deepcopy(getattr(o, k))) for o, k in attrs]
        def restore():
            env.set_state(saved)
            for obj, key, val in cache:
                setattr(obj, key, copy.deepcopy(val))
        try:
            yield restore
        finally:
            restore()
            env._task._remove_pybullet_state(saved["pybullet_state"])

    def push(self, action, after_step=None):
        x, y, _ = self.state()
        pivot = np.array([x, y])
        y += action.offset
        start = x + 0.065
        waypoints = [
            (start, y, 0.091, 12), (start, y, 0.029, 12),
            (start - action.travel, y, 0.029, 9),
            (start - action.travel, y, 0.10, 9),
        ]
        contact = False
        peak_force = 0.
        total_reward = 0.
        for tx, ty, tz, ticks in waypoints:
            if self.push_angle:
                cs, sn = np.cos(self.push_angle), np.sin(self.push_angle)
                tx, ty = pivot + np.array([[cs,-sn],[sn,cs]]) @ (np.array([tx,ty])-pivot)
            tips = self.rest.copy()
            tips[self.active] = [tx, ty, tz]
            q = self.robot.get_latest_full_state()["positions"].copy()
            for _ in range(3):
                q = self.robot.inverse_kinematics(tips.ravel(), q.tolist())
            for _ in range(ticks):
                _, reward, _, info = self.env.step(q)
                total_reward += reward
                hits = pb.getContactPoints(WorldConstants.ROBOT_ID, self.body,
                                           physicsClientId=self.client)
                hi = self.robot._finger_tip_ids[self.active]
                lo = self.robot._finger_tip_ids[self.active-1] if self.active else -1
                force = sum(c[9] for c in hits if lo < c[3] <= hi)
                peak_force = max(peak_force, force)
                contact |= force > 0.001
                if after_step is not None:
                    after_step()
        return dict(end=self.state(), contact=contact, force=peak_force,
                    overlap=info["fractional_success"], reward=total_reward)

    def close(self):
        self.env.close()
