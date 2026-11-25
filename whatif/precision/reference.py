"""Generate official Shapely reference values at independently sampled validation poses."""

import numpy as np
from .collect import DATA


def main():
    from diffusion_policy.env.pusht.pusht_env import PushTEnv, pymunk_to_shapely

    env = PushTEnv()
    env.seed(11)
    env.reset()
    rng = np.random.default_rng(621777)
    n = 18000
    goal = np.r_[np.zeros(2), env.goal_pose[2]]
    # Cover the full plane plus dense perturbations around the native goal.
    sig = np.repeat([0.9, 0.18, 0.016], n // 3)
    p = goal + rng.normal(size=(n, 3)) * sig[:, None]
    p = np.concatenate(
        [p, np.array([goal, goal + [1.0, 0, 0], goal + [0.0, 0, np.pi], goal + [0.015, 0, 0]])]
    )
    expected = []
    geom = pymunk_to_shapely(env._get_goal_pose_body(env.goal_pose), env.block.shapes)
    for x, y, theta in p:
        body = env._get_goal_pose_body(np.array([256 + 100 * x, 256 + 100 * y, theta]))
        expected.append(
            geom.intersection(pymunk_to_shapely(body, env.block.shapes)).area / geom.area
        )
    np.savez_compressed(DATA / "geometry-ref.npz", pose=p, goal=goal, expected=expected)
    env.close()


if __name__ == "__main__":
    main()
