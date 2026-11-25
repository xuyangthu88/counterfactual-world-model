"""Native observations, actuator commands and simulator-only training labels."""
import numpy as np

N, D, A = 6, 9, 7
TICKS = 12


def pose(xyz, quat=(0., 0., 0., 1.)):
    x, y, z, w = quat
    return np.r_[xyz, 1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
                 2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)]


def planar(xy, angle=0.):
    return pose([*xy, 0.], [0, 0, np.sin(angle/2), np.cos(angle/2)])


def roles(movable, actors, fixed):
    r = np.zeros((N, 3), np.float32)
    for k, nodes in enumerate([movable, actors, fixed]):
        r[nodes, k] = 1
    return r


def menu(state, role, rng, count, fine=False):
    """Shared object-relative waypoint proposals; no goal or benchmark argument."""
    movers = np.flatnonzero(role[:, 0])
    actors = np.flatnonzero(role[:, 1])
    actions = np.zeros((count, N, A), np.float32)
    phase = rng.uniform(-np.pi, np.pi)
    for i in range(count):
        center = state[rng.choice(movers), :3]
        angle = phase + 2*np.pi*(i+rng.uniform(-.35, .35))/count
        direction = np.array([np.cos(angle), np.sin(angle), 0.])
        side = np.array([-direction[1], direction[0], 0.])
        offset = side*rng.uniform(-.65, .65)
        entry = center - direction*rng.uniform(.85, 1.4) + offset
        if fine:
            entry = center-direction*rng.uniform(.38,1.35)+offset
            end = entry+direction*np.exp(rng.uniform(np.log(.06),np.log(2.0)))
        else:
            end = center + direction*rng.uniform(.15, 1.15) + offset
        node = actors[np.linalg.norm(state[actors, :2]-entry[:2], axis=1).argmin()]
        actions[i, node] = np.r_[1, entry-state[node,:3], end-state[node,:3]]
    return actions


class Simulator:
    def __init__(self, domain, seed, split):
        self.domain = domain
        self.success = False
        self.edge = np.zeros((N, N), np.float32)
        if domain == 'causalworld':
            from whatif.crossbench.tasks import World
            from causal_world.configs.world_constants import WorldConstants
            # Only reset and official P1/P2/P7/P8 interventions are reused.
            self.world = World(seed, split)
            self.scene = self.world.scene
            self.env = self.scene.env
            self.env.set_action_mode('end_effector_positions')
            self.raw = self.env._task.filter_structured_observations()
            self.floor = WorldConstants.FLOOR_ID
            self.role = roles([0], [1,2,3], [4])
        else:
            import pymunk
            from diffusion_policy.env.pusht.pusht_env import PushTEnv
            damping = ((0., .08, .55)[seed%3] if split in ('train','val')
                       else (0., .25)[seed%2])
            self.env = env = PushTEnv(damping=damping)
            env.seed(seed)
            self.raw = env.reset()
            self.role = roles([0], [1], [2,3,4,5])
            walls = sorted((s for s in env.space.shapes if isinstance(s,pymunk.Segment)),
                           key=lambda s: (s.a.x+s.b.x,s.a.y+s.b.y))
            self.walls = [(np.asarray(w.a)+np.asarray(w.b))/2 for w in walls]
            groups = [list(env.block.shapes), list(env.agent.shapes)] + [[w] for w in walls]
            for i, shapes in enumerate(groups):
                for shape in shapes:
                    shape.collision_type = i+1
            for i in range(1, len(groups)):
                def hit(arbiter, space, data, i=i):
                    if arbiter.total_impulse.length > 1e-6:
                        self.edge[0,i] = self.edge[i,0] = 1
                    if i == 1:
                        env._handle_collision(arbiter, space, data)
                env.space.add_collision_handler(1, i+1).post_solve = hit

    def observe(self):
        x = np.zeros((N,D),np.float32)
        o = self.raw
        if self.domain == 'causalworld':
            x[0] = pose(o[32:35]/.065, o[35:39])
            for i, xyz in enumerate(o[19:28].reshape(3,3)):
                x[i+1] = pose(xyz/.065)
            x[4] = pose([0,0,0])
            goal = pose(o[49:52]/.065,o[52:56])
        else:
            x[0] = planar((o[2:4]-256)/100, o[4])
            x[1] = planar((o[:2]-256)/100)
            for i, p in enumerate(self.walls):
                x[2+i] = planar((p-256)/100)
            g = self.env.goal_pose
            goal = planar((g[:2]-256)/100, g[2])
        return x, goal.astype(np.float32)

    def execute(self, action, stop=False, fine=False):
        x, _ = self.observe()
        node = action[:,0].argmax()
        self.edge.fill(0)
        rewards, scores = [], []
        done = False
        planned = x[node,:3].copy()
        for d in (action[node,1:4], action[node,4:7]):
            waypoint = x[node,:3]+d
            if self.domain == 'causalworld':
                command = x[1:4,:3].copy()*.065
                command[node-1] = waypoint*.065
                command = np.clip(command.ravel(),self.env.action_space.low,self.env.action_space.high)
            else:
                command = np.clip(waypoint[:2]*100+256,0,512)
            for _ in range(TICKS):
                if fine:
                    step = waypoint-planned
                    planned = planned+step*min(1., .12/max(1e-9,np.linalg.norm(step)))
                    if self.domain=='causalworld':
                        command = command.reshape(3,3)
                        command[node-1] = planned*.065
                        command = np.clip(command.ravel(),self.env.action_space.low,self.env.action_space.high)
                    else:
                        command = np.clip(planned[:2]*100+256,0,512)
                self.raw,r,done,info = self.env.step(command)
                if self.domain == 'causalworld':
                    score = info['fractional_success']
                    s, w = self.scene, self.world
                    for p in w.pb.getContactPoints(bodyA=s.body, physicsClientId=s.client):
                        j = None
                        if p[2] == self.floor:
                            j = 4
                        elif p[2] == w.robot_id and 0 <= p[4] <= w.links[-1]:
                            j = 1+np.searchsorted(w.links,p[4])
                        if j is not None and p[9] > .001:
                            self.edge[0,j] = self.edge[j,0] = 1
                    reached = score > .9
                else:
                    score = float(r)
                    reached = done
                self.success |= bool(reached)
                rewards.append(r)
                scores.append(score)
                if stop and (reached or done):
                    break
            if stop and (reached or done):
                break
        y, _ = self.observe()
        return y, self.edge.copy(), np.array([np.mean(rewards), scores[-1], max(scores)],np.float32)

    def close(self):
        self.env.close()

