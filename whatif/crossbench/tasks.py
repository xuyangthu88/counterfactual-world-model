"""Two upstream simulators with the same finite action interface and pose units."""
import numpy as np

# angle relative to goal direction, transverse offset and stroke in object units.
ACTIONS = np.array([(0, 0, .65), (0, 0, 1.10), (0, -.42, .85), (0, .42, .85),
                    (-.60, -.38, .85), (.60, .38, .85), (-1.15, .4, .90),
                    (1.15, -.4, .90), (0, 1.65, .65), (0, 0, .34),
                    (-.75, 0, .55), (.75, 0, .55)], dtype=float)
TASKS = ('causalworld', 'pusht')
SIZES = (0.065, 100.)
BASE = {'train': 61000, 'val': 71000, 'test': 81000, 'control': 91000, 'neartrain': 96000, 'nearval': 97000, 'controlval': 99000}
COUNTS = {'train': 600, 'val': 96, 'test': 192, 'neartrain': 240, 'nearval': 64}


def directions(pos, goal, action):
    theta = np.arctan2(*(goal-pos)[::-1])+action[0]
    forward = np.array([np.cos(theta), np.sin(theta)])
    return forward, np.array([-forward[1], forward[0]])


class GoalReached(Exception):
    pass


class World:
    def __init__(self, seed, split):
        import pybullet as pb
        from causal_world.benchmark.benchmarks import PUSHING_BENCHMARK
        from causal_world.configs.world_constants import WorldConstants
        from whatif.simulator import PushingScene
        self.pb = pb
        self.robot_id = WorldConstants.ROBOT_ID
        self.scene = s = PushingScene([0,-.08], 0, .02, None, [0,.08], seed)
        # Default configurations with mass variation, and joint layout/mass interventions.
        ab = 1 if split in ('test', 'control') and seed % 4 >= 2 else 0
        self.protocol = (1 if seed % 2 == 0 else 7) + ab
        protocol = next(p for p in PUSHING_BENCHMARK['evaluation_protocols']
                        if p.get_name() == f'P{self.protocol}')
        protocol.init_protocol(s.env,None)
        np.random.seed(seed)
        for _ in range(64):
            iv = protocol.get_intervention(seed,0)
            ok, _ = s.env.set_starting_state(iv)
            if ok:
                break
        else:
            raise RuntimeError(f'CausalWorld intervention {seed}')
        self.goal = s.stage.get_object_state('goal_block', 'cartesian_position')[:2]
        self.mass = float(s.stage.get_object_state('tool_block', 'mass'))
        self.physics = int(np.digitize(self.mass, [.025,.035]))
        self.rest = s.robot.get_latest_full_state()['end_effector_positions'].reshape(3,3).copy()
        s.rest = self.rest.copy()
        self.links = np.array(s.robot._finger_tip_ids)
        self.scale = SIZES[0]

    def obs(self):
        s = self.scene
        pos = s.state()
        v, w = self.pb.getBaseVelocity(s.body,physicsClientId=s.client)
        tips = s.robot.get_latest_full_state()['end_effector_positions'].reshape(3,3)[:,:2]
        t = s.env._task
        cover = t._goal_reward(t.get_achieved_goal(), t.get_desired_goal())
        return np.r_[pos[:2]/self.scale,pos[2],tips.ravel()/self.scale,self.goal/self.scale,
                     0.,cover,0.,np.array(v[:2])*.1/self.scale,w[2]*.1]

    def run(self, action, noise=0., null=-1):
        from whatif.simulator import Push
        s = self.scene
        f, _ = directions(s.state()[:2], self.goal, action)
        s.active = int(np.linalg.norm(self.rest[:,:2]-(s.state()[:2]-self.scale*f),axis=1).argmin())
        s.push_angle = np.arctan2(-f[1],-f[0])
        edge = np.zeros(3,bool)
        def toggle(value):
            if null < 0:
                return
            lo = self.links[null-1]+1 if null else 0
            for link in range(lo,self.links[null]+1):
                self.pb.setCollisionFilterPair(self.robot_id,s.body,link,-1,value,physicsClientId=s.client)
        self.steps = 0
        def watch():
            self.steps += 1
            for hit in self.pb.getContactPoints(self.robot_id,s.body,physicsClientId=s.client):
                if hit[9] > .001 and 0 <= hit[3] <= self.links[-1]:
                    edge[np.searchsorted(self.links,hit[3])] = True
            if getattr(self,"stop_on_success",False) and s.env._task._current_goal_reward > .90:
                raise GoalReached
        toggle(0)
        try:
            s.push(Push(-action[1]*self.scale,(action[2]+noise)*self.scale), after_step=watch)
        except GoalReached:
            pass
        finally:
            toggle(1)
        return self.obs(),edge

    def close(self):
        self.scene.close()


class Tee:
    def __init__(self, seed, split):
        import pymunk
        from diffusion_policy.env.pusht.pusht_env import PushTEnv
        self.py = pymunk
        damp = (0., .08, .55)[seed%3] if split in ('train','val','neartrain','nearval','controlval') else (0. if seed%2==0 else .25)
        self.mass = damp
        self.protocol = int(damp != 0.)
        self.physics = int(np.digitize(damp,[.04,.3]))
        start = None
        if split.startswith('near'):
            rng = np.random.default_rng(seed)
            # Goal-near reset curriculum for fine alignment; evaluation keeps native resets.
            pos = np.array([256.,256.]) + rng.normal(size=2)*rng.uniform(2.,52.)
            yaw = np.pi/4 + rng.normal()*rng.uniform(.04,1.1)
            direction = rng.uniform(-np.pi,np.pi)
            agent = pos + rng.uniform(70,145)*np.array([np.cos(direction),np.sin(direction)])
            start = np.r_[agent,pos,yaw]
        self.env = e = PushTEnv(damping=damp,reset_to_state=start)
        e.seed(int(seed))
        e.reset()
        self.goal = e.goal_pose[:2]
        self.scale = SIZES[1]
        self.sources = [list(e.agent.shapes),[],[]]
        for shape in e.space.shapes:
            if isinstance(shape,pymunk.Segment):
                self.sources[1 if abs(shape.b.x-shape.a.x)>0 else 2].append(shape)
        for sh in e.block.shapes:
            sh.collision_type = 4
            sh.filter = pymunk.ShapeFilter(categories=8)
        self.edge = np.zeros(3,bool)
        for i, group in enumerate(self.sources):
            for sh in group:
                sh.collision_type = i+1
                sh.filter = pymunk.ShapeFilter(categories=1<<i)
            def watch(arbiter, space, data, i=i):
                if arbiter.total_impulse.length > 1e-6:
                    self.edge[i]=True
            e.space.add_collision_handler(i+1,4).post_solve = watch

    def obs(self):
        from diffusion_policy.env.pusht.pusht_env import pymunk_to_shapely
        e = self.env
        b = np.asarray(e.block.position)
        agent = np.asarray(e.agent.position)
        wall_h = [b[0], 5. if b[1]<256 else 506.]
        wall_v = [5. if b[0]<256 else 506., b[1]]
        goal_shape = pymunk_to_shapely(e._get_goal_pose_body(e.goal_pose),e.block.shapes)
        cover = goal_shape.intersection(pymunk_to_shapely(e.block,e.block.shapes)).area / goal_shape.area
        points = (np.array([agent,wall_h,wall_v])-256.)/self.scale
        return np.r_[(b-256.)/self.scale,e.block.angle,points.ravel(),(self.goal-256.)/self.scale,
                     e.goal_pose[2],cover,1.,np.asarray(e.block.velocity)*.1/self.scale,e.block.angular_velocity*.1]

    def run(self, action, noise=0., null=-1):
        e = self.env
        pos = np.asarray(e.block.position)
        f, side = directions(pos, self.goal, action)
        start = pos - 1.12*self.scale*f + action[1]*self.scale*side
        end = start + (action[2]+noise+.45)*self.scale*f
        agent = np.asarray(e.agent.position)
        # Approach around the body, then slide in along the requested line of action.
        from_angle = np.arctan2(*(agent-pos)[::-1])
        dest_angle = np.arctan2(*(start-pos)[::-1])
        arc = np.arctan2(np.sin(dest_angle-from_angle),np.cos(dest_angle-from_angle))
        self.edge[:] = False
        if null >= 0:
            for sh in self.sources[null]:
                sh.filter = self.py.ShapeFilter(categories=1<<null,mask=self.py.ShapeFilter.ALL_MASKS() ^ 8)
        try:
            trajectory = []
            for frac in (.35,.70,1.):
                a = from_angle+arc*frac
                target = pos+1.52*self.scale*np.array([np.cos(a),np.sin(a)])
                trajectory.extend([target]*2)
            trajectory.extend([start]*3+[end]*5)
            self.steps = 0
            for target in trajectory:
                _,_,done,_ = e.step(np.clip(target,12,500))
                self.steps += 1
                if getattr(self,"stop_on_success",False) and done:
                    break
        finally:
            if null >= 0:
                for sh in self.sources[null]:
                    sh.filter = self.py.ShapeFilter(categories=1<<null)
        return self.obs(), self.edge.copy()

    def close(self):
        self.env.close()


def create(task,seed,split):
    return (World if task=='causalworld' else Tee)(seed,split)
