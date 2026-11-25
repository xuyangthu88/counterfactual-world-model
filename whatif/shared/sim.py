"""One executor per benchmark, used verbatim by collection and real control."""
import numpy as np
from shapely.geometry import Point
from whatif.crossbench.tasks import ACTIONS,create
from whatif.crossbench.surface import POLY,rot,visible,waypoints,DEST
from .actions import menu,rotate,wrap


class Scene:
    def __init__(self,task,seed,split,near=False):
        self.task=task
        initial_split='val' if split=='transfer_val' else split
        self.env=create(task,seed,('near'+initial_split) if near else initial_split)
        if task=='pusht' and split=='transfer_val':
            damping=.18 if seed%2 else 0.
            self.env.env.space.damping=damping
            self.env.mass=damping
            self.env.protocol=int(damping>0)
            self.env.physics=int(np.digitize(damping,[.04,.3]))
        self.rho=.45
        self.samples=0
        self.null=-1
        self.fixed_coast=None

    def state(self):
        return np.r_[self.env.obs(),self.rho,min(self.samples,6)/6].astype('float32')

    def push(self,index,budget=42,stop=False,noise=0.,null=-1,command=None,phases=None):
        env=self.env
        self.latest_free=[]
        if self.task=='causalworld':
            env.stop_on_success=stop
            peak_cw = [float(self.state()[12])] if phases is not None else None
            def observe(ticks, contacts, coverage):
                peak_cw[0]=max(peak_cw[0],float(coverage))
                if ticks in (24,33,42):
                    phases.append((self.state(),contacts.copy(),ticks,peak_cw[0]))
                    contacts[:]=False
            z,edge=env.run(ACTIONS[index],noise,null,observe if phases is not None else None)
            if phases is not None:
                edge=np.any([v[1] for v in phases]+[edge],axis=0)
                return self.state(),edge,env.steps,peak_cw[0]
            return self.state(),edge,env.steps,float(z[12])
        e=env.env
        initial=self.state();cmd=menu(initial)[index] if command is None else command
        site=(cmd[:2]+[0.,.45])*100
        inward=cmd[2:4]
        dest=site-17*inward
        nearest=np.argmin(np.linalg.norm(DEST-dest,axis=1))
        steps=0;success=False;peak=initial[12]
        env.edge[:]=False
        if null>=0:
            for shape in env.sources[null]:
                shape.filter=env.py.ShapeFilter(categories=1<<null,mask=env.py.ShapeFilter.ALL_MASKS() ^ 8)
        def motion():
            return np.r_[np.asarray(e.block.velocity),47*e.block.angular_velocity]
        def tick(target,limit):
            nonlocal steps,success,peak
            pos=np.asarray(e.agent.position);off=target-pos
            a=pos+off*min(1.,limit/max(np.linalg.norm(off),1e-7))
            v=motion(); raw=np.asarray(e.block.position); angle=e.block.angle
            far=POLY.distance(Point((pos-raw)@rot(angle)))>19 and np.max(np.abs(raw-256))<140
            _,reward,success,_=e.step(np.clip(a,17,495))
            vp=motion()
            if far and v@v>1. and POLY.distance(Point((np.asarray(e.agent.position)-np.asarray(e.block.position))@rot(e.block.angle)))>19:
                ratio=np.clip(vp@v/(v@v),0,.985)
                if np.linalg.norm(vp-ratio*v)<max(1.,.05*np.linalg.norm(v)):
                    self.rho=ratio if not self.samples else .75*self.rho+.25*ratio
                    self.samples+=1
                    self.latest_free.append(float(ratio))
            steps+=1;peak=max(peak,float(self.state()[12]) if phases is not None and success else float(reward)*.95)
        def remaining():
            return steps<budget and not (stop and success)
        # Approach in the frame of the moving body. Use a visibility path around the T.
        for _ in range(15):
            p=np.asarray(e.block.position);R=rot(e.block.angle);agent=(np.asarray(e.agent.position)-p)@R
            if np.linalg.norm(agent-dest)<4.5 or not remaining():break
            target=p+R@dest
            if not visible(agent,dest):
                for via in waypoints(p,R,agent,nearest):
                    if np.linalg.norm(via-np.asarray(e.agent.position))>4:break
                target=via
            for _ in range(2):
                if remaining():tick(target,30.)
        if phases is not None:
            phases.append((self.state(),env.edge.copy(),steps,peak))
            env.edge[:]=False
        # Feedback stroke. Estimate where the free body will coast from observed velocity.
        p=np.asarray(e.block.position);R=rot(e.block.angle)
        axis=R@inward;center=p+R@np.array([0.,45.]);theta=e.block.angle
        length=max(.8,100*float(cmd[4])+noise*10)
        torque=cmd[5]/.47
        denom=1+torque*torque
        for _ in range(12):
            if not remaining() or cmd[4]<=0:break
            r=rot(e.block.angle);c=np.asarray(e.block.position)+r@np.array([0.,45.])
            v=np.asarray(e.block.velocity);w=e.block.angular_velocity
            coasting=self.rho if self.fixed_coast is None else self.fixed_coast
            time=.1*coasting/max(.10,1-coasting)
            achieved=((c-center)@axis+torque*47*wrap(e.block.angle-theta))/denom
            coast=time*(v@axis+torque*47*w)/denom
            left=length-achieved-coast
            if left<.6:break
            step=np.clip(left*.65,2.,15.)
            contact=np.asarray(e.block.position)+r@dest
            tick(contact+axis*(2.+step),np.clip(step,3.,14.))
        if phases is not None:
            phases.append((self.state(),env.edge.copy(),steps,peak))
            env.edge[:]=False
        # Let the object coast while releasing the contact. All native ticks count.
        for _ in range(3):
            if not remaining():break
            contact=np.asarray(e.block.position)+rot(e.block.angle)@dest
            tick(contact-axis*4.0,7.)
        if null>=0:
            for sh in env.sources[null]:sh.filter=env.py.ShapeFilter(categories=1<<null)
        z=self.state()
        if phases is not None:
            phases.append((z,env.edge.copy(),steps,max(peak,float(z[12]))))
            return z,np.any([v[1] for v in phases],axis=0),steps,max(peak,float(z[12]))
        return z,env.edge.copy(),steps,max(peak,float(z[12]))

    def close(self):
        self.env.close()
