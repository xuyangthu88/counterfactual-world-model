"""Native-step trajectories for the two previously identified test branches in the figure."""
import json
import numpy as np
from whatif.feedback.native import command
from whatif.mechanism.native import Simulator
from .native import BANK,OUT

if __name__=='__main__':
    example=json.loads((OUT/'example.json').read_text())
    archive={}
    for arm in example['arms']:
        sim=Simulator('pusht',example['seed'],'test')
        try:
            x,goal=sim.observe()
            path=[]
            for t in range(1,37):
                if sim.success:
                    break
                a=command(x,sim.role,goal,BANK[0 if t==1 else arm])
                def snapshot(obs,edge,reward):
                    path.append([t,*obs[0,:2],*obs[1,:2],edge[0,example['wall']],edge[0,1]])
                x,_,_=sim.execute(a,landmarks=True,stop=True,on_tick=snapshot)
            archive[str(arm)]=np.array(path)
        finally:
            sim.close()
    np.savez_compressed(OUT/'example-paths.npz',**archive)
