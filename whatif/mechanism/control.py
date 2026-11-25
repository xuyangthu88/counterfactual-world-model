"""Run multiple independent native simulators, scoring a live observation at each decision."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from .native import Simulator,menu

ROOT=Path(__file__).resolve().parents[2]


def worker(domain,seed,connection,fine,proposal):
    sim=Simulator(domain,seed,'control')
    try:
        f0,goal=sim.observe()
        rng=np.random.default_rng(seed)
        if proposal:
            from .proposals import Proposal
            actor=Proposal()
            fa=actor.candidates(f0,sim.role,goal,rng,1)[0][0]
        else:
            fa=menu(f0,sim.role,rng,1,fine)[0]
        x,_,r=sim.execute(fa,stop=True,fine=fine,landmarks=proposal)
        best,last=float(r[2]),float(r[1])
        moves=1
        trace=None
        if proposal and seed==1223000:
            trace={'goal':goal.tolist(),'poses':[f0[0].tolist(),x[0].tolist()]}
        while not sim.success and moves<(36 if proposal else 12):
            # Common random numbers across policies, even when their states diverge.
            key=seed*31+moves
            if proposal:
                cmd,prior=actor.candidates(x,sim.role,goal,np.random.default_rng(key),24)
            else:
                cmd=menu(x,sim.role,np.random.default_rng(key),24,fine)
                prior=None
            d=dict(f0=f0,fa=fa,f1=x,x=x,role=sim.role,goal=goal,a=cmd,same=np.float32(0))
            connection.send(('select',d,key,prior))
            choice=connection.recv()
            f0=x
            fa=cmd[choice]
            x,_,r=sim.execute(fa,stop=True,fine=fine,landmarks=proposal)
            last=float(r[1])
            best=max(best,float(r[2]))
            moves+=1
            if trace is not None:
                trace['poses'].append(x[0].tolist())
        row=dict(seed=seed,success=int(sim.success),terminal=last,peak=best,moves=moves)
        if trace is not None:
            row['trace']=trace
        connection.send(('end',row))
    finally:
        sim.close()
        connection.close()


def server(domain,start,n,fine,proposal):
    """Simulator process uses Python 3.10. A binary local pipe carries one message per action."""
    import pickle,struct
    read=sys.stdin.buffer
    write=os.fdopen(int(os.environ['WHATIF_RESPONSE_FD']),'wb',buffering=0)
    def send(obj):
        data=pickle.dumps(obj,protocol=4)
        write.write(struct.pack('!I',len(data))+data)
    for seed in range(start,start+n):
        class Channel:
            def send(self,value):
                send(value)
            def recv(self):
                return int(read.readline())
            def close(self):
                pass
        worker(domain,seed,Channel(),fine,proposal)
    send(('done',))


def policy_run(job):
    import pickle,struct
    import torch
    from .planner import select
    from .train import build,OUT
    domain,policy,start,n,sim_python,fine,proposal,continuation=job
    torch.set_num_threads(1)
    name=policy if policy in ('pets','cophynet') else 'mechanism'
    model=build(name,'cpu')
    model_dir=OUT/'fine' if fine else OUT
    if proposal:
        model_dir=OUT/'demonstrations'
    model.load_state_dict(torch.load(model_dir/f'{name}.pt',map_location='cpu'))
    model.eval()
    head=None
    if continuation and policy not in ('actor','previous'):
        from .return_model import load_head,features,score
        head=load_head('pets' if policy=='pets' else 'graph')
    fd,writer=os.pipe()
    env=dict(os.environ,WHATIF_RESPONSE_FD=str(writer),PYTHONPATH=str(ROOT/'work/stage7/upstream/pusht')+':'+str(ROOT),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
    proc=subprocess.Popen([sim_python,'-m','whatif.mechanism.control','--server',domain,str(start),str(n),str(int(fine)),str(int(proposal))],
                          pass_fds=[writer],env=env,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL)
    os.close(writer)
    rows=[]
    try:
        with os.fdopen(fd,'rb') as source,torch.no_grad():
            while True:
                header=source.read(4)
                if len(header)!=4:
                    raise RuntimeError(f'native simulator terminated {domain} {proc.poll()}')
                size=struct.unpack('!I',header)[0]
                msg=pickle.loads(source.read(size))
                if msg[0]=='done':
                    break
                if msg[0]=='end':
                    rows.append(dict(domain=domain,policy=policy,**msg[1]))
                else:
                    d={k:torch.as_tensor(v).float().unsqueeze(0) for k,v in msg[1].items()}
                    prior=None if msg[3] is None else torch.as_tensor(msg[3])
                    if head is not None:
                        idx=score(head,features(model,d),prior[None],policy).item()
                    else:
                        selector='delta' if policy=='previous' else policy
                        idx=select(model,selector,d,msg[2],2 if fine or proposal else 1,prior)
                    proc.stdin.write(f'{idx}\n'.encode())
                    proc.stdin.flush()
        if proc.wait()!=0:
            raise RuntimeError('simulator error')
    finally:
        proc.stdin.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    return rows


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--server':
        return server(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),bool(int(sys.argv[5])),bool(int(sys.argv[6])))
    from .planner import POLICIES,CONTROL_N
    from .train import OUT
    p=argparse.ArgumentParser()
    p.add_argument('--sim-python',required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--fine',action='store_true')
    p.add_argument('--proposal',action='store_true')
    p.add_argument('--returns',action='store_true')
    args=p.parse_args()
    base,n=(823000,96) if args.fine else (423000,CONTROL_N)
    destination=OUT/'fine' if args.fine else OUT
    policies=POLICIES
    if args.proposal:
        base,n=1223000,96
        destination=OUT/'demonstrations'
        policies=POLICIES+['actor']
    if args.returns:
        base,n=1723000,128
        destination=OUT/'continuations'
        policies=['actor','previous','value','delta','delta_kl','pets']
    jobs=[(domain,policy,base+i,min(8,n-i),args.sim_python,args.fine,
           args.proposal or args.returns,args.returns) for i in range(0,n,8)
          for domain in ['causalworld','pusht'] for policy in policies]
    rows=[]
    with mp.get_context('fork').Pool(args.workers) as pool:
        for part in pool.imap_unordered(policy_run,jobs):
            rows.extend(part)
            print('episodes',len(rows),flush=True)
    rows.sort(key=lambda r:(r['domain'],r['policy'],r['seed']))
    (destination/'control.json').write_text(json.dumps(rows,indent=1))


if __name__=='__main__':
    main()

