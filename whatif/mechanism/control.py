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


def worker(domain,seed,connection,fine):
    sim=Simulator(domain,seed,'control')
    try:
        f0,goal=sim.observe()
        rng=np.random.default_rng(seed)
        fa=menu(f0,sim.role,rng,1,fine)[0]
        x,_,r=sim.execute(fa,stop=True,fine=fine)
        best,last=float(r[2]),float(r[1])
        moves=1
        while not sim.success and moves<12:
            # Common random numbers across policies, even when their states diverge.
            key=seed*31+moves
            cmd=menu(x,sim.role,np.random.default_rng(key),24,fine)
            d=dict(f0=f0,fa=fa,f1=x,x=x,role=sim.role,goal=goal,a=cmd,same=np.float32(0))
            connection.send(('select',d,key))
            choice=connection.recv()
            f0=x
            fa=cmd[choice]
            x,_,r=sim.execute(fa,stop=True,fine=fine)
            last=float(r[1])
            best=max(best,float(r[2]))
            moves+=1
        connection.send(('end',dict(seed=seed,success=int(sim.success),terminal=last,peak=best,moves=moves)))
    finally:
        sim.close()
        connection.close()


def server(domain,start,n,fine):
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
        worker(domain,seed,Channel(),fine)
    send(('done',))


def policy_run(job):
    import pickle,struct
    import torch
    from .planner import select
    from .train import build,OUT
    domain,policy,start,n,sim_python,fine=job
    torch.set_num_threads(1)
    name=policy if policy in ('pets','cophynet') else 'mechanism'
    model=build(name,'cpu')
    model_dir=OUT/'fine' if fine else OUT
    model.load_state_dict(torch.load(model_dir/f'{name}.pt',map_location='cpu'))
    model.eval()
    fd,writer=os.pipe()
    env=dict(os.environ,WHATIF_RESPONSE_FD=str(writer),PYTHONPATH=str(ROOT/'work/stage7/upstream/pusht')+':'+str(ROOT),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
    proc=subprocess.Popen([sim_python,'-m','whatif.mechanism.control','--server',domain,str(start),str(n),str(int(fine))],
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
                    idx=select(model,policy,d,msg[2],2 if fine else 1)
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
        return server(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),bool(int(sys.argv[5])))
    from .planner import POLICIES,CONTROL_N
    from .train import OUT
    p=argparse.ArgumentParser()
    p.add_argument('--sim-python',required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--fine',action='store_true')
    args=p.parse_args()
    base,n=(823000,96) if args.fine else (423000,CONTROL_N)
    destination=OUT/'fine' if args.fine else OUT
    jobs=[(domain,policy,base+i,min(8,n-i),args.sim_python,args.fine) for i in range(0,n,8)
          for domain in ['causalworld','pusht'] for policy in POLICIES]
    rows=[]
    with mp.get_context('fork').Pool(args.workers) as pool:
        for part in pool.imap_unordered(policy_run,jobs):
            rows.extend(part)
            print('episodes',len(rows),flush=True)
    rows.sort(key=lambda r:(r['domain'],r['policy'],r['seed']))
    (destination/'control.json').write_text(json.dumps(rows,indent=1))


if __name__=='__main__':
    main()

