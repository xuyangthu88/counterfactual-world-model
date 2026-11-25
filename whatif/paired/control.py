"""Original reference policies and matched Stage 22 ablations in the native environments."""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import struct
import subprocess
import sys
import numpy as np
from .data import OUT,ROOT,TEST_CONTROL

POLICIES=['actor','stage21','pets','marginal','pair_mean','pair_law','pair_kl']


def batch(job):
    import torch
    from whatif.mechanism.return_model import backbone,features,load_head,score
    from .fit import world
    from .decision import ModalAction,decision_head,trajectory_features,choices
    torch.set_num_threads(1)
    domain,policy,start,n,sim_python,*optional=job
    option=optional[0] if optional else None
    paired=policy not in ('actor','stage21','pets')
    segments=bool(option and option.get('segments') and paired)
    if paired:
        model,actor,head=world(),ModalAction(),decision_head(segments)
        config=json.loads((OUT/'selection.json').read_text())['config'] if option is None else option['config']
        if segments:
            from .segment_control import SegmentPlanner
            segment_policy=SegmentPlanner(model,actor,head,policy,config)
        elif policy.startswith('combined'):
            reference,reference_head=backbone('graph'),load_head('graph')
    elif policy!='actor':
        name='pets' if policy=='pets' else 'graph'
        model,head=backbone(name),load_head(name)
    fd,writer=os.pipe()
    env=dict(os.environ,WHATIF_RESPONSE_FD=str(writer),PYTHONPATH=str(ROOT/'work/stage7/upstream/pusht')+':'+str(ROOT),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
    proc=subprocess.Popen([sim_python,'-m','whatif.paired.control','native',domain,str(start),str(n),str(int(paired and option is None)),str(int(segments))],
                          pass_fds=[writer],env=env,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL)
    os.close(writer)
    rows=[]
    switches=kl_changes=decisions=cooldown=overrides=locked=reference_choices=0
    try:
        with os.fdopen(fd,'rb') as pipe,torch.no_grad():
            while True:
                header=pipe.read(4)
                if len(header)!=4:
                    raise RuntimeError(f'{domain}: simulator pipe closed, status {proc.poll()}')
                msg=pickle.loads(pipe.read(struct.unpack('!I',header)[0]))
                if msg[0]=='done':
                    break
                if msg[0]=='end':
                    row=dict(domain=domain,policy=policy,decisions=decisions,switches=switches,
                             kl_changes=kl_changes if paired else None,**msg[1])
                    if option is not None and not segments:
                        row.update(id=option['id'],overrides=overrides,locked=locked,reference_choices=reference_choices)
                    if segments:
                        row['kl_changes']=segment_policy.counts['kl_disagreements']
                        row.update(segment_policy.counts)
                        segment_policy.reset()
                    rows.append(row)
                    switches=kl_changes=decisions=cooldown=overrides=locked=reference_choices=0
                    continue
                prior=torch.tensor(msg[3])[None]
                d={k:torch.as_tensor(v).float()[None] for k,v in msg[1].items()}
                if segments:
                    cmd=segment_policy.act(d,prior,decisions)
                    selected=int(not np.allclose(cmd,msg[1]['a'][0],atol=1e-7))
                elif paired and option is not None:
                    from .option import choose
                    d['remaining']=torch.tensor([35-decisions],dtype=torch.float)
                    selected=0
                    if cooldown:
                        cooldown-=1
                        locked+=1
                    else:
                        if policy.startswith('combined'):
                            selected=score(reference_head,features(reference,d),prior,'delta_kl').item()
                            reference_choices+=int(selected!=0)
                        if selected==0:
                            f=trajectory_features(model,actor,d)
                            i=torch.arange(f.size(1))
                            o=head(f,i,torch.zeros_like(i))
                            selected=choose(o,prior,policy,config)
                            kl_changes+=int(choose(o,prior,'pair_law',config)!=choose(o,prior,'pair_kl',config))
                            if selected!=0:
                                overrides+=1
                                cooldown=config['hold']
                elif paired:
                    f=trajectory_features(model,actor,d)
                    i=torch.arange(f.size(1))
                    o=head(f,i,torch.zeros_like(i))
                    selected=choices(o,prior,policy,config).item()
                    kl_changes+=int(choices(o,prior,'pair_law',config).item()!=choices(o,prior,'pair_kl',config).item())
                elif policy=='actor':
                    selected=0
                else:
                    selected=score(head,features(model,d),prior,'delta_kl').item()
                switches+=int(selected!=0)
                decisions+=1
                response=json.dumps(cmd.tolist()) if segments else str(selected)
                proc.stdin.write((response+'\n').encode())
                proc.stdin.flush()
        if proc.wait()!=0:
            raise RuntimeError('native simulation failed')
    finally:
        proc.stdin.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    return rows


def main():
    if len(sys.argv)>1 and sys.argv[1]=='native':
        from whatif.mechanism.control import server
        from .data import proposals
        menu_builder=proposals if int(sys.argv[5]) else None
        return server(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),False,True,menu_builder,bool(int(sys.argv[6])))
    parser=argparse.ArgumentParser()
    parser.add_argument('--sim-python',required=True)
    parser.add_argument('--workers',type=int,default=20)
    parser.add_argument('--group',choices=['previous','new','all'],default='all')
    args=parser.parse_args()
    group={'previous':POLICIES[:3],'new':POLICIES[3:],'all':POLICIES}[args.group]
    rows=[]
    jobs=[(dom,name,TEST_CONTROL+i,8,args.sim_python) for i in range(0,128,8)
          for dom in ('causalworld','pusht') for name in group]
    with mp.get_context('fork').Pool(args.workers) as pool:
        for part in pool.imap_unordered(batch,jobs):
            rows.extend(part)
            print('completed',len(rows),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    rows.sort(key=lambda r:(r['domain'],r['policy'],r['seed']))
    (OUT/f'control-{args.group}.json').write_text(json.dumps(rows,separators=(',',':')))


if __name__=='__main__':
    main()
