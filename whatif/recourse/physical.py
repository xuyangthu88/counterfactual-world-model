"""Native graph, magnitude and paired-effect supervision shared with the decision model."""
import torch
from whatif.paired.fit import current,old_training,load_rows,four,loss,DOMAINS
from whatif.mechanism.train import goal_training,hindsight
from .native import DATA,OUT
from .core import Core

def sources(device):
    added=load_rows(DATA/'cophy-train.npz','cophy',device,DATA/'excision-train.npz')
    return [added,old_training('cophy',device)]+[current(d,'train',device) for d in DOMAINS]

def draw(sources,per):
    parts=[]
    for d in sources:
        ix=torch.randint(len(d['x']),(per,),device=d['x'].device)
        parts.append(four(d,ix))
    row={k:torch.cat([p[k] for p in parts]) for k in parts[0]}
    b=len(row['x'])
    row=hindsight(row,torch.rand(b,device=ix.device)<.25,torch.randint(4,(b,),device=ix.device))
    return goal_training(row,True)

if __name__=='__main__':
    torch.manual_seed(271)
    torch.set_num_threads(2)
    net=Core().cuda()
    rows=sources('cuda')
    opt=torch.optim.AdamW(net.parameters(),lr=.00035)
    for step in range(960):
        obj=loss(net,draw(rows,9))
        opt.zero_grad()
        obj.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),4)
        opt.step()
        if (step+1)%160==0:
            print(step+1,float(obj.detach()),flush=True)
    torch.save(net.state_dict(),OUT/'physical-init.pt')
