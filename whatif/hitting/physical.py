"""Atomic three-domain supervision, including real CoPhy collision excisions."""
import torch
from whatif.paired.fit import load_rows,loss
from whatif.recourse.physical import sources as previous_sources,draw
from .native import DATA,OUT
from .core import Core

def sources(device):
    new=load_rows(DATA/'cophy-train.npz','cophy',device,DATA/'excision-train.npz')
    return [new,*previous_sources(device)]

def objective(net,d):
    return loss(net,d)+.45*net.reconstruction(d)

if __name__=='__main__':
    torch.manual_seed(271)
    torch.set_num_threads(2)
    net=Core().cuda()
    ds=sources('cuda')
    opt=torch.optim.AdamW(net.parameters(),lr=.00033)
    for step in range(960):
        obj=objective(net,draw(ds,8))
        opt.zero_grad()
        obj.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),4.)
        opt.step()
        if (step+1)%160==0:
            print(step+1,float(obj.detach()),flush=True)
    torch.save(net.state_dict(),OUT/'initial.pt')
