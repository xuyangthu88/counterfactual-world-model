"""Choose a continuation from real midpoint observations before collecting early labels."""
import json
import numpy as np
import torch
from .native import DATA,OUT,SPLITS,DOMAINS
from .data import load_np,menus,tensor,take
from .policy import selector,features
from .fit import restore

def midpoint(raw,device):
    d={k:raw[k] for k in ('x','role','goal','hx','ha','hy','seed','first')}
    menu=np.stack([menus(x,r,g) for x,r,g in zip(d['x'],d['role'],d['goal'])])
    d.update(f0=d['hx'][:,-1],fa=d['ha'][:,-1],f1=d['hy'][:,-1],a=menu,same=np.zeros(len(menu)))
    return tensor(d,device)

def winner(early):
    return json.loads((OUT/('early-selection.json' if early else 'late-selection.json')).read_text())['mode']

@torch.no_grad()
def run():
    torch.set_num_threads(2)
    dev='cuda'
    mode=winner(False)
    world=restore('joint',False,dev)
    decision=selector(False,dev)
    for split in SPLITS:
        for dom in DOMAINS:
            raw=load_np(DATA/f'mid-{dom}-{split}.npz')
            d=midpoint(raw,dev)
            selections=[]
            for ix in torch.arange(len(d['x']),device=dev).split(64):
                f=features(world,take(d,ix),False)
                logits=f['m'][...,0] if mode=='direct' else decision(f,mode)
                selections.append(logits.argmax(1).cpu().numpy())
            b=np.where(raw['mid_alive'],np.concatenate(selections),raw['first']).astype(np.int64)
            np.save(OUT/f'continuation-{dom}-{split}.npy',b)
            print(split,dom,mode,len(b),flush=True)
if __name__=='__main__':
    run()
