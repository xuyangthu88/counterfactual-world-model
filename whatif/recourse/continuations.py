"""Evaluate the frozen second decision on measured midpoints before labeling its return."""
import numpy as np
import torch
from .native import DOMAINS,ROOT,OUT,BANK
from .backup_data import SPLITS
from .fit import restore
from .load import read,take,early,late

CACHE=ROOT/'work/recourse'
@torch.no_grad()
def run():
    torch.set_num_threads(2)
    net=restore('cuda')
    CACHE.mkdir(exist_ok=True,parents=True)
    for split in SPLITS:
        for dom in DOMAINS:
            d=read(dom,split,'cuda',backup=True)
            selected=[]
            for ix in torch.arange(len(d['x']),device='cuda').split(64):
                row=take(d,ix)
                b=net.last_scores(late(row),row['first'])['mean'].argmax(1)
                selected.append(torch.where(row['midscore'][:,0].bool(),row['first'],b).cpu())
            np.save(OUT/f'continuation-{dom}-{split}.npy',torch.cat(selected).numpy())
            idx=torch.arange(0,len(d['x']),len(BANK),device='cuda')
            z=[net.encode(early(take(d,ix)),8)[0].cpu() for ix in idx.split(48)]
            np.save(CACHE/f'features-{dom}-{split}.npy',torch.cat(z).numpy())
            print(dom,split,len(idx),flush=True)

if __name__=='__main__':
    run()
