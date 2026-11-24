"""Train and select checkpoints only on one/two-cube data."""
import argparse
from pathlib import Path
from time import monotonic
import torch
from .chain_model import Branches,fit


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/chains"))
    p.add_argument("--out",type=Path,default=Path("results/chains"))
    p.add_argument("--device",default="cpu")
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--seeds",nargs="+",type=int,default=[11,29,47])
    args=p.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    train,val=[Branches(args.data/(name+".npz"),args.device) for name in ("train","val")]
    for seed in args.seeds:
        for gated in (True,False):
            name=("graph" if gated else "dense")+f"-{seed}"
            start=monotonic()
            fit(train,val,seed,gated,args.out/(name+".pt"),args.epochs)
            print(name,round(monotonic()-start,1),"seconds",flush=True)


if __name__=="__main__":
    main()
