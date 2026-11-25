"""Frozen Stage 07 controller on the new Stage 08 control layouts."""
import csv
from concurrent.futures import ProcessPoolExecutor
from whatif.crossbench.feedback import episode,matched_setup
from .actions import START,OUT


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=128);a=ap.parse_args()
    with ProcessPoolExecutor(20,initializer=matched_setup) as pool:
        rows=list(pool.map(episode,[('control',s,'graph',.12,True) for s in range(START['control'],START['control']+a.count)]))
    with (OUT/'previous.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
