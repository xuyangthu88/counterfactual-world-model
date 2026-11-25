"""Replay the frozen published-in-repository controllers under the new reset seeds."""
import argparse
import json
import multiprocessing as mp
import os
from whatif.paired.control import batch
from .native import DOMAINS,START,OUT

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--anchor',action='store_true')
    anchor=parser.parse_args().anchor
    from .anchor_native import BASE,COUNT
    base,n=(BASE,COUNT) if anchor else START['control']
    jobs=[(d,p,base+i,8,os.environ.get('SIM_PY','/tmp/whatif-push-venv/bin/python'))
          for i in range(0,n,8) for d in DOMAINS for p in ('stage21','pets')]
    with mp.get_context('fork').Pool(24) as pool:
        rows=[r for chunk in pool.imap_unordered(batch,jobs) for r in chunk]
    (OUT/('anchor-references.json' if anchor else 'references.json')).write_text(json.dumps(sorted(rows,key=lambda s:(s['domain'],s['policy'],s['seed']))))
