"""Replay the fixed Stage 21 and PETS controllers on the registered control seeds."""
import json
import multiprocessing as mp
import os
from whatif.paired.control import batch
from .native import CONTROL,DOMAINS,OUT

if __name__=='__main__':
    base,n=CONTROL
    jobs=[(dom,arm,base+i,8,os.getenv('SIM_PY','/tmp/whatif-push-venv/bin/python'))
          for i in range(0,n,8) for dom in DOMAINS for arm in ('stage21','pets')]
    with mp.get_context('fork').Pool(24) as pool:
        rows=[r for group in pool.imap_unordered(batch,jobs) for r in group]
    (OUT/'references.json').write_text(json.dumps(sorted(rows,key=lambda r:(r['domain'],r['policy'],r['seed']))))
