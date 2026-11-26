"""Frozen-budget reference controllers on the same Stage 26 layouts."""
import json
import os
import multiprocessing as mp
from whatif.paired.control import batch
from .native import CONTROL,DOMAINS,OUT

if __name__=='__main__':
    base,n=CONTROL
    jobs=[(dom,arm,s,min(8,base+n-s),os.getenv('SIM_PY','/tmp/whatif-push-venv/bin/python'))
          for s in range(base,base+n,8) for dom in DOMAINS for arm in ('pets','stage21')]
    with mp.get_context('fork').Pool(24) as pool:
        rows=[r for group in pool.imap_unordered(batch,jobs) for r in group]
    (OUT/'references.json').write_text(json.dumps(sorted(rows,key=lambda r:(r['domain'],r['policy'],r['seed']))))
