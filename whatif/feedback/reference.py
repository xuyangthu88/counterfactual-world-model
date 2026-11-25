"""Frozen Stage 21 and PETS, evaluated on exactly the feedback control layouts."""
import json
import multiprocessing as mp
import os
from .native import DOMAINS, OUT, SPLITS
from whatif.paired.control import batch

if __name__ == '__main__':
    start, count = SPLITS['control']
    sim = os.environ.get('SIM_PY', '/tmp/whatif-push-venv/bin/python')
    jobs = [(d, p, start + i, 8, sim)
            for i in range(0, count, 8) for d in DOMAINS for p in ('stage21','pets')]
    with mp.get_context('fork').Pool(24) as workers:
        rows = [r for group in workers.imap_unordered(batch, jobs) for r in group]
    (OUT / 'references.json').write_text(json.dumps(sorted(rows, key=lambda r:(r['domain'],r['policy'],r['seed']))))
