"""Execute only the selected controls in the real benchmark; replay deterministic prefixes."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .tasks import ACTIONS, create


def execute(job):
    task,seed,history=job[:3]
    split=job[3] if len(job)>3 else 'control'
    env=create(task,seed,split)
    env.stop_on_success=True
    try:
        initial=env.obs()
        for a in history:
            result,_=env.run(ACTIONS[a])
        return [initial.tolist(),result.tolist(),env.steps,env.protocol]
    finally:
        env.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('jobs',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--workers',type=int,default=12)
    args=ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        rows=list(ex.map(execute,json.loads(args.jobs.read_text())))
    args.output.write_text(json.dumps(rows))
