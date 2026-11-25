"""Persistent native instances, one per paired layout in a worker shard."""
import json
import os
import sys


def main():
    output=os.fdopen(os.dup(1),'w',buffering=1)
    os.dup2(2,1)
    from .sim import Scene
    scenes={}
    try:
        for line in sys.stdin:
            job=json.loads(line)
            if job['op']=='quit':break
            responses=[]
            if job['op']=='reset':
                for env in scenes.values():env.close()
                scenes.clear()
                for seed in job['seeds']:
                    e=Scene(job['task'],seed,job['split'])
                    scenes[seed]=e
                    responses.append([seed,e.state().tolist(),e.env.protocol])
            else:
                for seed,i,budget in job['moves']:
                    z,g,n,peak=scenes[seed].push(i,budget,stop=True)
                    row=[seed,z.tolist(),int(n),float(peak)]
                    if job.get('measure'):
                        v=scenes[seed].latest_free
                        row.extend([len(v),sum(v)/len(v) if v else 0.])
                    responses.append(row)
            output.write(json.dumps(responses)+'\n')
    finally:
        for env in scenes.values():env.close()


if __name__=='__main__': main()
