"""Fresh, paired CausalWorld scenes for single-seed prediction and control."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .chain_data import ACTIONS, TICKS, collect_layout


# start, layouts, number of blocks, first push index, angle, interpolated physics
SETS = {
    'train3_41': (41000, 160, 3, 5, 0., False),
    'train3_54': (42000, 160, 3, 1, 0., False),
    'val3': (43000, 96, 3, 5, 0., False),
    'test2': (44000, 144, 2, 5, 0., False),
    'test3_41': (45000, 192, 3, 5, 0., False),
    'test3_32': (45000, 192, 3, 4, 0., False),
    'test3_54': (45000, 192, 3, 1, 0., False),
    'test3_69': (45000, 192, 3, 6, 0., False),
    'shift': (46000, 160, 3, 5, 2*np.pi/3, True),
}


BASE_SETS = tuple(SETS)
for group, start, count in [('fit', 50000, 192), ('val', 51000, 64), ('test', 52000, 192)]:
    for length, probe in [(32, 4), (41, 5), (54, 1), (69, 6)]:
        SETS[f'value_{group}_{length}'] = (start, count, 3, probe, 0., False)


def collect(job):
    name, seed = job
    _, _, n, probe, angle, physical = SETS[name]
    return collect_layout(seed, np.random.default_rng(seed), n, physical, angle, probe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', choices=SETS)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', type=Path, default=Path('data/pilot'))
    ap.add_argument('--value', action='store_true', help='Collect four first actions per layout for value learning')
    args = ap.parse_args()
    if args.sets is None:
        args.sets = [k for k in SETS if k.startswith('value_')] if args.value else BASE_SETS
    args.out.mkdir(parents=True, exist_ok=True)
    names = ('seed', 'count', 'states', 'edges', 'physics', 'noise', 'goal', 'tips')
    with ProcessPoolExecutor(args.workers) as pool:
        for name in args.sets:
            start, count, _, probe, angle, _ = SETS[name]
            rows = list(pool.map(collect, ((name, i) for i in range(start, start+count))))
            np.savez_compressed(args.out / (name+'.npz'), actions=ACTIONS, ticks=TICKS, probe=probe,
                                frame=angle, **dict(zip(names, map(np.asarray, zip(*rows)))))
            print(name, count, flush=True)


if __name__ == '__main__':
    main()
