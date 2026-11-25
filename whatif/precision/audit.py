"""Record native P9 physical settings, separately from any planner input."""

import csv
from concurrent.futures import ProcessPoolExecutor
from .collect import OUT


def inspect(job):
    from whatif.shared.sim import Scene

    group, seed = job
    env = Scene("causalworld", seed, "p9")
    stage = env.env.scene.stage
    vals = stage.get_current_variable_values_for_arena()
    row = group, seed, env.env.mass, float(vals["floor_friction"])
    env.close()
    return row


if __name__ == "__main__":
    jobs = [("prediction", i) for i in range(637000, 637128)] + [
        ("control", i) for i in range(657000, 657080)
    ]
    with ProcessPoolExecutor(5) as pool:
        rows = list(pool.map(inspect, jobs))
    with (OUT / "p9-conditions.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["split", "seed", "mass_kg", "floor_friction"])
        w.writerows(rows)
