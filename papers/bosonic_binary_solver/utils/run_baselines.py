"""Run the classical baselines at the solver's Appendix B budget and aggregate them.

The baselines in :mod:`lib.baselines` take the candidate budget as an argument, so
the only comparison that tests the paper's claim is one where every method may
evaluate the objective the same number of times. This script runs them on the same
instance seeds as the BBS runs (``instance_seed_base`` 0, ``instances`` 100 in the
configs), so every row is paired with the corresponding BBS row.

    python utils/run_baselines.py --sizes 20 30 --instances 100

Per-instance rows are not committed, as for the BBS runs; ``results/baselines.json``
carries the aggregates and is what README.md quotes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.baselines import BASELINES  # noqa: E402
from lib.problems import (  # noqa: E402
    knapsack_cost_batch,
    knapsack_instance,
    knapsack_optimum,
)
from lib.tbi import candidate_budget  # noqa: E402

DELAYS = (1, 3, 9)
UPDATES, SAMPLES = 200, 50
SEED_OFFSET = 1000  # baseline RNG stream, kept clear of the instance seeds


def run_one(job):
    """Run one (method, size, instance) triple and return its row."""
    method, m, seed = job
    instance = knapsack_instance(m, seed)
    optimum = knapsack_optimum(instance)
    budget = candidate_budget(m, DELAYS, UPDATES, SAMPLES)

    def cost_batch(bits):
        return knapsack_cost_batch(instance, bits)

    started = time.time()
    result = BASELINES[method](
        cost_batch, m, budget, np.random.default_rng(SEED_OFFSET + seed)
    )
    best = result["best_cost"]
    return {
        "method": method,
        "m": m,
        "instance_seed": seed,
        "budget": budget,
        "evaluations": result["evaluations"],
        "found_optimum": bool(abs(best - optimum) < 1e-9),
        "relative_error_percent": 0.0
        if optimum == 0
        else abs(best - optimum) / abs(optimum) * 100,
        "seconds": round(time.time() - started, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default="results/baselines.json")
    args = parser.parse_args()

    jobs = [
        (method, m, seed)
        for m in args.sizes
        for method in BASELINES
        for seed in range(args.instances)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, row in enumerate(pool.map(run_one, jobs, chunksize=1), start=1):
            rows.append(row)
            if done % 25 == 0:
                print(f"{done}/{len(jobs)}", flush=True)

    arms = []
    for m in args.sizes:
        for method in BASELINES:
            members = [r for r in rows if r["method"] == method and r["m"] == m]
            errors = [r["relative_error_percent"] for r in members]
            arms.append(
                {
                    "method": method,
                    "family": "knapsack",
                    "m": m,
                    "instances": len(members),
                    "budget": members[0]["budget"],
                    "percent_optimal": round(
                        100 * sum(r["found_optimum"] for r in members) / len(members), 2
                    ),
                    "mean_percent_error": round(st.mean(errors), 5),
                    "std_percent_error": round(st.stdev(errors), 5)
                    if len(members) > 1
                    else 0.0,
                    "median_seconds_per_instance": round(
                        st.median(r["seconds"] for r in members), 3
                    ),
                }
            )

    summary = {
        "paper": "arXiv:2510.08274",
        "note": (
            "Classical baselines at the solver's Appendix B candidate budget, on the same "
            "instance seeds as the BBS runs in summary.json. Per-instance rows are not "
            "committed, as for the BBS runs."
        ),
        "generated_at": datetime.date.today().isoformat(),
        "arms": arms,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} runs -> {len(arms)} arms -> {args.out}")


if __name__ == "__main__":
    main()
