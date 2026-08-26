"""Print a run's metrics as the table layout used in the paper.

Usage
-----
    python utils/summarize.py outdir/run_YYYYMMDD-HHMMSS [more_run_dirs...]

Accepts several run directories so that studies launched separately (for
example one target per job) can be aggregated into a single table.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

LATENT_ORDER = ("gaussian", "bernoulli", "distinguishable", "boson")
LABELS = {
    "shuffled_boson": "Shuffled boson",
    "copula_boson": "Copula boson",
    "negative_binomial": "Neg. binomial",
    "gaussian": "Gaussian",
    "bernoulli": "Bernoulli",
    "distinguishable": "Dist. sampler",
    "boson": "Boson sampler",
}


def _mean_sem(values: list[float]) -> str:
    import statistics

    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    sem = statistics.stdev(values) / len(values) ** 0.5
    return f"{statistics.fmean(values):.3f} ± {sem:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()

    records = []
    for run_dir in args.run_dirs:
        records.extend(json.loads((run_dir / "metrics.json").read_text()))

    if "target" in records[0]:  # synthetic datasets, paper Table I
        grouped = defaultdict(list)
        for record in records:
            grouped[(record["target"], record["latent"])].append(
                record["l1_nearest_int"]
            )
        targets = sorted({key[0] for key in grouped})
        print(f"{'latent':<16}" + "".join(f"{t + ' dataset':<22}" for t in targets))
        for kind in LATENT_ORDER:
            cells = [_mean_sem(grouped.get((t, kind), [])) for t in targets]
            print(f"{LABELS.get(kind, kind):<16}" + "".join(f"{c:<22}" for c in cells))
        return

    grouped = defaultdict(lambda: defaultdict(list))  # mixture of Gaussians, Fig. 2
    for record in records:
        for metric in ("interpolation_rate", "n_modes_covered", "mmd"):
            grouped[record["latent"]][metric].append(float(record[metric]))
    print(f"{'latent':<16}{'interpolation':<20}{'modes covered':<20}{'mmd':<20}")
    for kind in LATENT_ORDER:
        row = grouped.get(kind, {})
        cells = [
            _mean_sem(row.get(metric, []))
            for metric in ("interpolation_rate", "n_modes_covered", "mmd")
        ]
        print(f"{LABELS.get(kind, kind):<16}" + "".join(f"{c:<20}" for c in cells))


if __name__ == "__main__":
    main()
