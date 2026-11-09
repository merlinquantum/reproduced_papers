#!/usr/bin/env python3
"""Plot the Fig. 4 / Fig. 5 analogues and MAERM-vs-time from a Navier-Stokes sweep.

The pressure panels show, per candidate, the seed with the median mean pressure
MAERM; the MAERM panels show the mean over seeds with a min-max band.

Example (from the paper folder)::

    python utils/plot_ns.py --sweep-dir outdir/sweeps/e6_table1 --candidates tf ff --time 3.5 \\
        --name e6_table1 --scope "paper settings, 10 paired seeds"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
for import_root in (PAPER_DIR.parents[1], PAPER_DIR):
    sys.path.insert(0, str(import_root))

from lib.data import N_X, N_Y  # noqa: E402
from lib.plotting import (  # noqa: E402
    completed_runs,
    plot_maerm_over_time,
    plot_pressure_fields,
    read_metrics,
    save_figure,
)


def _median_pressure_seed(by_seed: dict[int, Path]) -> int:
    values = sorted(
        (read_metrics(p)["maerm_table"]["p_maerm"]["mean"], s)
        for s, p in by_seed.items()
    )
    return values[(len(values) - 1) // 2][1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument(
        "--extra-sweep-dir",
        action="append",
        default=[],
        type=Path,
        help="Additional sweep folder whose runs are merged (repeatable).",
    )
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument(
        "--time",
        type=float,
        required=True,
        help="Snapshot time for the pressure panels.",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--scope", required=True, help="Result-scope label printed on the figure."
    )
    args = parser.parse_args(argv)
    sweep_dir = (
        args.sweep_dir if args.sweep_dir.is_absolute() else PAPER_DIR / args.sweep_dir
    )
    runs = completed_runs(sweep_dir)
    for extra in args.extra_sweep_dir:
        extra_dir = extra if extra.is_absolute() else PAPER_DIR / extra
        for candidate, by_seed in completed_runs(extra_dir).items():
            merged = runs.setdefault(candidate, {})
            duplicates = set(merged) & set(by_seed)
            if duplicates:
                raise ValueError(
                    f"{candidate}: seeds {sorted(duplicates)} completed twice."
                )
            merged.update(by_seed)
    destinations = [sweep_dir / "figures", PAPER_DIR / "results"]

    # Grid coordinates in X_star order (x fastest), as in lib/data.py.
    x = np.tile(np.linspace(1, 8, N_X), N_Y)
    y = np.repeat(np.linspace(-2, 2, N_Y), N_X)
    reference = None
    predictions = {}
    for candidate in args.candidates:
        seed = _median_pressure_seed(runs[candidate])
        archive = np.load(runs[candidate][seed] / "predictions.npz")
        time_index = int(np.argmin(np.abs(archive["times"] - args.time)))
        if not np.isclose(archive["times"][time_index], args.time):
            raise ValueError(
                f"t = {args.time} is not an evaluation time of {candidate}."
            )
        pressure_index = list(archive["fields"]).index("p")
        reference = archive["reference"][time_index, :, pressure_index]
        predictions[f"{candidate.upper()} (seed {seed})"] = archive["prediction"][
            time_index, :, pressure_index
        ]
    figure = plot_pressure_fields(reference, predictions, x, y, args.time, args.scope)
    for path in save_figure(figure, f"{args.name}_pressure_t{args.time}", destinations):
        print(path)

    per_candidate = {
        c: {s: read_metrics(p)["maerm_per_time"] for s, p in sorted(runs[c].items())}
        for c in args.candidates
    }
    figure = plot_maerm_over_time(per_candidate, args.scope)
    for path in save_figure(figure, f"{args.name}_maerm_over_time", destinations):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
