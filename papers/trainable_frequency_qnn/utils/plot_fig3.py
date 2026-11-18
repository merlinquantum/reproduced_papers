#!/usr/bin/env python3
"""Plot the Fig. 3 analogue (spectral richness box plot) from the E5 sweep.

Example (from the paper folder)::

    python utils/plot_fig3.py --sweep-dir outdir/sweeps/e5_fig3 --models simple tower exponential tf ffs \\
        --name fig3_spectral_richness --scope "paper settings, 5 seeds per dataset"
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
for import_root in (PAPER_DIR.parents[1], PAPER_DIR):
    sys.path.insert(0, str(import_root))

from lib.plotting import (  # noqa: E402
    completed_runs,
    plot_fig3_boxes,
    read_metrics,
    save_figure,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--scope", required=True, help="Result-scope label printed on the figure."
    )
    args = parser.parse_args(argv)
    sweep_dir = (
        args.sweep_dir if args.sweep_dir.is_absolute() else PAPER_DIR / args.sweep_dir
    )
    runs = completed_runs(sweep_dir)
    per_dataset: dict[str, dict[int, float]] = {model: {} for model in args.models}
    for candidate, by_seed in runs.items():
        model, k = candidate.rsplit("_k", 1)
        if model in per_dataset:
            per_dataset[model][int(k)] = statistics.fmean(
                read_metrics(p)["train_mse"] for p in by_seed.values()
            )
    figure = plot_fig3_boxes(per_dataset, args.scope)
    for path in save_figure(
        figure, args.name, [sweep_dir / "figures", PAPER_DIR / "results"]
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
