#!/usr/bin/env python3
"""Strip plot of a per-seed toy metric for every candidate of a sweep.

Example (from the paper folder)::

    python utils/plot_toy_metrics.py --sweep-dir outdir/sweeps/e3_ff_depth --metric train_mse \\
        --name e3_ff_depth_mse --title "FF depth ablation on {1, 1.2, 3}" --scope "3 seeds"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
for import_root in (PAPER_DIR.parents[1], PAPER_DIR):
    sys.path.insert(0, str(import_root))

from lib.plotting import (  # noqa: E402
    completed_runs,
    plot_metric_by_candidate,
    read_metrics,
    save_figure,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--metric", default="train_mse")
    parser.add_argument(
        "--candidates", nargs="*", help="Subset and order of candidates (default: all)."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--scope", required=True, help="Result-scope label printed on the figure."
    )
    args = parser.parse_args(argv)
    sweep_dir = (
        args.sweep_dir if args.sweep_dir.is_absolute() else PAPER_DIR / args.sweep_dir
    )
    runs = completed_runs(sweep_dir)
    order = args.candidates or list(runs)
    values = {
        c: [read_metrics(path)[args.metric] for _, path in sorted(runs[c].items())]
        for c in order
    }
    figure = plot_metric_by_candidate(values, args.metric, args.title, args.scope)
    for path in save_figure(
        figure, args.name, [sweep_dir / "figures", PAPER_DIR / "results"]
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
