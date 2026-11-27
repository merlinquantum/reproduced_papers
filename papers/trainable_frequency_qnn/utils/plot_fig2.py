#!/usr/bin/env python3
"""Plot the Fig. 2 analogue (fit + learned spectrum) from a toy sweep.

Example (from the paper folder)::

    python utils/plot_fig2.py --sweep-dir outdir/sweeps/e2_fig2b --candidates ff tf ffs \\
        --name fig2b_fit_spectrum --scope "paper-accurate settings, 10 seeds, median seed shown"
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
    plot_fit_and_spectrum,
    read_metrics,
    save_figure,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--scope", required=True, help="Result-scope label printed on the figure."
    )
    args = parser.parse_args(argv)
    sweep_dir = (
        args.sweep_dir if args.sweep_dir.is_absolute() else PAPER_DIR / args.sweep_dir
    )
    runs = completed_runs(sweep_dir)
    first_run = next(iter(runs[args.candidates[0]].values()))
    frequencies = read_metrics(first_run)["data_frequencies"]
    figure = plot_fit_and_spectrum(runs, args.candidates, frequencies, args.scope)
    for path in save_figure(
        figure, args.name, [sweep_dir / "figures", PAPER_DIR / "results"]
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
