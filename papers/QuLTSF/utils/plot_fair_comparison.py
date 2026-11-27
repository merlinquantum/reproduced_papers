"""Fair-comparison figure (Phase 5): every local system against QuLTSF and the paper values.

Usage (from the paper folder)::

    python utils/plot_fair_comparison.py --comparison outdir/fair_comparison_<stamp>/fair_comparison.json

Saves ``fair_comparison_mse.png`` next to the comparison file and in ``results/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

from papers.QuLTSF.lib.plotting import plot_fair_comparison, save_figure  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args(argv)
    comparison = json.loads(args.comparison.read_text())
    figure = plot_fair_comparison(comparison)
    save_figure(figure, "fair_comparison_mse", args.comparison.parent, args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
