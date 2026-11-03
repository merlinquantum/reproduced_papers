"""Fig. 3a: normalized effective dimension vs n, with the paper's published curves.

Usage: python utils/plot_fig3a_effective_dimension.py --run-dir outdir/run_<stamp> --scope paper-accurate --name <stem>
Writes <run-dir>/<stem>.png and results/<stem>.png.
"""

from __future__ import annotations

from figure_io import (  # noqa: F401
    PAPER_TABLE1_LOSS,
    load_metrics,
    outputs,
    paper_fig3a_curves,
    paper_fig6_points,
    parse_arguments,
)
from power_of_QNN.lib import plotting


def main() -> None:
    args = parse_arguments(__doc__.splitlines()[0])
    metrics = load_metrics(args.run_dir)
    for path in outputs(args.run_dir, args.name):
        plotting.plot_effective_dimension(
            metrics, path, args.scope, paper_fig3a_curves()
        )


if __name__ == "__main__":
    main()
