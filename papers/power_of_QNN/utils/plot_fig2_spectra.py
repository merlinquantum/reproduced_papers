"""Fig. 2: Fisher eigenvalue histograms.

Usage: python utils/plot_fig2_spectra.py --run-dir outdir/run_<stamp> --scope paper-accurate --name <stem>
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
        plotting.plot_spectra(args.run_dir, metrics, 5, path, args.scope)


if __name__ == "__main__":
    main()
