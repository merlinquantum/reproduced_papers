"""Fig. 9: Fisher eigenvalue histograms per input size (one figure per s_in).

Usage: python utils/plot_fig9_spectrum_scaling.py --run-dir outdir/run_<stamp> --scope paper-accurate --name <stem>
Writes ``<stem>_s<S>.png`` in the run directory and in results/, grouping models by input size.
"""

from __future__ import annotations

from figure_io import load_metrics, outputs, parse_arguments
from power_of_QNN.lib import plotting


def main() -> None:
    args = parse_arguments(__doc__.splitlines()[0])
    metrics = load_metrics(args.run_dir)
    sizes = sorted({entry["input_size"] for entry in metrics["models"].values()})
    for size in sizes:
        subset = {
            **metrics,
            "models": {
                k: v for k, v in metrics["models"].items() if v["input_size"] == size
            },
        }
        for path in outputs(args.run_dir, f"{args.name}_s{size}"):
            plotting.plot_spectra(args.run_dir, subset, 5, path, args.scope)


if __name__ == "__main__":
    main()
