"""Reproduce paper Fig. 5 (E3): Fourier-coefficient clouds per ansatz and depth.

Usage, from the paper directory::

    python utils/plot_fig5.py --run outdir/<e3 run>

Rows follow the paper (l = 0 control, Circuit A l = 1 and 5, Circuit B l = 1
and 5); columns are ``c_0 .. c_5``. Each panel title gives
``Var(Re) + Var(Im)``, or "0" when the coefficient vanishes in every draw.
Saves ``fig5_reproduction.png`` to the run directory and ``results/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _figures import load_completed_metrics, save_to_run_and_results, stamp_scope

ROW_ORDER = (
    ("none_l0", "l = 0 (no W)"),
    ("strongly_l1", "Circuit A, l = 1"),
    ("strongly_l5", "Circuit A, l = 5"),
    ("basic_l1", "Circuit B, l = 1"),
    ("basic_l5", "Circuit B, l = 5"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    metrics = load_completed_metrics(args.run, "coefficients")
    samples = np.load(args.run / metrics["artifacts"]["coefficients"])
    statistics = {
        setting["label"]: setting["per_frequency"] for setting in metrics["settings"]
    }
    n_columns = metrics["max_frequency"] + 1

    figure, axes = plt.subplots(
        len(ROW_ORDER),
        n_columns,
        figsize=(2.0 * n_columns, 2.1 * len(ROW_ORDER)),
        squeeze=False,
    )
    for row, (label, title) in enumerate(ROW_ORDER):
        for column in range(n_columns):
            axis = axes[row, column]
            values = samples[label][:, column]
            axis.scatter(
                values.real,
                values.imag,
                s=9,
                facecolor="white",
                edgecolor="tab:purple" if "strongly" in label else "teal",
            )
            entry = statistics[label][column]
            variance = (
                "0"
                if entry["fraction_zero"] == 1.0
                else f"var {entry['var_total']:.1e}"
            )
            axis.set_title(f"$c_{column}$: {variance}", fontsize=7)
            axis.set_xlim(-1, 1)
            axis.set_ylim(-1, 1)
            axis.set_aspect("equal")
            axis.tick_params(labelsize=5)
            if column == 0:
                axis.set_ylabel(title, fontsize=8)
    stamp_scope(
        figure,
        f"paper-accurate: reference settings, {metrics['n_samples']} draws per row, {metrics['n_qubits']} qubits",
        str(args.run),
    )
    figure.tight_layout(rect=(0, 0.02, 1, 1))
    save_to_run_and_results(figure, args.run, "fig5_reproduction.png")


if __name__ == "__main__":
    main()
