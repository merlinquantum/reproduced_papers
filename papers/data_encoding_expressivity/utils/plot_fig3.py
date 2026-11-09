"""Reproduce paper Fig. 3 (E1) from a fit run, with the B1 floor and all seeds.

Usage, from the paper directory::

    python utils/plot_fig3.py --run outdir/<e1 quantum run> [--classical-run outdir/<e1 B2 run>]

Left column: target samples, every seed's final fit (thin) and the B1 oracle
(dashed). Right column: full-grid MSE per step for every seed, with the B1
floor; B2 curves are added in green when ``--classical-run`` is given. Saves
``fig3_reproduction.png`` to the quantum run directory and ``results/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _figures import load_completed_metrics, save_to_run_and_results, stamp_scope

ROW_TITLES = {
    "fig3_row_a_degree1": "(a) degree-1 target, encoding RX(x)",
    "fig3_row_b_mismatch": "(b) degree-1 target, encoding RX(2x)",
    "fig3_row_c_degree2": "(c) degree-2 target, encoding RX(x)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--classical-run", type=Path, default=None)
    args = parser.parse_args()

    metrics = load_completed_metrics(args.run, "fit")
    predictions = np.load(args.run / metrics["artifacts"]["predictions"])
    inputs = predictions["inputs"]
    classical = (
        load_completed_metrics(args.classical_run, "fit")
        if args.classical_run
        else None
    )

    cases = metrics["cases"]
    figure, axes = plt.subplots(
        len(cases), 2, figsize=(10, 3.0 * len(cases)), squeeze=False
    )
    for row, case in enumerate(cases):
        fit_axis, loss_axis = axes[row]
        name = case["name"]
        fit_axis.scatter(
            inputs,
            predictions[f"target/{name}"],
            s=14,
            facecolor="white",
            edgecolor="black",
            zorder=3,
            label="target samples",
        )
        fit_axis.plot(
            inputs,
            predictions[f"b1/{name}"],
            color="tab:red",
            ls="--",
            lw=1.3,
            label="B1 oracle",
        )
        for index, seed in enumerate(metrics["seeds"]):
            fit_axis.plot(
                inputs,
                predictions[f"model/{name}/quantum/{seed}"],
                color="tab:blue",
                lw=0.9,
                alpha=0.55,
                label="quantum, all seeds" if index == 0 else None,
            )
        fit_axis.set_ylim(-1, 1)
        fit_axis.set_title(ROW_TITLES.get(name, name), fontsize=9)
        fit_axis.set_xlabel("x")
        fit_axis.set_ylabel("f(x)")
        fit_axis.legend(fontsize=6, loc="upper right")

        for run in metrics["runs"]:
            if run["case"] == name:
                loss_axis.semilogy(
                    run["mse_history"], color="tab:blue", lw=0.8, alpha=0.6
                )
        if classical is not None:
            for run in classical["runs"]:
                if run["case"] == name:
                    loss_axis.semilogy(
                        run["mse_history"], color="tab:green", lw=0.8, alpha=0.6
                    )
        loss_axis.axhline(max(case["b1_mse"], 1e-32), color="tab:red", ls="--", lw=1.3)
        loss_axis.set_ylim(1e-12, 2)
        loss_axis.set_xlabel("optimizer step")
        loss_axis.set_ylabel("full-grid MSE")
        loss_axis.set_title("blue: quantum; green: B2; dashed: B1 floor", fontsize=8)

    sources = str(args.run) + (f", {args.classical_run}" if args.classical_run else "")
    stamp_scope(
        figure,
        f"paper-accurate: paper protocol, {len(metrics['seeds'])} seeds",
        sources,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 1))
    save_to_run_and_results(figure, args.run, "fig3_reproduction.png")


if __name__ == "__main__":
    main()
