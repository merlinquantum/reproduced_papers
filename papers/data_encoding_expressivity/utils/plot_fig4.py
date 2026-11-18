"""Reproduce paper Fig. 4 (E2) and plot final MSE against the number of repetitions r.

Usage, from the paper directory::

    python utils/plot_fig4.py --fits-run outdir/<e2 quantum run> \
        --compare "quantum, 200 steps=outdir/<run>" --compare "B2, 200 steps=outdir/<run>" \
        --scope "paper-accurate: paper protocol, 6 seeds"

``fig4_reproduction<suffix>.png`` mirrors the paper layout (rows r = 1, 3, 5;
columns serial and parallel): target samples, the median-MSE seed's fit in
bold, every other seed thin, and the B1 oracle dashed.
``fig4_mse_vs_r<suffix>.png`` shows every seed's final full-grid MSE against r
for every ``--compare`` run, with B1 floors. Both are saved to the
``--fits-run`` directory and ``results/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _figures import load_completed_metrics, save_to_run_and_results, stamp_scope

LAYOUTS = ("serial", "parallel")
# float64 MSE values below this are round-off; they are drawn at this level.
ROUND_OFF_FLOOR = 1e-16
COMPARE_COLORS = (
    "tab:blue",
    "tab:green",
    "tab:purple",
    "tab:orange",
    "tab:brown",
    "tab:gray",
)


def _cases_by_layout(metrics: dict) -> dict[str, list[dict]]:
    grouped = {layout: [] for layout in LAYOUTS}
    for case in metrics["cases"]:
        grouped[case["model"]["layout"]].append(case)
    for layout in grouped:
        grouped[layout].sort(key=lambda case: case["model"]["repetitions"])
    return grouped


def plot_fits(run_dir: Path, scope: str, suffix: str) -> None:
    metrics = load_completed_metrics(run_dir, "fit")
    predictions = np.load(run_dir / metrics["artifacts"]["predictions"])
    inputs = predictions["inputs"]
    grouped = _cases_by_layout(metrics)
    n_rows = max(len(cases) for cases in grouped.values())
    figure, axes = plt.subplots(
        n_rows, 2, figsize=(10, 2.8 * n_rows), squeeze=False, sharex=True
    )
    for column, layout in enumerate(LAYOUTS):
        for row, case in enumerate(grouped[layout]):
            axis = axes[row, column]
            name = case["name"]
            case_runs = [run for run in metrics["runs"] if run["case"] == name]
            ordered = sorted(case_runs, key=lambda run: run["final_mse"])
            median_seed = ordered[(len(ordered) - 1) // 2]["seed"]
            axis.scatter(
                inputs,
                predictions[f"target/{name}"],
                s=10,
                facecolor="white",
                edgecolor="black",
                zorder=3,
            )
            axis.plot(
                inputs, predictions[f"b1/{name}"], color="tab:red", ls="--", lw=1.1
            )
            for run in case_runs:
                bold = run["seed"] == median_seed
                axis.plot(
                    inputs,
                    predictions[f"model/{name}/quantum/{run['seed']}"],
                    color="tab:blue",
                    lw=1.8 if bold else 0.6,
                    alpha=1.0 if bold else 0.35,
                    zorder=4 if bold else 2,
                )
            repetitions = case["model"]["repetitions"]
            axis.set_title(
                f"{layout}, r = {repetitions}; bold = median seed ({median_seed}), dashed = B1",
                fontsize=8,
            )
            axis.set_ylim(-1, 1)
            if column == 0:
                axis.set_ylabel("f(x)")
    for axis in axes[-1]:
        axis.set_xlabel("x")
    stamp_scope(figure, scope, str(run_dir))
    figure.tight_layout(rect=(0, 0.02, 1, 1))
    save_to_run_and_results(figure, run_dir, f"fig4_reproduction{suffix}.png")


def plot_mse_against_repetitions(
    run_dir: Path, comparisons: list[tuple[str, Path]], scope: str, suffix: str
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for column, layout in enumerate(LAYOUTS):
        axis = axes[column]
        floors_drawn = False
        for index, (label, compare_dir) in enumerate(comparisons):
            metrics = load_completed_metrics(compare_dir, "fit")
            color = COMPARE_COLORS[index % len(COMPARE_COLORS)]
            offset = (index - (len(comparisons) - 1) / 2) * 0.12
            cases = _cases_by_layout(metrics)[layout]
            medians = []
            for case in cases:
                repetitions = case["model"]["repetitions"]
                values = [
                    max(run["final_mse"], ROUND_OFF_FLOOR)
                    for run in metrics["runs"]
                    if run["case"] == case["name"]
                ]
                axis.scatter(
                    np.full(len(values), repetitions + offset),
                    values,
                    s=14,
                    color=color,
                    alpha=0.7,
                )
                medians.append((repetitions, float(np.median(values))))
            axis.plot(*zip(*medians), color=color, lw=1.2, label=label)
            if not floors_drawn:
                floor = [
                    (case["model"]["repetitions"], max(case["b1_mse"], ROUND_OFF_FLOOR))
                    for case in cases
                ]
                axis.plot(
                    *zip(*floor),
                    color="tab:red",
                    ls="--",
                    lw=1.2,
                    marker="_",
                    label="B1 floor",
                )
                floors_drawn = True
        axis.axhspan(ROUND_OFF_FLOOR / 10, ROUND_OFF_FLOOR * 3, color="0.9", zorder=0)
        axis.text(
            1,
            ROUND_OFF_FLOOR * 1.6,
            "≤ round-off (drawn at 1e-16)",
            fontsize=6,
            color="0.4",
        )
        axis.set_yscale("log")
        axis.set_ylim(ROUND_OFF_FLOOR / 10, 1)
        axis.set_xlabel("repetitions r")
        axis.set_title(
            f"{layout} encoding: final full-grid MSE per seed (line = median)",
            fontsize=9,
        )
        axis.set_xticks([1, 3, 5])
    axes[0].set_ylabel("final full-grid MSE")
    axes[1].legend(fontsize=7, loc="lower left")
    stamp_scope(figure, scope, ", ".join(str(path) for _, path in comparisons))
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save_to_run_and_results(figure, run_dir, f"fig4_mse_vs_r{suffix}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fits-run", required=True, type=Path)
    parser.add_argument(
        "--compare", action="append", default=[], help="'<label>=<run directory>'"
    )
    parser.add_argument(
        "--scope", required=True, help="Result-scope label printed on the figures"
    )
    parser.add_argument(
        "--suffix", default="", help="Filename suffix, e.g. _s2 for the caption variant"
    )
    args = parser.parse_args()
    comparisons = []
    for item in args.compare:
        label, separator, path = item.rpartition("=")
        if not separator:
            raise ValueError(f"--compare needs '<label>=<run directory>', got {item!r}")
        comparisons.append((label, Path(path)))
    plot_fits(args.fits_run, args.scope, args.suffix)
    if comparisons:
        plot_mse_against_repetitions(
            args.fits_run, comparisons, args.scope, args.suffix
        )


if __name__ == "__main__":
    main()
