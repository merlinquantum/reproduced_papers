"""Plot per-configuration accuracy of our Rice runs against the released runs.

Usage (from the repository root)::

    python papers/QNN_hyperparameters/utils/plot_grid_agreement.py \
        papers/QNN_hyperparameters/outdir/sweeps/E-C_rice_grid \
        papers/QNN_hyperparameters/outdir/sweeps/E-C2_rice_grid_author_signs \
        --labels "installed PCA signs" "author PCA signs" --init beta \
        --name E-C_vs_E-C2_minmax

Every sweep is read from its runs' ``metrics.json`` (``load_sweep``) and
paired with the released run of the same grid index. The figure is saved into
each sweep directory and into ``papers/QNN_hyperparameters/results/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

from QNN_hyperparameters.lib.reanalysis import load_released_results  # noqa: E402
from QNN_hyperparameters.lib.sweep_analysis import (  # noqa: E402
    grid_as_released,
    load_sweep,
)
from QNN_hyperparameters.utils.analyze_sweeps import RELEASED_RESULTS  # noqa: E402

# Categorical slots 1 and 2 of the dataviz reference palette, in fixed order.
SERIES_COLORS = ("#2a78d6", "#eb6834")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
OPTIMIZER_TITLES = {"cobyla": "COBYLA", "spsa": "SPSA", "neldermead": "Nelder-Mead"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--init", choices=["uniform", "beta"], required=True)
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--name", required=True)
    parser.add_argument("--scope", default="V2 reduced: Rice, 1 seed per configuration")
    args = parser.parse_args()
    if len(args.labels) != len(args.sweep_dirs):
        parser.error("one label per sweep directory")
    if len(args.sweep_dirs) > len(SERIES_COLORS):
        parser.error(f"at most {len(SERIES_COLORS)} sweeps per figure")

    released = load_released_results(RELEASED_RESULTS)
    reference = released[
        (released["dataset"] == "rice")
        & (released["noisy"] == args.noisy)
        & (released["init"] == args.init)
    ].set_index(["optimizer", "index"])["accuracy"]

    series = []
    for sweep_dir in args.sweep_dirs:
        frame, coverage = load_sweep(sweep_dir)
        if not coverage["complete"]:
            raise SystemExit(f"PARTIAL SWEEP {sweep_dir}: {coverage}")
        ours = grid_as_released(frame, released, noisy=args.noisy)
        ours = ours[ours["init"] == args.init].set_index(["optimizer", "index"])
        series.append(
            ours["accuracy"].to_frame("ours").join(reference.rename("released"))
        )

    optimizers = [
        name
        for name in OPTIMIZER_TITLES
        if any(name in frame.index.get_level_values(0) for frame in series)
    ]
    figure, axes = plt.subplots(
        1, len(optimizers), figsize=(4.2 * len(optimizers), 4.4), squeeze=False
    )
    figure.patch.set_facecolor(SURFACE)
    for axis, optimizer in zip(axes[0], optimizers):
        axis.set_facecolor(SURFACE)
        axis.plot(
            [0.25, 1.0], [0.25, 1.0], color=TEXT_SECONDARY, lw=1, ls="--", zorder=1
        )
        for paired, label, color in zip(series, args.labels, SERIES_COLORS):
            if optimizer not in paired.index.get_level_values(0):
                continue
            points = paired.loc[optimizer].dropna()
            axis.scatter(
                points["released"],
                points["ours"],
                s=36,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.5,
                alpha=0.85,
                label=f"{label} (mean diff {(points['ours'] - points['released']).mean():+.3f})",
                zorder=2,
            )
        axis.set_title(OPTIMIZER_TITLES[optimizer], color=TEXT_PRIMARY, fontsize=11)
        axis.set_xlim(0.25, 1.0)
        axis.set_ylim(0.25, 1.0)
        axis.set_aspect("equal")
        axis.set_xlabel("released validation accuracy", color=TEXT_SECONDARY)
        axis.grid(color="#e6e5e0", lw=0.6)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#d4d3cd")
        axis.legend(
            fontsize=7.5, frameon=False, loc="upper left", labelcolor=TEXT_PRIMARY
        )
    axes[0][0].set_ylabel("our validation accuracy", color=TEXT_SECONDARY)
    arm = (
        "MinMax inputs (author 'beta' arm)"
        if args.init == "beta"
        else "raw inputs (author 'uniform' arm)"
    )
    noise = "FakePerth noise" if args.noisy else "noiseless"
    figure.suptitle(
        f"Rice PCA, {arm}, {noise}: ours vs released, one point per circuit\n"
        f"Scope: {args.scope}. Dashed line: equal accuracy.",
        color=TEXT_PRIMARY,
        fontsize=10,
    )
    figure.tight_layout()

    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    for directory in (*args.sweep_dirs, results_dir):
        figure.savefig(
            directory / f"{args.name}.png",
            dpi=150,
            facecolor=SURFACE,
            bbox_inches="tight",
        )
    print(f"saved {args.name}.png to {len(args.sweep_dirs) + 1} directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
