"""Per-seed test AUC of the Task 1 models next to the paper's values.

One panel per sweep: every seed's fold-5 test AUC for the leak-free QNN, ANN,
parameter-matched ANN and the floors, the seed mean as a short bar, and the
paper's QNN/ANN values as hollow diamonds. Reads only ``sweep_status.json`` and
each run's ``metrics.json``. Saves ``results/<output>.png`` and a copy in every
sweep directory.

Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/plot_directional_auc.py \\
        --output e1_e3_test_auc \\
        --sweep "KCHOL 3-D, lagged (paper-accurate, 5 seeds)" <S1 dir> 0.7968 0.7475 \\
        --sweep "AAPL 64-D (paper-accurate, 5 seeds)" <S2 dir> 0.635 0.597
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Reference palette slots 1-3 (dataviz skill), with distinct marker shapes so
# identity never relies on color alone.
MODELS = [
    ("leak_free:qnn", "QNN", "#2a78d6", "o"),
    ("leak_free:ann", "ANN", "#eb6834", "s"),
    ("leak_free:ann_parameter_matched", "ANN\n(param-matched)", "#1baf7a", "^"),
    ("logistic_regression", "Logistic\nregression", "#52514e", "D"),
    ("previous_day_direction", "Previous-day\ndirection", "#52514e", "v"),
]
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_test_auc(sweep_dir: Path) -> dict[str, list[float]]:
    """Test AUC per model across the completed seeds of one sweep."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    values: dict[str, list[float]] = {key: [] for key, *_ in MODELS}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        test = json.loads((sweep_dir / run["metrics_path"]).read_text())["test"]
        for key in values:
            values[key].append(float(test[key]["auc"]))
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sweep",
        nargs=4,
        action="append",
        required=True,
        metavar=("LABEL", "SWEEP_DIR", "PAPER_QNN_AUC", "PAPER_ANN_AUC"),
    )
    parser.add_argument("--output", required=True, help="File stem for the PNG")
    args = parser.parse_args(argv)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
        }
    )
    figure, axes = plt.subplots(
        1,
        len(args.sweep),
        figsize=(5.2 * len(args.sweep), 4.6),
        sharey=True,
        facecolor=SURFACE,
    )
    axes = np.atleast_1d(axes)
    generator = np.random.default_rng(0)
    for axis, (label, sweep_dir, paper_qnn, paper_ann) in zip(axes, args.sweep):
        values = load_test_auc(Path(sweep_dir))
        axis.set_facecolor(SURFACE)
        axis.axhline(
            0.5, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=1
        )
        axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for position, (key, _, color, marker) in enumerate(MODELS):
            seeds = np.asarray(values[key])
            jitter = generator.uniform(-0.12, 0.12, size=len(seeds))
            axis.scatter(
                position + jitter,
                seeds,
                s=46,
                color=color,
                marker=marker,
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
            )
            axis.hlines(
                seeds.mean(),
                position - 0.28,
                position + 0.28,
                color=TEXT_PRIMARY,
                linewidth=2,
                zorder=4,
            )
        for position, paper_value in ((0, float(paper_qnn)), (1, float(paper_ann))):
            axis.scatter(
                position + 0.36,
                paper_value,
                s=70,
                marker="D",
                facecolor="none",
                edgecolor=MODELS[position][2],
                linewidth=2,
                zorder=5,
            )
            axis.annotate(
                f"paper {paper_value:.3f}",
                (position + 0.36, paper_value),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
        axis.set_xticks(
            range(len(MODELS)), [name for _, name, *_ in MODELS], fontsize=8.5
        )
        axis.set_title(label, fontsize=10.5, color=TEXT_PRIMARY, loc="left")
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].set_ylabel("Fold-5 test AUC (2020-2021)")
    axes[0].set_ylim(0.38, 1.0)
    figure.text(
        0.01,
        0.01,
        "Dots: one seed each. Bar: seed mean. Hollow diamond: paper value. "
        "Dashed line: chance (AUC 0.5). Models selected on validation folds 1-4.",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    targets = [
        results_dir / f"{args.output}.png",
        *[Path(sweep_dir) / f"{args.output}.png" for _, sweep_dir, *_ in args.sweep],
    ]
    for target in targets:
        figure.savefig(target, dpi=160, facecolor=SURFACE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
