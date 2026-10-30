"""QuaRK's normalized test MSE against the number of training windows, over all seeds.

Multi-seed version of the paper's Fig. 4 for the model at the paper settings, with the
ridge chosen on validation data. Thin lines are seeds, thick lines their mean; the gray
line has the slope -0.5 of the paper's "1/sqrt(N) shape", drawn through the mean of the
first task at the smallest N of the slope fit. The labels give the mean slope fitted
over that range (the frozen decay metric).

Usage (from the repository root)::

    python papers/QuaRK/utils/plot_training_size_seeds.py \\
        papers/QuaRK/outdir/sweep_main_comparison --candidate quark --label "<scope>"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import (
    PAPER_TASKS,
    TASK_COLORS,
    TASK_LABELS,
    TEXT_SECONDARY,
    apply_style,
    load_sweep,
    save,
    selected,
    spread_log_labels,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    apply_style()
    runs = [r for r in load_sweep(args.sweep_dir) if r["candidate"] == args.candidate]
    if not runs:
        raise SystemExit(f"no run of candidate {args.candidate!r}")
    snapshot = json.loads((runs[0]["run_dir"] / "config_snapshot.json").read_text())
    slope_min = int(snapshot["readout"]["slope_min_size"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    anchor, ends, texts = None, {}, {}
    for task in PAPER_TASKS:
        sizes = np.array(selected(runs[0]["metrics"], task, "training_sizes"))
        curves = []
        for run in runs:
            if selected(run["metrics"], task, "training_sizes") != sizes.tolist():
                raise SystemExit("runs use different training-size grids")
            test_mse = np.array(
                selected(run["metrics"], task, "training_size_mse_test")
            )
            curves.append(
                test_mse / run["metrics"]["tasks"][task]["label_variance_test"]
            )
        curves = np.array(curves)
        color = TASK_COLORS[task]
        for curve in curves:
            ax.plot(sizes, curve, color=color, lw=0.8, alpha=0.35)
        mean = curves.mean(axis=0)
        ax.plot(sizes, mean, color=color, marker="o", ms=4)
        slope = np.mean([selected(r["metrics"], task, "decay_slope") for r in runs])
        ends[task] = mean[-1]
        texts[task] = f"{TASK_LABELS[task].split(' (')[0]}: slope {slope:.2f}"
        if anchor is None:
            first = int(np.flatnonzero(sizes >= slope_min)[0])
            anchor = (sizes[first], mean[first])
    reference = np.array([anchor[0], sizes[-1]])
    reference_y = anchor[1] * (reference / anchor[0]) ** -0.5
    ax.plot(reference, reference_y, color=TEXT_SECONDARY, ls="--", lw=1.2)
    ends["reference"] = reference_y[-1]
    texts["reference"] = "slope -0.5 (1/√N)"
    for key, y in spread_log_labels(ends, min_ratio=1.18).items():
        ax.annotate(
            texts[key],
            (sizes[-1], ends[key]),
            xytext=(sizes[-1] * 1.15, y),
            textcoords="data",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(sizes[0] / 1.3, sizes[-1] * 6)
    ax.set_xlabel("Number of training windows N (log scale)")
    ax.set_ylabel("Normalized test MSE (log scale)")
    ax.set_title(
        "QuaRK test error against N (paper Fig. 4), ridge chosen on validation data; "
        f"thin lines are seeds\n{args.label}",
        fontsize=9.5,
        loc="left",
    )
    fig.tight_layout()
    saved = save(fig, args.sweep_dir, "training_size_seeds")
    print("\n".join(str(p) for p in saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
