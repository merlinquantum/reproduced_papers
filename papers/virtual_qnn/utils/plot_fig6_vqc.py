"""Fig. 6d reproduction: virtual quantum classifier accuracy vs budget, with baselines.

Usage::

    python3 papers/virtual_qnn/utils/plot_fig6_vqc.py \
        papers/virtual_qnn/outdir/sweeps/e4_vqc_budget \
        papers/virtual_qnn/outdir/sweeps/e4_vqc_baselines --results papers/virtual_qnn/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR.parent))

from virtual_qnn.lib import paper_values as paper  # noqa: E402
from virtual_qnn.utils import plot_style as style  # noqa: E402

BASELINE_LABELS = {
    "e4_vqc_noiseless": "Noiseless QC (8 params)",
    "e4_vqc_param_matched": "Noisy QC, depth 4 (16 params)",
    "mlp": "Classical MLP (17 params)",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("budget_sweep_dir", type=Path)
    parser.add_argument("baseline_sweep_dir", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    style.apply_style()
    budget_summary = style.load_summary(args.budget_sweep_dir)
    baseline_summary = style.load_summary(args.baseline_sweep_dir)
    rows = sorted(
        budget_summary["candidates"], key=lambda c: c["grid_point"]["--vqc-budget"]
    )
    budgets = np.array([c["grid_point"]["--vqc-budget"] for c in rows])

    fig, axes = style.plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    width = 0.17
    for ax, split, paper_mean, paper_std in (
        (axes[0], "train", paper.FIG6_TRAIN_MEAN, paper.FIG6_TRAIN_STD),
        (axes[1], "test", paper.FIG6_TEST_MEAN, paper.FIG6_TEST_STD),
    ):
        ours = np.array([c["aggregates"][f"{split}_accuracy"]["mean"] for c in rows])
        ours_sd = np.array([c["aggregates"][f"{split}_accuracy"]["std"] for c in rows])
        ax.bar(
            budgets - width / 2 - 0.01,
            ours,
            width,
            yerr=ours_sd,
            color=style.SERIES[0],
            error_kw={"ecolor": style.TEXT_SECONDARY, "elinewidth": 1, "capsize": 2},
            label="Reproduction (10 paired reps)",
        )
        ax.bar(
            budgets + width / 2 + 0.01,
            paper_mean,
            width,
            yerr=paper_std,
            color="none",
            edgecolor=style.SERIES[0],
            hatch="////",
            linewidth=1.2,
            error_kw={"ecolor": style.TEXT_SECONDARY, "elinewidth": 1, "capsize": 2},
            label="Paper (reference data)",
        )
        for index, (key, label) in enumerate(BASELINE_LABELS.items()):
            candidate = next(
                c
                for c in baseline_summary["candidates"]
                if (
                    c["grid_point"].get("--classical-kind")
                    or Path(c["grid_point"]["--config"]).stem
                )
                == key
            )
            value = candidate["aggregates"][f"{split}_accuracy"]["mean"]
            ax.axhline(
                value,
                color=style.SERIES[index + 1],
                linestyle=["--", ":", "-."][index],
                linewidth=1.5,
                label=label,
            )
        ax.set_xticks(budgets)
        ax.set_xlabel("Budget γ (γ = 1: conventional classifier)")
        ax.set_title(f"{split.capitalize()} accuracy", loc="left")
        ax.set_ylim(0.5, 1.0)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=8,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.suptitle(
        "Virtual quantum classifier on the circle data set (Fig. 6d)",
        x=0.06,
        ha="left",
        fontsize=11,
    )
    fig.text(
        0.06,
        -0.22,
        "Scope: paper-accurate (reference hyperparameters), 10 repetitions; data sets match the paper in distribution only (seeds unrecorded).",
        fontsize=8,
        color=style.TEXT_SECONDARY,
    )
    paths = style.save(
        fig,
        "fig6d_vqc_accuracy",
        [args.budget_sweep_dir, args.baseline_sweep_dir, args.results],
    )
    print("\n".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
