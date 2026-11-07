"""Dot-and-interval chart of the Phase 5 fair comparison.

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/plot_fair_comparison.py \\
        --comparison papers/contextuality_inductive_bias/results/fair_comparison.json \\
        --final-dir papers/contextuality_inductive_bias/outdir/sweeps/fair_final \\
        --out papers/contextuality_inductive_bias/results/fair_comparison.png

Every system appears with its mean final-step test KL and a t-based 95%
confidence interval over its seeds (filled marker), plus its mean validation
early-stopped test KL (hollow marker). Hue encodes the model family; all
systems are locally executed under the same fair protocol (LOG.md, Phase 5
Comparison Plan). The PNG is also copied into the final sweep directory.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

# Reference palette, categorical slots in fixed order (dataviz skill).
FAMILY_COLOURS = {
    "quantum (paper model)": "#2a78d6",
    "MerLin adaptation": "#4a3aa7",
    "paper classical baseline": "#1baf7a",
    "additional classical baseline": "#eb6834",
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _family(role: str) -> str:
    if role.startswith("paper method") or role.startswith("paper baseline (fair"):
        return "quantum (paper model)"
    if role.startswith("MerLin"):
        return "MerLin adaptation"
    if role.startswith("paper baseline B0"):
        return "paper classical baseline"
    return "additional classical baseline"


def plot(comparison_path: Path, final_dir: Path, out: Path) -> Path:
    """Draw the chart and return the written path."""
    rows = json.loads(comparison_path.read_text(encoding="utf-8"))["rows"]
    rows = sorted(rows, key=lambda r: r["final_test_kl_mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.6))
    for index, row in enumerate(rows):
        colour = FAMILY_COLOURS[_family(row["role"])]
        n = row["successful"]
        half = stats.t.ppf(0.975, n - 1) * row["final_test_kl_std"] / math.sqrt(n)
        ax.errorbar(
            row["final_test_kl_mean"],
            index,
            xerr=half,
            fmt="o",
            color=colour,
            markersize=8,
            capsize=4,
            linewidth=2,
        )
        ax.plot(
            row["early_stop_test_kl_mean"],
            index,
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor=colour,
            linestyle="none",
        )
        label = (
            f"{row['system']}  ({row['n_parameters']} params, n={n}/{row['expected']})"
        )
        ax.text(
            1.0,
            index,
            "  " + label,
            transform=ax.get_yaxis_transform(),
            va="center",
            color=TEXT,
            fontsize=8,
        )
    ax.set_yticks([])
    ax.set_xlabel("test KL divergence (lower is better)", color=TEXT)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    handles = [
        Line2D([], [], marker="o", color=c, linestyle="none", markersize=8, label=f)
        for f, c in FAMILY_COLOURS.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            color=MUTED,
            linestyle="none",
            markersize=8,
            label="final step 2000, mean with 95% CI",
        ),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="none",
            markeredgecolor=MUTED,
            linestyle="none",
            markersize=8,
            label="validation early-stopped, mean",
        ),
    ]
    ax.legend(
        handles=handles, loc="upper right", frameon=False, fontsize=8, labelcolor=TEXT
    )
    ax.set_title(
        "Fair comparison, protocol 'paper': every model tuned on validation NLL only; seeds 10-19, locally executed",
        color=TEXT,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(out, final_dir / out.name)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(plot(args.comparison, args.final_dir, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
