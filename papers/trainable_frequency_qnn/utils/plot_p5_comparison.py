#!/usr/bin/env python3
"""Forest plot of the Phase 5 pairwise comparisons (method vs baseline).

Reads the JSON written by ``utils/compare_fair.py`` and plots, per comparison,
the relative difference r = (A - B) / B of the declared metric (A = TF side;
negative = TF lower = better). Error bars show ±1 std of the per-seed paired
difference (relative to B); box and unpaired rows have none. The grey band is the
frozen ±5 % tie tolerance. Learning-rate robustness rows and E5 best-case rows
are left out of the figure (they are in the table).

Example (from the paper folder)::

    python utils/plot_p5_comparison.py --input results/p5_comparison.json \
        --output results/p5_comparison_forest.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

QUBIT_COLOR = "#2a78d6"
MERLIN_COLOR = "#eb6834"
CONTEXT_COLOR = "#8a8a85"
INK = "#2b2b29"
MUTED = "#6b6b66"
CLIP = (-1.0, 1.6)

DEPTH_ROWS_KEPT = {"E3.ff_L4", "E3.ff_L128"}
METRIC_LABEL = {
    "train_mse": "train MSE",
    "box_mean": "box mean",
    "box_worst": "box worst",
    "u_mean_maerm": "MAERM u",
    "v_mean_maerm": "MAERM v",
    "p_mean_maerm": "MAERM p",
}


def selected_rows(comparisons: list[dict]) -> list[dict]:
    rows = []
    for row in comparisons:
        if row["claim"] == "C7" or row["metric"] == "box_best":
            continue
        if row["b"].startswith("E3.") and row["b"] not in DEPTH_ROWS_KEPT:
            continue
        rows.append(row)
    return rows


def row_label(row: dict) -> str:
    seeds = {
        "paired": f"n={row.get('n_paired')}",
        "unpaired": "unpaired",
        "box": "7 datasets",
    }[row["mode"]]
    partial = " [partial]" if row["commensurability"] != "COMMENSURABLE" else ""
    return f"{row['claim']}  {row['a']} vs {row['b']} · {METRIC_LABEL[row['metric']]} ({seeds}){partial}"


def plot(comparisons: list[dict], output: Path) -> None:
    rows = selected_rows(comparisons)
    fig, ax = plt.subplots(figsize=(10, 0.24 * len(rows) + 1.6))
    ax.axvspan(-0.05, 0.05, color="#e6e5e0", zorder=0, label="±5 % tie tolerance")
    ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
    for index, row in enumerate(rows):
        y = len(rows) - 1 - index
        if row["commensurability"] != "COMMENSURABLE":
            color = CONTEXT_COLOR
        elif row["claim"] == "MerLin":
            color = MERLIN_COLOR
        else:
            color = QUBIT_COLOR
        decided = row["decision"] not in ("NO_SEPARATION", "TIE")
        value = row["relative_difference"]
        shown = min(max(value, CLIP[0]), CLIP[1])
        error = row.get("paired_difference_std_rel")
        if error is not None:
            ax.errorbar(
                shown,
                y,
                xerr=abs(error),
                color=color,
                linewidth=1.5,
                capsize=0,
                zorder=2,
            )
        ax.plot(
            shown, y, marker="o", markersize=7,
            markerfacecolor=color if decided else "white",
            markeredgecolor=color, markeredgewidth=1.5, linestyle="none", zorder=3,
        )  # fmt: skip
        if value > CLIP[1]:
            ax.annotate(f"{value:+.0%} →", (CLIP[1], y), xytext=(-4, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=7, color=MUTED)  # fmt: skip
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row_label(r) for r in reversed(rows)], fontsize=7.5, color=INK)
    ax.set_xlim(CLIP[0] - 0.05, CLIP[1] + 0.05)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(
        "r = (TF − baseline) / baseline   ← TF lower (better)   |   baseline lower →",
        color=INK,
    )
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color=QUBIT_COLOR,
            linestyle="none",
            label="qubit, commensurable (paper-accurate)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color=MERLIN_COLOR,
            linestyle="none",
            label="MerLin adaptation, simulation",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            color=CONTEXT_COLOR,
            linestyle="none",
            label="partially commensurable / context",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=MUTED,
            linestyle="none",
            label="hollow: no separation / tie under frozen rule",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=2,
        fontsize=7.5,
        frameon=False,
    )
    ax.set_title(
        "Phase 5: TF vs matched baselines (local runs; bars ±1 std of paired difference)",
        fontsize=10, color=INK, loc="left", pad=34,
    )  # fmt: skip
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plot(json.loads(args.input.read_text(encoding="utf-8"))["comparisons"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
