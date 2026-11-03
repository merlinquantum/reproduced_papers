"""Forest plot of the Phase 5 paired comparisons, one panel per comparative claim.

Reads ``results/phase5_comparison.json`` written by ``utils/compare_phase5.py``;
never reads logs or run directories.

Usage (from the repository root)::

    python papers/photonic_QKD/utils/plot_phase5.py \
        --report papers/photonic_QKD/results/phase5_comparison.json

Writes ``results/phase5_paired_differences.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
PANELS = {
    "A": "A · compactness\n(accuracy at the trainable budget)",
    "B": "B · photonic contribution\n(same architecture, other z)",
    "C": "C · shot noise and EMA\n(sampled z)",
    "check": "check · photon number\n(4 vs 8 photons)",
}
INK = "#2a78d6"
NEUTRAL = "#6b6a64"


def _label(entry: dict) -> str:
    def short(arm: str, sweep: str) -> str:
        name = "PQKD" if arm == "pqkd" else arm.split("_")[0]
        return f"{name} ({sweep.upper()})"

    return f"{short(entry['arm'], entry['arm_sweep'])} − {short(entry['reference'], entry['reference_sweep'])}"


def plot(report: dict) -> Path:
    claims = [
        claim for claim in PANELS if any(e["claim"] == claim for e in report["paired"])
    ]
    heights = [
        sum(e["claim"] == claim for e in report["paired"]) + 1.2 for claim in claims
    ]
    figure, axes = plt.subplots(
        len(claims),
        1,
        figsize=(8.5, 0.42 * sum(heights) + 1.6),
        gridspec_kw={"height_ratios": heights},
    )
    for axis, claim in zip(axes, claims):
        entries = [e for e in report["paired"] if e["claim"] == claim]
        for row, entry in enumerate(reversed(entries)):
            difference = entry["test_accuracy_difference"]
            separated = difference["separated"]
            axis.errorbar(
                difference["mean"],
                row,
                xerr=difference["std"],
                fmt="o",
                color=INK,
                markerfacecolor=INK if separated else "white",
                markeredgewidth=1.5,
                markersize=7,
                capsize=3,
                linewidth=1.5,
            )
            axis.scatter(
                difference["values"],
                [row] * len(difference["values"]),
                color=NEUTRAL,
                s=10,
                alpha=0.6,
                zorder=3,
            )
        axis.axvline(0, color=NEUTRAL, linewidth=1.0)
        axis.set_yticks(
            range(len(entries)), [_label(e) for e in reversed(entries)], fontsize=8
        )
        axis.set_ylim(-0.7, len(entries) - 0.3)
        axis.set_title(PANELS[claim], fontsize=9, loc="left")
        axis.grid(axis="x", color="0.9", linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[-1].set_xlabel("paired Δ final-epoch test accuracy, arm − reference (pp)")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=INK,
            markersize=7,
            label="separated: |mean Δ| > s.d. of Δ",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=INK,
            markerfacecolor="white",
            markeredgewidth=1.5,
            markersize=7,
            label="not separated",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=NEUTRAL,
            markersize=4,
            label="per-seed Δ",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(
        "Phase 5 paired comparisons, MNIST (partial RED-tier run: 30 epochs, seeds 0–2, CPU)\n"
        "S4 = Phase 3 sweep (8 photons / classical); S6 = Phase 4 MerLin adaptation (4 photons, simulated)",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    path = PAPER_DIR / "results" / "phase5_paired_differences.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(plot(json.loads(args.report.read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
