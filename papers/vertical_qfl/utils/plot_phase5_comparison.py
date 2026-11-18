"""Plot the Phase 5 comparison from ``results/phase5_comparison.json``.

Each panel is one commensurable pair (quantum or photonic model against its
parameter-matched classical baseline) across the four layouts; the centralised
references are drawn as horizontal lines. Paper-reported values are marked
separately and are not ranked.

Example
-------
    python utils/plot_phase5_comparison.py --comparison results/phase5_comparison.json
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
LAYOUTS = ["E1", "E2", "E3", "E4"]
PANELS = [
    ("gate", "classical_gate", "classical_central:gate", "Gate model (72k params)"),
    (
        "photonic_h",
        "classical_photonic_h",
        "classical_central:photonic_h",
        "Photonic H (3.2k, 410 classical)",
    ),
    (
        "photonic_f",
        "classical_photonic_f",
        "classical_central:photonic_f",
        "Photonic F (2.7k, no classical layer)",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=PROJECT_DIR / "results/phase5_comparison.json",
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_DIR / "results/phase5_comparison.png"
    )
    args = parser.parse_args()
    data = json.loads(args.comparison.read_text())
    rows = {(r["system"], r["layout"]): r for r in data["systems"]}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    x = np.arange(len(LAYOUTS))
    for ax, (quantum, classical, central, title) in zip(axes, PANELS):
        for offset, system, color, label in (
            (-0.17, quantum, "#1b6ca8", "quantum / photonic (1 seed per distribution)"),
            (0.17, classical, "#b5651d", "matched classical vertical FL (seed 42)"),
        ):
            means = [100 * rows[(system, layout)]["mean_seed42"] for layout in LAYOUTS]
            stds = [100 * rows[(system, layout)]["std_seed42"] for layout in LAYOUTS]
            ax.errorbar(
                x + offset,
                means,
                yerr=stds,
                fmt="o",
                color=color,
                capsize=4,
                label=label,
            )
        ref = 100 * rows[(central, "central")]["mean_seed42"]
        ax.axhline(ref, color="grey", ls=":", lw=1)
        ax.text(
            3.4,
            ref + 0.6,
            "centralised MLP (no federation)",
            color="grey",
            fontsize=7,
            ha="right",
        )
        if quantum == "gate":
            ax.scatter(
                [0],
                [94],
                marker="x",
                color="black",
                zorder=5,
                label="paper text E1 (>94%), not local",
            )
            ax.scatter([3], [90], marker="x", color="black", zorder=5)
        ax.set_xticks(x, LAYOUTS)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel(
        "final-epoch test accuracy (%), mean +- std over 5 distributions"
    )
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Phase 5: each model vs a parameter-matched classical baseline (mean +- std over 5 distributions; same data, split, epochs; "
        "photonic = MerLin adaptation, simulator)",
        fontsize=9,
    )
    fig.tight_layout()
    args.out.parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
