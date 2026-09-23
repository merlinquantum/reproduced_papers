"""Click-density figure for README.md.

Reads results/summary.json (the committed aggregate over every run) and
results/baselines.json (classical baselines at the same Appendix B budget) and
writes results/click_density_m30.png. Nothing here is taken from the paper.

    python utils/plot_density.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#ffffff"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#0ca30c"

# Where each source label sits relative to its point, in typographic points.
OFFSETS = {
    "bernoulli@0.87": ((-8, -15), "center"),
    "bernoulli@1.0": ((0, 11), "center"),
    "bernoulli@1.15": ((0, 11), "center"),
    "bernoulli@1.3": ((0, 11), "center"),
    "bernoulli@1.5": ((0, 11), "center"),
    "boson": ((12, -4), "left"),
    "shuffled boson": ((12, -4), "left"),
    "distinguishable": ((12, -4), "left"),
}
LADDER = [
    "bernoulli@0.87",
    "bernoulli@1.0",
    "bernoulli@1.15",
    "bernoulli@1.3",
    "bernoulli@1.5",
]
PHYSICAL = ["boson", "shuffled boson", "distinguishable"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results", type=Path)
    parser.add_argument("--out", default=None)
    parser.add_argument("--m", type=int, default=30)
    args = parser.parse_args()

    summary = json.loads((args.results / "summary.json").read_text())
    arms = {
        a["arm"]: a
        for a in summary["arms"]
        if a["family"] == "knapsack" and a["m"] == args.m and a["arm"] in OFFSETS
    }

    baselines_path = args.results / "baselines.json"
    baselines = (
        json.loads(baselines_path.read_text()) if baselines_path.exists() else {}
    )
    sa = next(
        (
            b
            for b in baselines.get("arms", [])
            if b["method"] == "simulated_annealing" and b["m"] == args.m
        ),
        None,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    if sa is not None:
        ax.axhline(
            sa["percent_optimal"],
            color=GREEN,
            linewidth=2,
            linestyle=(0, (5, 3)),
            zorder=2,
        )
        ax.annotate(
            f"simulated annealing, same budget ({sa['percent_optimal']:.0f}%)",
            xy=(0.015, sa["percent_optimal"]),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, 6),
            fontsize=9,
            color=GREEN,
        )

    ladder = [arms[name] for name in LADDER if name in arms]
    ax.plot(
        [a["mean_click_density"] for a in ladder],
        [a["percent_optimal"] for a in ladder],
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=3,
        label=r"independent Bernoulli clicks, rate scaled by $r$",
    )
    physical = [arms[name] for name in PHYSICAL if name in arms]
    ax.scatter(
        [a["mean_click_density"] for a in physical],
        [a["percent_optimal"] for a in physical],
        s=95,
        color=ORANGE,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=4,
        label="sources with photon statistics",
    )

    for arm in ladder + physical:
        offset, align = OFFSETS[arm["arm"]]
        is_physical = arm in physical
        label = (
            arm["arm"].replace("bernoulli@", "r=") if not is_physical else arm["arm"]
        )
        ax.annotate(
            label,
            (arm["mean_click_density"], arm["percent_optimal"]),
            textcoords="offset points",
            xytext=offset,
            ha=align,
            fontsize=8.5,
            color=ORANGE if is_physical else SECONDARY,
            fontweight="bold" if is_physical else "normal",
        )

    ax.set_xlabel("mean click density (clicks per mode)", fontsize=10, color=SECONDARY)
    ax.set_ylabel("instances solved to optimality (%)", fontsize=10, color=SECONDARY)
    ax.set_title(
        f"Knapsack, m = {args.m}: same circuit, same candidate budget, only the click source changes",
        fontsize=10.5,
        color=INK,
        pad=12,
        loc="left",
    )
    ax.set_xlim(0.352, 0.575)
    ax.set_ylim(65, 106)
    ax.set_yticks([70, 80, 90, 100])
    ax.set_xticks([0.35, 0.40, 0.45, 0.50, 0.55])
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    legend = ax.legend(loc="lower right", frameon=False, fontsize=9, borderaxespad=0.8)
    for text in legend.get_texts():
        text.set_color(SECONDARY)

    out = args.out or (args.results / f"click_density_m{args.m}.png")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
