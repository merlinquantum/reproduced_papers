"""Phase 5: plot the fair comparison from ``results/phase5_comparison.json``.

Usage (from the paper folder)::

    python utils/plot_phase5_comparison.py <comparison.json> <sweep_dir>

Left: C4, Fig. 3a boson-sampling data, primary metric (Gaussian MMD² on
binarized samples) per sigma for every system. Right: C8 part 2, Fig. 3e
movies, ``walsh_mmd`` per kernel for QCBM, paper RBM and B4. Dots: seeds;
bars: mean ± std. All values locally executed (paper values are plots only,
not digitized). The figure is saved to ``<sweep_dir>`` and ``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plot_common import NEUTRAL, SERIES, TEXT, plt, save, style_axes

C4_SYSTEMS = [
    # (key in the JSON, legend label, color, marker)
    ("QCBM (Phase 3, JAX + Perceval)", "QCBM (reproduction)", SERIES["model"][1], "o"),
    (
        "QCBM (Phase 4 MerLin adaptation)",
        "QCBM (MerLin adaptation)",
        SERIES["model"][1],
        "s",
    ),
    ("B5 distinguishable twin", SERIES["b5"][0], SERIES["b5"][1], "o"),
    ("B6 tuned RBM", "Tuned RBM (B6, Phase 5)", SERIES["rbm"][1], "D"),
    ("paper RBM", "Paper RBM", SERIES["rbm"][1], "^"),
    ("B4 marginal", SERIES["marginal"][0], SERIES["marginal"][1], "o"),
    ("uniform random", SERIES["random"][0], SERIES["random"][1], "o"),
]


def draw(ax, x, stats, color, marker, label, filled=True):
    values = np.asarray(stats["values"])
    ax.scatter(
        np.full(values.shape, x),
        values,
        s=10,
        color=color,
        alpha=0.45,
        linewidths=0,
        zorder=2,
    )
    ax.errorbar(
        x,
        stats["mean"],
        yerr=stats["std"],
        fmt=marker,
        markersize=7,
        color=color,
        markerfacecolor=color if filled else "white",
        markeredgewidth=1.5,
        elinewidth=1.5,
        capsize=0,
        label=label,
        zorder=3,
    )


def main() -> None:
    """Draw and save the two-panel comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("comparison", type=Path)
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    result = json.loads(args.comparison.read_text())
    systems = result["c4"]["systems"]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [3, 2]}
    )
    for ax in (left, right):
        style_axes(ax)
        ax.set_yscale("log")
    sigmas = ["2.0", "3.0", "4.5"]
    width = 0.8 / len(C4_SYSTEMS)
    for i, (key, label, color, marker) in enumerate(C4_SYSTEMS):
        for j, sigma in enumerate(sigmas):
            entry = systems[key].get(sigma)
            if entry is None:
                continue
            draw(
                left,
                j - 0.4 + width * (i + 0.5),
                entry["gaussian_mmd_binarized"],
                color,
                marker,
                label
                if j == 0 or sigma == "3.0" and key.endswith("adaptation)")
                else None,
                filled=key != "paper RBM",
            )
    left.set_xticks(range(len(sigmas)), [f"σ = {s}" for s in sigmas])
    left.set_ylabel(
        "Test MMD² on binarized samples (lower is better)", color=TEXT, fontsize=9
    )
    left.set_title(
        "C4 · boson-sampling data m=256, n=16 (Fig. 3a) · 5 seeds\n"
        "locally executed, simulated; QCBM MerLin adaptation at σ=3 only",
        fontsize=9,
        color=TEXT,
    )
    handles, labels = left.get_legend_handles_labels()
    order = {lab: k for k, (_, lab, _, _) in enumerate(C4_SYSTEMS)}
    pairs = sorted(zip(handles, labels), key=lambda p: order[p[1]])
    left.legend(*zip(*pairs), fontsize=7, frameon=False, loc="upper left", ncol=2)
    left.set_ylim(top=left.get_ylim()[1] * 4)

    kernels = list(result["c8_part2"]["kernels"])
    c8_series = [
        ("qcbm", "QCBM (reproduction)", SERIES["model"][1], "o", True),
        ("paper_rbm", "Paper RBM", SERIES["rbm"][1], "^", False),
        ("b4_marginal", SERIES["marginal"][0], SERIES["marginal"][1], "o", True),
    ]
    width = 0.8 / len(c8_series)
    for i, (key, label, color, marker, filled) in enumerate(c8_series):
        for j, kernel in enumerate(kernels):
            draw(
                right,
                j - 0.4 + width * (i + 0.5),
                result["c8_part2"]["kernels"][kernel][key],
                color,
                marker,
                label if j == 0 else None,
                filled=filled,
            )
    names = {
        "polynomial2": "polynomial deg 2\n(claim)",
        "gaussian2.5": "Gaussian σ=2.5",
        "gaussian3.5": "Gaussian σ=3.5",
    }
    right.set_xticks(range(len(kernels)), [names[k] for k in kernels])
    right.set_ylabel(
        "Test MMD² (kernel of the column; lower is better)", color=TEXT, fontsize=9
    )
    right.set_title(
        "C8 · movies m=100, n=10 (Fig. 3e) · 5 seeds\nrebuilt data; values not comparable across kernels",
        fontsize=9,
        color=TEXT,
    )
    right.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(
        "Phase 5 fair comparison — dots: seeds, markers and bars: mean ± std. "
        "Paper values are plots only (not digitized).\n"
        "A bar reaching the bottom axis means mean − std ≤ 0 (tuned RBM at σ = 3: one seed at 2.0e-3).",
        fontsize=9,
        color=NEUTRAL,
        y=1.02,
    )
    paths = save(fig, args.sweep_dir, "phase5_comparison")
    print(paths[-1])


if __name__ == "__main__":
    main()
