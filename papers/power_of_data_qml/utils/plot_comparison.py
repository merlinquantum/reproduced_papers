"""Phase 5 figure: quantum - best classical (paired, 95% CI) vs n, per family.

Usage (from the paper folder)::

    python utils/plot_comparison.py --comparison results/comparison_extended.json.gz \
        --output results/figures/phase5_comparison.png

Each panel compares a quantum kernel with *its own family's* best classical
model on the same labels (gate: paper method; photonic: MerLin adaptation,
simulated). Gate and photonic panels are not a single leaderboard: their labels
differ (LOG.md, Phase 5 commensurability). Protocol: cv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

from lib.result_io import load_json  # noqa: E402

COLORS = {
    "E1": "#2a78d6",
    "E2": "#eb6834",
    "E3": "#1baf7a",
    "PA": "#2a78d6",
    "PB": "#eb6834",
}
MARKERS = {"E1": "o", "E2": "s", "E3": "^", "PA": "o", "PB": "s"}
PANELS = [
    (
        "C1/C10 quantum dataset (MAE)",
        600,
        "QK (fidelity)",
        "FQK − best classical MAE\n(> 0: classical better)",
    ),
    (
        "C5/C6 PQK-engineered (accuracy)",
        600,
        "PQK (projected)",
        "PQK − best classical accuracy\n(> 0: PQK better)",
    ),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = load_json(args.comparison)["rows"]
    largest_n = {
        family: max(r["n"] for r in rows if r["family"] == family)
        for family in ("gate", "photonic")
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    for col, family in enumerate(("gate", "photonic")):
        for row_index, (claim, train_size, system, ylabel) in enumerate(PANELS):
            ax = axes[row_index, col]
            ax.set_facecolor("#fcfcfb")
            ax.grid(True, color="#e4e3df", linewidth=0.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.axhline(0, color="#52514e", linewidth=0.8)
            # The capped-photon regime (k <= 8) is a separate family drawn with
            # hollow markers next to the k = m/2 photonic curve.
            for row_family, suffix, hollow in (
                (family, "", False),
                (f"{family}_capped", " (k ≤ 8)", True),
            ):
                selected = [
                    r
                    for r in rows
                    if r["claim"] == claim
                    and r["N"] == train_size
                    and r["family"] == row_family
                    and r["protocol"] == "cv"
                    and r["system"].startswith(system)
                ]
                for embedding in sorted({r["embedding"] for r in selected}):
                    points = sorted(
                        (r["n"], r["difference_vs_best"])
                        for r in selected
                        if r["embedding"] == embedding
                    )
                    ax.errorbar(
                        [p[0] for p in points],
                        [p[1]["mean"] for p in points],
                        yerr=[p[1]["ci95"] for p in points],
                        color=COLORS[embedding],
                        marker=MARKERS[embedding],
                        markerfacecolor="white" if hollow else COLORS[embedding],
                        markersize=6 if hollow else 5,
                        linewidth=1.6,
                        capsize=3,
                        label=embedding + suffix,
                    )
            title = (
                f"gate (paper method, n ≤ {largest_n['gate']})"
                if family == "gate"
                else f"MerLin, simulated (k = m/2, n ≤ {largest_n['photonic']})"
            )
            ax.set_title(f"{claim.split(' (')[0]} — {title}", fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row_index == 1:
                ax.set_xlabel("n (system size)", fontsize=9)
            ax.legend(fontsize=8, frameon=False)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(
        "Phase 5: quantum − best classical (paired over 10 seeds, 95% t-CI; protocol cv; N = 600)\n"
        "each family vs its own classical baseline on identical data — not one leaderboard",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor="white")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
