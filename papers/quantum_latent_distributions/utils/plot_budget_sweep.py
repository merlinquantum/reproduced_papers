"""Plot how the boson-vs-distinguishable gap moves with the training budget.

Usage
-----
    python utils/plot_budget_sweep.py results/budget_sweep --out results/budget_sweep.png

Reads ``<latent>_<iterations>.json`` metrics files (as written by
``lib.experiments.run_synthetic_datasets``) and overlays the paper's reported
Table I values at 40k iterations.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Paper Table I, quantum dataset, 40k iterations, 12 runs.
PAPER = {
    "boson": (40_000, 0.036, 0.001),
    "distinguishable": (40_000, 0.041, 0.002),
    "gaussian": (40_000, 0.061, 0.001),
}
STYLE = {
    "boson": ("#3b6ea5", "o", "Boson sampler"),
    "distinguishable": ("#d1495b", "s", "Distinguishable sampler"),
    "gaussian": ("#7a6f9b", "^", "Gaussian"),
}


def collect(directory: Path) -> dict[str, dict[int, list[float]]]:
    """Group L1 values by latent and by iteration count."""
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(directory.glob("*.json")):
        match = re.fullmatch(r"(.+)_(\d+)", path.stem)
        if not match:
            continue
        latent, iterations = match.group(1), int(match.group(2))
        for record in json.loads(path.read_text()):
            out[latent][iterations].append(record["l1_nearest_int"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/budget_sweep.png"))
    args = parser.parse_args()

    grouped = collect(args.directory)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for latent, (color, marker, label) in STYLE.items():
        points = sorted(grouped.get(latent, {}).items())
        if not points:
            continue
        x = [it for it, _ in points]
        y = [st.fmean(v) for _, v in points]
        err = [st.stdev(v) / sqrt(len(v)) if len(v) > 1 else 0.0 for _, v in points]
        ax.errorbar(x, y, yerr=err, color=color, marker=marker, capsize=3, label=label)
        p_it, p_mean, p_sem = PAPER[latent]
        ax.errorbar(
            [p_it],
            [p_mean],
            yerr=[p_sem],
            color=color,
            marker=marker,
            markerfacecolor="white",
            capsize=3,
            linestyle="none",
        )
        ax.plot([x[-1], p_it], [y[-1], p_mean], color=color, linestyle=":", linewidth=1)

    ax.set_xscale("log")
    ticks = sorted({it for series in grouped.values() for it in series} | {40_000})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(ticks[0] * 0.8, ticks[-1] * 1.35)
    ax.set_xlabel("training iterations")
    ax.set_ylabel("L1 to nearest integer (lower is better)")
    ax.set_title("Converging on the paper as the budget approaches theirs", fontsize=11)
    ax.annotate(
        "paper (12 runs)",
        xy=(40_000, 0.0405),
        xytext=(40_000, 0.047),
        fontsize=8,
        color="#555555",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#999999", "linewidth": 0.8},
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
