"""Plot the classical-challenger study (not a paper figure).

Shows each latent's L1 against the boson sampler, ordered by how much of the
boson sampling distribution's structure it reproduces.

Usage
-----
    python utils/plot_challengers.py results/classical_challengers
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

QUANTUM = "#3b6ea5"
CLASSICAL = "#d1495b"
NEUTRAL = "#8d99ae"

# (file stem, label, what it matches, colour)
ROWS = [
    ("copula_boson", "Copula boson", "marginals + pairwise corr.", CLASSICAL),
    ("shuffled_boson", "Shuffled boson", "marginals exactly", CLASSICAL),
    (
        "distinguishable",
        "Distinguishable\n(paper's control)",
        "mean occupancy, fixed total",
        NEUTRAL,
    ),
    ("negative_binomial", "Negative binomial", "mean + variance", CLASSICAL),
    (
        "dirichlet_multinomial",
        "Dirichlet-multinomial",
        "mean + variance + fixed total",
        CLASSICAL,
    ),
    ("boson", "Boson sampler", "the real thing", QUANTUM),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("results/classical_challengers.png")
    )
    args = parser.parse_args()

    labels, means, errs, colors, notes = [], [], [], [], []
    for stem, label, note, color in ROWS:
        path = args.directory / f"{stem}.json"
        if not path.exists():
            continue
        values = [r["l1_nearest_int"] for r in json.loads(path.read_text())]
        labels.append(label)
        means.append(st.fmean(values))
        errs.append(st.stdev(values) / sqrt(len(values)) if len(values) > 1 else 0.0)
        colors.append(color)
        notes.append(note)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = range(len(labels))
    ax.barh(
        list(y),
        means,
        xerr=errs,
        color=colors,
        height=0.62,
        error_kw={"ecolor": "#444444", "capsize": 3, "linewidth": 1},
    )
    for i, (value, note) in enumerate(zip(means, notes)):
        ax.text(value + 0.0035, i, note, va="center", fontsize=8, color="#555555")

    boson = means[labels.index("Boson sampler")]
    ax.axvline(boson, color=QUANTUM, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("L1 distance to nearest integer (lower is better)")
    ax.set_xlim(0, max(means) * 1.42)
    ax.set_title(
        "A classical latent matching the boson sampler's dispersion\n"
        "and photon-number constraint closes the whole gap",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
