"""Plot FD comparison across {quantum, classical, merlin, random_decoder}.

Reads ``results/sweep_summary.csv`` and writes ``results/fd_comparison.png``.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    summary = HERE / "results" / "sweep_summary.csv"
    if not summary.exists():
        print(f"No sweep summary at {summary}; run utils/run_sweep.py first.")
        sys.exit(1)
    rows = list(csv.DictReader(open(summary)))
    by_model_digit: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        m = r["model"]
        d = int(r["digit"])
        fd = float(r["best_fd"]) if r["best_fd"] else float("nan")
        by_model_digit[(m, d)].append(fd)

    digits = sorted({int(r["digit"]) for r in rows})
    models = ["quantum", "merlin", "classical", "random_decoder"]
    labels = {
        "quantum": "LatentQGAN (gate)",
        "merlin": "LatentQGAN (MerLin photonic)",
        "classical": "LatentGAN (classical, ~iso-param)",
        "random_decoder": "RandomDecoder",
    }
    colors = {
        "quantum": "#1f77b4",
        "merlin": "#9467bd",
        "classical": "#2ca02c",
        "random_decoder": "#d62728",
    }

    n_seeds = max((len(v) for v in by_model_digit.values()), default=1)

    x = np.arange(len(digits))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        means, stds = [], []
        for d in digits:
            vals = by_model_digit.get((m, d), [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0.0)
            else:
                means.append(np.nan)
                stds.append(0.0)
        ax.bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=labels[m],
            color=colors[m],
        )

    ax.set_xlabel("MNIST class")
    ax.set_ylabel("Best Fréchet Distance (lower is better)")
    ax.set_title(
        "LatentQGAN reproduction — best FD by model and class\n"
        f"Reduced compute (AE 40 epochs / 20k imgs, 1000 GAN iters, "
        f"{n_seeds} seed{'s' if n_seeds != 1 else ''})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in digits])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = HERE / "results" / "fd_comparison.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
