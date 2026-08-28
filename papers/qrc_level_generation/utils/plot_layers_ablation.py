"""Plot the 1-vs-2 post-encoding entangling-layer comparison.

Reads the curated metrics of the two comparison runs
(``results/layers_ablation_post{1,2}.json``) and draws originality (L=2)
and broken-rate (rule 2) against temperature for both depths. The two
curves coinciding within single-seed noise is the visual evidence that
post-encoding mesh depth is not an expressivity knob for a frozen
linear-optical reservoir.

Usage (from the paper directory):

    python utils/plot_layers_ablation.py [--out results/layers_ablation.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]

SERIES = [
    ("1 layer", "layers_ablation_post1.json", "#1f77b4", "o", "-"),
    ("2 layers", "layers_ablation_post2.json", "#ff7f0e", "s", "--"),
]
PANELS = [
    ("originality", "2", "Originality (L = 2)"),
    ("broken_rate_per_rule", "2", 'Broken-rate (rule "2")'),
]


def load_curve(path: Path, section: str, key: str) -> tuple[list[float], list[float]]:
    metrics = json.loads(path.read_text())["qrc"]
    temps = sorted(metrics, key=lambda s: float(s.split("=")[1]))
    x = [float(t.split("=")[1]) for t in temps]
    y = [metrics[t][section][key] for t in temps]
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=PROJECT / "results" / "layers_ablation.png"
    )
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, (section, key, title) in zip(axes, PANELS):
        for label, filename, color, marker, linestyle in SERIES:
            x, y = load_curve(PROJECT / "results" / filename, section, key)
            ax.plot(
                x,
                y,
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=6,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Temperature")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[0].legend(title="Post-encoding depth", fontsize=9, title_fontsize=9)
    fig.suptitle(
        "Frozen photonic reservoir: post-encoding mesh depth is inert", fontsize=12
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
