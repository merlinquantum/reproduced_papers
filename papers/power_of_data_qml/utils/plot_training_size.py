"""Fig. S2 (journal Supp. Fig. 3): prediction error vs training size, with paper values.

Usage (from the paper folder)::

    python utils/plot_training_size.py --aggregate results/training_size.json.gz \
        --label "paper-scale: n=10-13, 10 seeds" --output results/training_size_comparison

Reads the aggregate written by ``utils/aggregate_study.py`` and the per-seed
Source Data sheet; writes a comparison table (``.md``/``.json``) and a figure to
``results/figures/`` and to ``figures/`` inside every aggregated run directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

from lib import paper_values  # noqa: E402
from lib.result_io import load_json  # noqa: E402

SERIES = {
    "quantum_kernel": ("Quantum kernel (QK)", "#2a78d6", "o"),
    "projected_quantum_kernel": ("Projected QK (PQK)", "#eb6834", "s"),
    "best_classical": ("Best classical ML", "#1baf7a", "^"),
}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = parser.parse_args(argv)
    aggregate = load_json(args.aggregate)
    ours = {
        (r["quantity"], r["n"], r["train_size"]): r
        for r in aggregate["summary"]
        if r["component"] == "training_size" and r["protocol"] == "paper"
    }
    paper_rows = paper_values.load_training_size_values(args.data_root)
    paper = {}
    for quantity in SERIES:
        for n in sorted({r["n"] for r in paper_rows}):
            for size in sorted({r["train_size"] for r in paper_rows}):
                values = [
                    r[quantity]
                    for r in paper_rows
                    if r["n"] == n and r["train_size"] == size
                ]
                if values:
                    paper[(quantity, n, size)] = (
                        statistics.mean(values),
                        statistics.stdev(values),
                        len(values),
                    )
    sizes_n = sorted({key[1] for key in ours})
    train_sizes = sorted({key[2] for key in ours})

    lines = [
        f"# Fig. S2 comparison [{args.label}]",
        "",
        "MAE on Dataset (Q, E2), mean ± std over seeds (paper: Source Data per-seed sheet).",
        "",
        "| n | N | QK ours | QK paper | PQK ours | PQK paper | best classical ours | best classical paper | PQK − best (ours) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    table = []
    for n in sizes_n:
        for size in train_sizes:

            def cell(quantity, n=n, size=size):
                row = ours.get((quantity, n, size))
                return f"{row['mean']:.4f} ± {row['std']:.4f}" if row else "—"

            def pcell(quantity, n=n, size=size):
                value = paper.get((quantity, n, size))
                return (
                    f"{value[0]:.4f} ± {value[1]:.4f} (k={value[2]})" if value else "—"
                )

            difference = ours.get(
                ("projected_quantum_kernel_minus_best_classical", n, size)
            )
            diff_text = (
                f"{difference['mean']:+.4f} [±{difference['ci95_half_width']:.4f}]"
                if difference
                else "—"
            )
            lines.append(
                f"| {n} | {size} | {cell('quantum_kernel')} | {pcell('quantum_kernel')} | "
                f"{cell('projected_quantum_kernel')} | {pcell('projected_quantum_kernel')} | "
                f"{cell('best_classical')} | {pcell('best_classical')} | {diff_text} |"
            )
            table.append(
                {
                    "n": n,
                    "train_size": size,
                    **{
                        f"{q}_ours": ours[(q, n, size)]["mean"]
                        for q in SERIES
                        if (q, n, size) in ours
                    },
                    **{
                        f"{q}_paper": paper[(q, n, size)][0]
                        for q in SERIES
                        if (q, n, size) in paper
                    },
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    args.output.with_suffix(".json").write_text(json.dumps(table, indent=2) + "\n")

    fig, axes = plt.subplots(
        1, len(sizes_n), figsize=(3.4 * len(sizes_n), 3.6), sharey=True
    )
    for ax, n in zip(axes, sizes_n):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for quantity, (name, color, marker) in SERIES.items():
            rows = [
                ours[(quantity, n, s)] for s in train_sizes if (quantity, n, s) in ours
            ]
            ax.plot(
                [r["train_size"] for r in rows],
                [r["mean"] for r in rows],
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.6,
                label=f"{name} (ours)",
            )
            ax.fill_between(
                [r["train_size"] for r in rows],
                [r["mean"] - r["std"] for r in rows],
                [r["mean"] + r["std"] for r in rows],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
            points = [
                (s, paper[(quantity, n, s)][0])
                for s in train_sizes
                if (quantity, n, s) in paper
            ]
            if points:
                ax.plot(
                    *zip(*points),
                    color=color,
                    marker=marker,
                    markersize=4.5,
                    linewidth=1.2,
                    linestyle="--",
                    markerfacecolor="none",
                    label=f"{name} (paper)",
                )
        ax.set_xscale("log")
        ax.set_title(f"n = {n}", fontsize=9, color=TEXT_PRIMARY)
        ax.set_xlabel("training size N", fontsize=9, color=TEXT_PRIMARY)
    axes[0].set_ylabel("prediction error (MAE)", fontsize=9, color=TEXT_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        labelcolor=TEXT_PRIMARY,
    )
    fig.suptitle(
        f"Fig. S2 (journal Supp. Fig. 3): error vs training size, Dataset (Q, E2)\n[{args.label}]",
        fontsize=10,
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    for directory in [Path("results/figures")] + [
        Path(run) / "figures" for run in aggregate["runs"]
    ]:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "figS2_training_size.png", dpi=150, facecolor="white")
    print(f"wrote {args.output.with_suffix('.md')} and figS2_training_size.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
