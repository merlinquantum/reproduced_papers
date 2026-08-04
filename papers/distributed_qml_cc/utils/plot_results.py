"""Plotting helpers reproducing Fig. 4c, Fig. 4d, and Table I of the paper.

Inputs:
    --sweep PATH       Path to ``results/sweep/sweep.json`` produced by
                       ``utils/run_sweep.py``.
    --outdir PATH      Directory to write figures and tables into. Defaults
                       to the same directory as the sweep file.

Outputs:
    fig4c_training_curves.png   Validation accuracy versus iteration for
                                the four schemes at the largest L in the
                                sweep (mirrors Fig. 4c, L=9 in the paper).
    fig4d_acc_vs_layers.png     Mean validation accuracy at the final
                                iteration versus convolutional sub-layers L
                                with error bars (mirrors Fig. 4d).
    table1.md                   Markdown table mirroring Table I.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCHEME_ORDER = ["non", "nc", "cc", "qc"]
SCHEME_LABELS = {
    "non": "Non-DQML",
    "nc": "NC-DQML",
    "cc": "CC-DQML",
    "qc": "QC-DQML",
}
SCHEME_COLORS = {
    "non": "#888888",
    "nc": "#1f77b4",
    "cc": "#d62728",
    "qc": "#2ca02c",
}


def _grouped(results, key="scheme"):
    out: dict = {}
    for entry in results:
        out.setdefault(entry[key], []).append(entry)
    return out


def plot_fig4c(sweep: dict, outdir: Path) -> Path:
    layers_available = sorted({r["n_layers"] for r in sweep["results"]})
    target_L = max(layers_available)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for scheme in SCHEME_ORDER:
        entries = [r for r in sweep["results"] if r["scheme"] == scheme and r["n_layers"] == target_L]
        if not entries:
            continue
        entry = entries[0]
        seeds = entry["runs"]
        # Each seed has a list of (iteration, val_acc).
        iters = np.array(seeds[0]["history"]["iteration"], dtype=float)
        accs = np.stack([np.array(s["history"]["val_acc"]) for s in seeds], axis=0)
        mean = accs.mean(0)
        std = accs.std(0)
        ax.plot(iters, mean, label=SCHEME_LABELS[scheme], color=SCHEME_COLORS[scheme], lw=2)
        ax.fill_between(iters, mean - std, mean + std, color=SCHEME_COLORS[scheme], alpha=0.18)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_title(f"Reproduction of Fig. 4c (L={target_L})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = outdir / "fig4c_training_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_fig4d(sweep: dict, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    layers_available = sorted({r["n_layers"] for r in sweep["results"]})
    for scheme in SCHEME_ORDER:
        means = []
        stds = []
        layers_for_scheme = []
        for L in layers_available:
            entries = [r for r in sweep["results"] if r["scheme"] == scheme and r["n_layers"] == L]
            if not entries:
                continue
            entry = entries[0]
            accs = np.array([s["final_val_acc"] for s in entry["runs"]])
            means.append(accs.mean())
            stds.append(accs.std())
            layers_for_scheme.append(L)
        if not means:
            continue
        ax.errorbar(layers_for_scheme, means, yerr=stds, marker="o",
                    label=SCHEME_LABELS[scheme], color=SCHEME_COLORS[scheme],
                    capsize=4, lw=2)
    ax.set_xlabel("Convolutional sub-layers (L)")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.55, 1.02)
    ax.set_title("Reproduction of Fig. 4d")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = outdir / "fig4d_acc_vs_layers.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_table1(sweep: dict, outdir: Path) -> Path:
    layers_available = sorted({r["n_layers"] for r in sweep["results"]})
    lines = []
    lines.append(
        "| L | non-DQML | NC-DQML | CC-DQML | QC-DQML |\n"
        "|---|---------:|--------:|--------:|--------:|"
    )
    for L in layers_available:
        row = [f"| {L}"]
        for scheme in SCHEME_ORDER:
            entries = [r for r in sweep["results"] if r["scheme"] == scheme and r["n_layers"] == L]
            if not entries:
                row.append("| n/a")
                continue
            accs = np.array([s["final_val_acc"] for s in entries[0]["runs"]])
            row.append(f"| {accs.mean()*100:.2f} ± {accs.std()*100:.2f}")
        row.append("|")
        lines.append(" ".join(row))
    text = "\n".join(lines) + "\n"
    path = outdir / "table1.md"
    path.write_text(text)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=str, required=True, help="Path to sweep.json")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    sweep_path = Path(args.sweep)
    sweep = json.loads(sweep_path.read_text())
    outdir = Path(args.outdir) if args.outdir else sweep_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    p1 = plot_fig4c(sweep, outdir)
    p2 = plot_fig4d(sweep, outdir)
    p3 = write_table1(sweep, outdir)
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {p3}")


if __name__ == "__main__":
    main()
