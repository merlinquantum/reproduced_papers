"""Aggregate the (3,5)-MNIST QKS error rates across qubit counts and plot.

Usage:
    python utils/plot_mnist35_scaling.py \
        --run-dir outdir/run_1q outdir/run_2q outdir/run_4q \
        --label 1q 2q 4q \
        --baseline-lr <test_error_value> \
        --baseline-svm <test_error_value> \
        --out results/mnist35_error_vs_qubits.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--label", type=str, nargs="+", required=True)
    parser.add_argument("--baseline-lr", type=float, default=None)
    parser.add_argument("--baseline-svm", type=float, default=None)
    parser.add_argument("--paper-points", type=str, default=None,
                        help="comma-separated list of (label,error) pairs from the paper, e.g. '1q,0.033;2q,0.018;4q,0.014'")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if len(args.run_dir) != len(args.label):
        raise ValueError("--run-dir and --label must have the same length")

    means = []
    stds = []
    for d in args.run_dir:
        metrics = json.loads((d / "metrics.json").read_text())
        results = metrics["results"]
        errs = [1.0 - float(r["test_accuracy"]) for r in results]
        means.append(float(np.mean(errs)))
        stds.append(float(np.std(errs)))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(args.label))
    ax.errorbar(x, means, yerr=stds, marker="o", capsize=4, label="QKS (this work)")
    ax.set_xticks(x)
    ax.set_xticklabels(args.label)
    ax.set_xlabel("Circuit ansatz")
    ax.set_ylabel("(3,5)-MNIST test error")
    ax.grid(alpha=0.3)

    if args.baseline_lr is not None:
        ax.axhline(args.baseline_lr, color="C2", ls="--",
                   label=f"Logistic regression baseline ({args.baseline_lr:.3f})")
    if args.baseline_svm is not None:
        ax.axhline(args.baseline_svm, color="C3", ls=":",
                   label=f"SVM-RBF baseline ({args.baseline_svm:.3f})")

    if args.paper_points:
        labels = []
        values = []
        for tok in args.paper_points.split(";"):
            lab, val = tok.split(",")
            labels.append(lab.strip())
            values.append(float(val))
        xs = [args.label.index(lab) for lab in labels if lab in args.label]
        ys = [values[labels.index(args.label[i])] for i in xs]
        ax.scatter(xs, ys, color="C1", marker="s",
                   label="Paper Fig. 5 (QVM)", zorder=3)

    ax.legend()
    ax.set_title("(3,5)-MNIST: QKS error scaling with number of qubits")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
