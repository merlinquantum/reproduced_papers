#!/usr/bin/env python3
"""Render comparison figures from one or more ``metrics.json`` files.

Usage:
    python utils/plot_from_metrics.py outdir/run_<a>/metrics.json [outdir/run_<b>/metrics.json ...]

Saves ``comparison_curves.png`` and ``comparison_table.csv`` next to the
first metrics file. Useful for assembling a single figure across runs that
share the same dataset preprocessing but vary the model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _collect(metrics_files: list[Path]) -> dict[str, dict]:
    by_name: dict[str, dict] = {}
    for path in metrics_files:
        payload = json.loads(path.read_text())
        for name, summary in payload["summary"].items():
            label = name if name not in by_name else f"{name}@{path.parent.name}"
            by_name[label] = {
                "summary": summary,
                "epochs": payload["per_seed"][name][0]["epochs"],
                "source": str(path),
            }
    return by_name


def _plot(by_name: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for ax, (key, title) in zip(
        axes,
        [
            ("loss", "Train loss"),
            ("accuracy", "Test accuracy"),
            ("precision", "Macro precision"),
            ("time_s", "Epoch wall-clock (s)"),
        ],
    ):
        for name, entry in by_name.items():
            xs = list(range(1, len(entry["epochs"]) + 1))
            ys = [e[key] for e in entry["epochs"]]
            ax.plot(xs, ys, marker="o", label=name)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _table(by_name: dict[str, dict], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "accuracy_mean",
                "accuracy_std",
                "precision_mean",
                "recall_mean",
                "param_count",
                "train_time_mean_s",
                "source",
            ]
        )
        for name, entry in by_name.items():
            s = entry["summary"]
            writer.writerow(
                [
                    name,
                    f"{s['accuracy_mean']:.4f}",
                    f"{s['accuracy_std']:.4f}",
                    f"{s['precision_mean']:.4f}",
                    f"{s['recall_mean']:.4f}",
                    s["param_count"],
                    f"{s['train_time_mean_s']:.2f}",
                    entry["source"],
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    args = parser.parse_args()
    files = [p.resolve() for p in args.metrics]
    by_name = _collect(files)
    out_dir = files[0].parent
    _plot(by_name, out_dir / "comparison_curves.png")
    _table(by_name, out_dir / "comparison_table.csv")
    print("wrote:", out_dir / "comparison_curves.png")
    print("wrote:", out_dir / "comparison_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
