#!/usr/bin/env python3
"""Create a CPU/PneumoniaMNIST adaptation of the paper's Table I."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_qsvm_results(result_dirs):
    rows = []
    for result_dir in result_dirs:
        metric_paths = sorted(result_dir.rglob("metrics_summary.csv"))
        if not metric_paths:
            raise FileNotFoundError(f"No metrics_summary.csv under {result_dir}")
        for metrics_path in metric_paths:
            info_path = metrics_path.parent / "dataset_info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"Missing {info_path}")
            with info_path.open() as file:
                info = json.load(file)
            metrics = pd.read_csv(metrics_path)
            if "alpha_sweep" in metrics.columns:
                best_rows = metrics[metrics["alpha_sweep"] == "best"]
                metrics = best_rows if not best_rows.empty else metrics.iloc[[-1]]
            row = metrics.iloc[-1]
            rows.append(
                {
                    "seed": int(info["seed"]),
                    "pca_dim": int(info["qubits"]),
                    "test_f1_qsvm": float(row["test_minority_f1"]),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["seed", "pca_dim"], keep="last")


def compare(qsvm, classical_path, kernel):
    classical = pd.read_csv(classical_path)
    classical = classical[classical["kernel"] == kernel].copy()
    classical = classical.rename(columns={"test_minority_f1": "test_f1_classical"})
    paired = qsvm.merge(
        classical[["seed", "pca_dim", "test_f1_classical"]],
        on=["seed", "pca_dim"],
        how="inner",
    )
    if paired.empty:
        raise ValueError(f"No common (seed, q) pairs between QSVM and {classical_path}")
    per_configuration = (
        paired.groupby("pca_dim", as_index=False)
        .agg(
            qsvm_f1_mean=("test_f1_qsvm", "mean"),
            qsvm_f1_std=("test_f1_qsvm", "std"),
            classical_f1_mean=("test_f1_classical", "mean"),
            classical_f1_std=("test_f1_classical", "std"),
            seeds=("seed", "nunique"),
        )
        .rename(columns={"pca_dim": "q"})
    )
    per_configuration["delta_f1"] = (
        per_configuration["qsvm_f1_mean"] - per_configuration["classical_f1_mean"]
    )
    deltas = per_configuration["delta_f1"]
    summary = {
        "wins": int((deltas > 1e-12).sum()),
        "ties": int(np.isclose(deltas, 0.0, atol=1e-12).sum()),
        "configurations": len(per_configuration),
        "min_seeds": int(per_configuration["seeds"].min()),
        "mean_delta": float(deltas.mean()),
    }
    return summary, per_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qsvm_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--tier1_csv", type=Path, required=True)
    parser.add_argument("--tier2_csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qsvm = load_qsvm_results(args.qsvm_dirs)
    tier1, tier1_details = compare(qsvm, args.tier1_csv, "linear")
    tier2, tier2_details = compare(qsvm, args.tier2_csv, "rbf")

    rows = [
        [
            "1",
            "C=1, q qubits\n(untuned, CPU)",
            "C=1, PCA-q\nlinear (untuned)",
            f"{tier1['wins']}/{tier1['configurations']} minority-F1 wins\n"
            f"({tier1['ties']} ties, {tier1['min_seeds']} seeds/q)",
        ],
        [
            "2",
            "C=1, q qubits\n(untuned, CPU)",
            "best-C, PCA-q\nRBF (validation-tuned)",
            f"{tier2['wins']}/{tier2['configurations']} minority-F1 wins\n"
            f"({tier2['ties']} ties, {tier2['min_seeds']} seeds/q)",
        ],
    ]

    figure, axis = plt.subplots(figsize=(10, 3.2))
    axis.axis("off")
    axis.set_title(
        "Adapted Table I — PneumoniaMNIST / CPU\n"
        "Wins are counted per q after averaging paired seeds",
        fontsize=12,
        pad=14,
    )
    table = axis.table(
        cellText=rows,
        colLabels=["Tier", "QSVM", "Classical SVM", "Result"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.08, 0.28, 0.31, 0.33],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    for column in range(4):
        table[(0, column)].set_text_props(weight="bold")
        table[(0, column)].set_facecolor("#e8e8e8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    summary_path = args.output.with_suffix(".csv")
    pd.DataFrame(
        [
            {"tier": 1, **tier1},
            {"tier": 2, **tier2},
        ]
    ).to_csv(summary_path, index=False)
    details_path = args.output.with_name(f"{args.output.stem}_details.csv")
    pd.concat(
        [
            tier1_details.assign(tier=1),
            tier2_details.assign(tier=2),
        ],
        ignore_index=True,
    ).to_csv(details_path, index=False)
    print(f"Saved: {args.output}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {details_path}")


if __name__ == "__main__":
    main()
