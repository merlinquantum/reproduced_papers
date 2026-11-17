"""Tabulate classification and kernel runs from their structured metrics.

Usage::

    python utils/summarize_results.py outdir/run_A outdir/run_B ... \
        --out results/summary_tables.md

Every number is read from ``metrics.json`` at full precision; the Markdown
table only rounds for display. Rows are labelled with the run's ``validity``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Paper accuracies (main text, Fig. 5): qubit = B2, qutrit = B3.
PAPER_ACCURACY = {
    ("nonlinear_1d", "qubit"): 0.90,
    ("nonlinear_1d", "qutrit"): 0.80,
    ("moons_kmeans", "qubit"): 0.65,
    ("moons_kmeans", "qutrit"): 0.90,
}


def paper_value(dataset: str, label: str) -> float | None:
    """Return the paper accuracy matching a model label, if the label names one."""
    for (paper_dataset, encoding), value in PAPER_ACCURACY.items():
        if paper_dataset == dataset and encoding in label:
            return value
    return None


def classification_rows(run_dir: Path) -> list[dict]:
    """Return one row per (model, arm) of a classification run."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    rows = []
    for label, result in metrics["results"].items():
        for arm, arm_result in result["arms"].items():
            aggregate = arm_result["aggregate"]
            rows.append(
                {
                    "run": run_dir.name,
                    "dataset": metrics["dataset"],
                    "validity": metrics["validity"],
                    "model": label,
                    "arm": arm,
                    "n_instances": aggregate["n_instances"],
                    "mean_test_accuracy": aggregate[
                        "mean_test_accuracy_over_instances"
                    ],
                    "std_over_instances": aggregate["std_test_accuracy_over_instances"],
                    "min_instance": aggregate["min_instance_mean_test_accuracy"],
                    "max_instance": aggregate["max_instance_mean_test_accuracy"],
                    "pooled_std_over_partitions": aggregate["pooled_std_test_accuracy"],
                    "paper_accuracy": paper_value(metrics["dataset"], label),
                }
            )
    return rows


def kernel_rows(run_dir: Path) -> list[dict]:
    """Return one row per U_0 instance of a platform-kernel run."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    rows = []
    for instance in metrics["instances"]:
        comparisons = dict(instance.get("vs_digitized", {}))
        if "vs_designed_kernel" in instance:
            comparisons["designed_kernel"] = instance["vs_designed_kernel"]
        for reference, comparison in comparisons.items():
            rows.append(
                {
                    "run": run_dir.name,
                    "platform": metrics["platform"],
                    "u0_strategy": instance["u0_selection"]["strategy"],
                    "u0_seed": instance["u0_seed"],
                    "reference": reference,
                    "mean_abs_error": comparison["mean_abs_error"],
                    "max_abs_error": comparison["max_abs_error"],
                    "pearson_r": comparison["pearson_r"],
                    "total_postselection_probability": instance[
                        "total_postselection_probability"
                    ],
                }
            )
        if "printed_entries" in instance:
            rows.append(
                {
                    "run": run_dir.name,
                    "platform": metrics["platform"],
                    "u0_strategy": "fixed_mesh",
                    "u0_seed": None,
                    "reference": "fig3b_printed_entries",
                    "mean_abs_error": None,
                    "max_abs_error": instance["printed_entries"]["max_abs_error"],
                    "pearson_r": None,
                    "total_postselection_probability": instance[
                        "total_postselection_probability"
                    ],
                }
            )
    return rows


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows: list[dict]) -> str:
    """Render rows as a Markdown table with rounded display values."""
    if not rows:
        return ""
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(_fmt(row[key]) for key in keys) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    classification, kernels = [], []
    for run_dir in args.run_dirs:
        experiment = json.loads((run_dir / "metrics.json").read_text())["experiment"]
        if experiment == "classification":
            classification += classification_rows(run_dir)
        elif experiment == "platform_kernel":
            kernels += kernel_rows(run_dir)
        else:
            raise ValueError(f"{run_dir}: unknown experiment {experiment!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, rows in (("classification", classification), ("kernels", kernels)):
        if rows:
            csv_path = args.out.with_name(f"{args.out.stem}_{name}.csv")
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    args.out.write_text(
        "# Summary tables\n\n## Classification\n\n"
        + markdown_table(classification)
        + "\n\n## Kernels\n\n"
        + markdown_table(kernels)
        + "\n"
    )


if __name__ == "__main__":
    main()
