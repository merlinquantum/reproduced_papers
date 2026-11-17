"""Uncertainty and paired comparisons for the Phase 03 classification runs.

Usage::

    python utils/compute_statistics.py \
        --digitized-1d outdir/run_<classify_1d_original> \
        --assignments outdir/run_<classify_moons_assignments_original> \
        --alternatives outdir/run_<alternative_kernels_1d_fitted> \
        --x-order outdir/run_<classify_moons_xorder_original> ... \
        --out results/statistics.json

Reports, from ``metrics.json`` at full precision:

- the standard error over the 100 partitions of each digitised-kernel 1D
  accuracy and its gap to the paper value, in standard errors;
- the qutrit-minus-qubit accuracy difference paired over the moons
  cluster-to-outcome assignments (random assignments only), with a paired
  t-test and a sign count (claim C3);
- the share of alternative 1D kernels inside the paper's 0.73-1.0 range (C4);
- for each ``--x-order`` run, the qutrit-minus-qubit difference paired over the
  dataset variants (experimental digitised kernels).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

PAPER_1D = {"B2": 0.90, "B3": 0.80}
PAPER_ALTERNATIVE_RANGE = (0.73, 1.0)


def paper_instances(run_dir: Path) -> dict[str, list[dict]]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return {
        label: result["arms"]["paper"]["instances"]
        for label, result in metrics["results"].items()
    }


def one_d_gaps(run_dir: Path) -> dict:
    gaps = {}
    for label, instances in paper_instances(run_dir).items():
        (instance,) = instances
        per_partition = np.asarray(instance["test_accuracy_per_partition"])
        mean = float(per_partition.mean())
        standard_error = float(per_partition.std(ddof=1) / np.sqrt(per_partition.size))
        paper = PAPER_1D[label[:2]]
        gaps[label] = {
            "mean_test_accuracy": mean,
            "standard_error": standard_error,
            "paper_accuracy": paper,
            "gap": paper - mean,
            "gap_in_standard_errors": (paper - mean) / standard_error,
        }
    return gaps


def paired_assignment_difference(run_dir: Path) -> dict:
    instances = paper_instances(run_dir)
    qubit_label = next(label for label in instances if label.startswith("B2"))
    qutrit_label = next(label for label in instances if label.startswith("B3"))

    def by_variant(label):
        return {
            (inst["dataset_config"]["assignment"], inst["dataset_config"].get(
                "assignment_seed")): inst["mean_test_accuracy"]
            for inst in instances[label]
        }  # fmt: skip

    qubit, qutrit = by_variant(qubit_label), by_variant(qutrit_label)
    if qubit.keys() != qutrit.keys():
        raise ValueError("qubit and qutrit runs cover different assignment variants")
    random_keys = sorted(key for key in qubit if key[0] == "random")
    difference = np.array([qutrit[key] - qubit[key] for key in random_keys])
    t_result = stats.ttest_rel(
        [qutrit[key] for key in random_keys], [qubit[key] for key in random_keys]
    )
    return {
        "qubit_model": qubit_label,
        "qutrit_model": qutrit_label,
        "n_random_assignments": len(random_keys),
        "mean_qubit": float(np.mean([qubit[key] for key in random_keys])),
        "mean_qutrit": float(np.mean([qutrit[key] for key in random_keys])),
        "mean_difference_qutrit_minus_qubit": float(difference.mean()),
        "std_difference": float(difference.std(ddof=1)),
        "n_qutrit_better": int(np.sum(difference > 0)),
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_pvalue": float(t_result.pvalue),
        "structured_variants": {
            f"{key[0]}": {"qubit": qubit[key], "qutrit": qutrit[key]}
            for key in qubit
            if key[0] != "random"
        },
    }


def paired_variant_difference(run_dir: Path) -> dict:
    instances = paper_instances(run_dir)
    qubit = np.array(
        [
            inst["mean_test_accuracy"]
            for inst in instances["B2_qubit_experimental_fig4a"]
        ]
    )
    qutrit = np.array(
        [
            inst["mean_test_accuracy"]
            for inst in instances["B3_qutrit_experimental_fig4b"]
        ]
    )
    difference = qutrit - qubit
    t_result = stats.ttest_rel(qutrit, qubit)
    return {
        "n_variants": int(difference.size),
        "mean_qubit": float(qubit.mean()),
        "mean_qutrit": float(qutrit.mean()),
        "mean_difference_qutrit_minus_qubit": float(difference.mean()),
        "std_difference": float(difference.std(ddof=1)),
        "n_qutrit_better": int(np.sum(difference > 0)),
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_pvalue": float(t_result.pvalue),
    }


def alternative_coverage(run_dir: Path) -> dict:
    low, high = PAPER_ALTERNATIVE_RANGE
    coverage = {}
    for label, instances in paper_instances(run_dir).items():
        accuracies = np.array([inst["mean_test_accuracy"] for inst in instances])
        coverage[label] = {
            "n_kernels": int(accuracies.size),
            "mean": float(accuracies.mean()),
            "min": float(accuracies.min()),
            "max": float(accuracies.max()),
            "fraction_in_paper_range": float(
                np.mean((accuracies >= low) & (accuracies <= high))
            ),
        }
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digitized-1d", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--alternatives", type=Path, required=True)
    parser.add_argument("--x-order", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "validity": "substitute-data",
        "one_d_digitized_vs_paper": one_d_gaps(args.digitized_1d),
        "moons_qutrit_vs_qubit_random_assignments": paired_assignment_difference(
            args.assignments
        ),
        "alternative_kernels_1d": alternative_coverage(args.alternatives),
        "moons_x_order_paired": {
            str(run): paired_variant_difference(run) for run in args.x_order
        },
        "sources": {
            "digitized_1d": str(args.digitized_1d),
            "assignments": str(args.assignments),
            "alternatives": str(args.alternatives),
            "x_order": [str(run) for run in args.x_order],
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
