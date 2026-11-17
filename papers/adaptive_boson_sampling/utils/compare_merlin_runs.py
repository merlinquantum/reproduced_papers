"""Compare Phase 4 MerLin runs with their Phase 3 numpy counterparts.

Usage::

    python utils/compare_merlin_runs.py \
        --kernel-pair outdir/<merlin kernel run> outdir/<phase 3 kernel run> ... \
        --classification-pair outdir/<merlin run> outdir/<phase 3 run> ... \
        --noisy outdir/<noisy kernel run> ... \
        --out results/merlin_comparison.json

Pass criteria are fixed by the Phase 4 adaptation plan in ``LOG.md``.

- **Kernels.** Every kernel array shared by the two ``kernels.npz`` files agrees
  to ``KERNEL_TOLERANCE``.
- **Classification.** Every Phase 3 model has a MerLin twin labelled
  ``<label>_merlin``. Their per-instance mean test accuracies agree to
  ``ACCURACY_TOLERANCE`` for every arm.

Noisy runs are summarised (Pearson r and MAE against each digitised panel, kernel
minimum and mean), with no pass/fail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KERNEL_TOLERANCE = 1e-9
ACCURACY_TOLERANCE = 1e-12


def compare_kernel_runs(merlin_run: Path, reference_run: Path) -> dict:
    merlin = np.load(merlin_run / "kernels.npz")
    reference = np.load(reference_run / "kernels.npz")
    keys = sorted(
        key
        for key in set(merlin.files) & set(reference.files)
        if not key.startswith("paper_")
    )
    if not keys:
        raise ValueError(f"{merlin_run} and {reference_run} share no kernel arrays")
    differences = {
        key: float(np.max(np.abs(merlin[key] - reference[key]))) for key in keys
    }
    worst = max(differences.values())
    return {
        "merlin_run": str(merlin_run),
        "reference_run": str(reference_run),
        "max_abs_difference": differences,
        "tolerance": KERNEL_TOLERANCE,
        "pass": worst <= KERNEL_TOLERANCE,
    }


def _accuracies(metrics: dict, label: str, arm: str) -> np.ndarray:
    return np.array(
        [
            instance["mean_test_accuracy"]
            for instance in metrics["results"][label]["arms"][arm]["instances"]
        ]
    )


def compare_classification_runs(merlin_run: Path, reference_run: Path) -> dict:
    merlin = json.loads((merlin_run / "metrics.json").read_text())
    reference = json.loads((reference_run / "metrics.json").read_text())
    rows, passed = [], True
    for label, result in reference["results"].items():
        twin = f"{label}_merlin"
        if twin not in merlin["results"]:
            raise ValueError(f"{merlin_run} has no MerLin twin {twin!r}")
        for arm in result["arms"]:
            ours = _accuracies(merlin, twin, arm)
            theirs = _accuracies(reference, label, arm)
            difference = float(np.max(np.abs(ours - theirs)))
            passed &= difference <= ACCURACY_TOLERANCE
            rows.append(
                {
                    "model": label,
                    "arm": arm,
                    "n_instances": int(ours.size),
                    "mean_test_accuracy_merlin": float(ours.mean()),
                    "mean_test_accuracy_numpy": float(theirs.mean()),
                    "max_abs_instance_difference": difference,
                }
            )
    return {
        "merlin_run": str(merlin_run),
        "reference_run": str(reference_run),
        "rows": rows,
        "tolerance": ACCURACY_TOLERANCE,
        "pass": bool(passed),
    }


def summarise_noisy_run(run_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    upper = None
    rows = []
    for instance in metrics["instances"]:
        kernel = np.asarray(instance["kernel"])
        upper = upper or np.triu_indices(len(kernel), 1)
        rows.append(
            {
                "u0_seed": instance["u0_seed"],
                "simulation": instance["simulation"],
                "kernel_min": float(kernel[upper].min()),
                "kernel_mean": float(kernel[upper].mean()),
                "vs_digitized": instance["vs_digitized"],
            }
        )
    return {"run": str(run_dir), "platform": metrics["platform"], "instances": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel-pair", nargs=2, action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--classification-pair", nargs=2, action="append", type=Path, default=[]
    )
    parser.add_argument("--noisy", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "validity": "MerLin adaptation (Phase 4) vs Phase 3 numpy reference",
        "kernels": [compare_kernel_runs(a, b) for a, b in args.kernel_pair],
        "classification": [
            compare_classification_runs(a, b) for a, b in args.classification_pair
        ],
        "noisy": [summarise_noisy_run(run) for run in args.noisy],
    }
    report["exact_backend_preserves_phase3"] = all(
        entry["pass"] for entry in report["kernels"] + report["classification"]
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("exact backend preserves Phase 3:", report["exact_backend_preserves_phase3"])
    for entry in report["kernels"]:
        print("kernel", entry["merlin_run"], max(entry["max_abs_difference"].values()))
    for entry in report["classification"]:
        print("classification", entry["merlin_run"], entry["pass"])


if __name__ == "__main__":
    main()
