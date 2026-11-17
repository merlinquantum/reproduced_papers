"""Sensitivity of the 1D accuracy of the digitised Fig. 4 kernels (E6 diagnosis).

Usage::

    python utils/digitization_sensitivity_1d.py --out results/digitization_sensitivity_1d.json

For each digitised kernel, reports the mean test accuracy over 100 partitions:

1. under several partition protocols (the runner's own seeds, sklearn
   ``train_test_split`` with ``random_state`` 0-99, stratified, a single global
   RNG) and C in {1, 50, 1000};
2. under symmetric Gaussian perturbations of the off-diagonal entries with
   sigma 0.01, 0.02, 0.04 (100 draws each, projected back to PSD), which is the
   size of the digitisation error. Partitions are sklearn seeds 0-99 and C = 50.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parents[1]
# The paper folder provides lib/; the repository root provides runtime_lib/.
sys.path[:0] = [str(PROJECT_DIR), str(PROJECT_DIR.parents[1])]

from lib.data import make_nonlinear_1d, partition_indices  # noqa: E402

from runtime_lib.data_paths import paper_data_dir  # noqa: E402

KEYS = {
    "fig4a_B2_experimental": 0.90,
    "fig4a_B2_simulated": 0.90,
    "fig4b_B3_experimental": 0.80,
    "fig4b_B3_simulated": 0.80,
}
N_POINTS, N_PARTITIONS = 15, 100


def partitions(
    protocol: str, labels: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    points = np.arange(N_POINTS)
    if protocol == "runner_seeds":
        return [partition_indices(N_POINTS, 0.2, s) for s in range(N_PARTITIONS)]
    if protocol == "sklearn_seeds":
        return [
            tuple(np.asarray(part) for part in train_test_split(
                points, test_size=0.2, random_state=s))
            for s in range(N_PARTITIONS)
        ]  # fmt: skip
    if protocol == "sklearn_stratified":
        return [
            tuple(np.asarray(part) for part in train_test_split(
                points, test_size=0.2, random_state=s, stratify=labels))
            for s in range(N_PARTITIONS)
        ]  # fmt: skip
    if protocol == "global_rng":
        rng = np.random.RandomState(0)
        orders = [rng.permutation(N_POINTS) for _ in range(N_PARTITIONS)]
        return [(order[3:], order[:3]) for order in orders]
    raise ValueError(protocol)


def mean_accuracy(kernel, labels, splits, c_value) -> float:
    accuracies = []
    for train, test in splits:
        model = SVC(kernel="precomputed", C=c_value)
        model.fit(kernel[np.ix_(train, train)], labels[train])
        predictions = model.predict(kernel[np.ix_(test, train)])
        accuracies.append(np.mean(predictions == labels[test]))
    return float(np.mean(accuracies))


def perturbed(kernel, sigma, rng) -> np.ndarray:
    noise = rng.normal(0, sigma, kernel.shape)
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0)
    values, vectors = np.linalg.eigh(np.clip(kernel + noise, 0, 1))
    return (vectors * np.clip(values, 0, None)) @ vectors.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    digitized = np.load(
        paper_data_dir("adaptive_boson_sampling") / "digitized" / "figure_kernels.npz"
    )
    labels = make_nonlinear_1d(N_POINTS).labels
    protocols = ["runner_seeds", "sklearn_seeds", "sklearn_stratified", "global_rng"]
    split_sets = {name: partitions(name, labels) for name in protocols}

    report = {"validity": "substitute-data (digitised paper kernels)", "kernels": {}}
    for key, paper in KEYS.items():
        kernel = (digitized[key] + digitized[key].T) / 2
        protocol_grid = {
            f"{name}/C={c}": mean_accuracy(kernel, labels, split_sets[name], c)
            for name in protocols
            for c in (1, 50, 1000)
        }
        rng = np.random.default_rng(0)
        perturbation = {}
        for sigma in (0.01, 0.02, 0.04):
            values = np.array([
                mean_accuracy(perturbed(kernel, sigma, rng), labels,
                              split_sets["sklearn_seeds"], 50)
                for _ in range(100)
            ])  # fmt: skip
            perturbation[str(sigma)] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
                "fraction_at_or_above_paper": float(np.mean(values >= paper - 0.005)),
            }
        report["kernels"][key] = {
            "paper_accuracy": paper,
            "protocol_grid": protocol_grid,
            "perturbation_sklearn_seeds_C50": perturbation,
        }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for key, entry in report["kernels"].items():
        print(key, entry["perturbation_sklearn_seeds_C50"]["0.01"])


if __name__ == "__main__":
    main()
