"""Exploratory grid of parameter-free cluster-to-outcome rules for moons (E8).

Usage::

    python utils/explore_moons_assignments.py --out results/moons_assignment_rules.json

The paper does not state how its 15 K-means clusters map to the 15 outcomes.
This grid ranks clusters by a scalar of their centroid (x, -x, y, -y, x + y,
x - y, angle about (0.5, 0.25)) and reports the SVM accuracy (C = 50, sklearn
``train_test_split`` seeds 0-29) of the four digitised Fig. 4 kernels, over
noise 0.1 / 0.15 / 0.2 and 3 data seeds x 3 K-means seeds. It is the exploration
that singled out ``x_order``; every rule tried is recorded here, so the choice
is visible as post hoc.
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

from lib.data import make_moons_kmeans  # noqa: E402

from runtime_lib.data_paths import paper_data_dir  # noqa: E402

RULES = {
    "x": lambda c: c[:, 0],
    "-x": lambda c: -c[:, 0],
    "y": lambda c: c[:, 1],
    "-y": lambda c: -c[:, 1],
    "x+y": lambda c: c[:, 0] + c[:, 1],
    "x-y": lambda c: c[:, 0] - c[:, 1],
    "angle": lambda c: np.arctan2(c[:, 1] - 0.25, c[:, 0] - 0.5),
}
KEYS = [
    "fig4a_B2_experimental",
    "fig4a_B2_simulated",
    "fig4b_B3_experimental",
    "fig4b_B3_simulated",
]


def accuracy(kernel, outcome_index, labels, n_partitions=30) -> float:
    gram = kernel[np.ix_(outcome_index, outcome_index)]
    scores = []
    for seed in range(n_partitions):
        train, test = train_test_split(
            np.arange(len(labels)), test_size=0.2, random_state=seed
        )
        model = SVC(kernel="precomputed", C=50).fit(
            gram[np.ix_(train, train)], labels[train]
        )
        scores.append(np.mean(model.predict(gram[np.ix_(test, train)]) == labels[test]))
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    digitized = np.load(
        paper_data_dir("adaptive_boson_sampling") / "digitized" / "figure_kernels.npz"
    )
    rows = []
    for noise in (0.1, 0.15, 0.2):
        for rule, score in RULES.items():
            per_variant = []
            for data_seed in range(3):
                for kmeans_seed in range(3):
                    dataset = make_moons_kmeans({
                        "n_clusters": 15, "n_samples": 200, "noise": noise,
                        "data_seed": data_seed, "kmeans_seed": kmeans_seed,
                        "assignment": "kmeans_label", "assignment_seed": 0,
                    })  # fmt: skip
                    rank = np.empty(15, dtype=np.int64)
                    rank[
                        np.argsort(score(dataset.extras["centroids"]), kind="stable")
                    ] = np.arange(15)
                    outcome_index = rank[dataset.extras["cluster"]]
                    per_variant.append(
                        [accuracy(digitized[key], outcome_index, dataset.labels)
                         for key in KEYS]
                    )  # fmt: skip
            values = np.array(per_variant)
            rows.append({
                "noise": noise, "rule": rule,
                **{f"{key}_mean": float(values[:, i].mean()) for i, key in enumerate(KEYS)},
                **{f"{key}_std": float(values[:, i].std()) for i, key in enumerate(KEYS)},
            })  # fmt: skip
            print(noise, rule, np.round(values.mean(axis=0), 3))
    report = {
        "validity": "substitute-data (digitised paper kernels, regenerated moons)",
        "paper": {"qubit": 0.65, "qutrit": 0.90},
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
