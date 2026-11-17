"""Explain the one accuracy mismatch between the MerLin and numpy backends (E7).

Usage::

    python utils/diagnose_merlin_svm_tie.py --out results/merlin_svm_tie.json

`results/merlin_comparison.json` reports one differing (instance, partition)
out of 25,000 in `alternative_kernels_1d_merlin`. The instance is B3, fitted U_0
pool seed 2, assignment random_35, partition 88. This script rebuilds that
instance's kernel with both backends and prints the SVM decision values on the
partition's test points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parents[1]
# The paper folder provides lib/; the repository root provides runtime_lib/.
sys.path[:0] = [str(PROJECT_DIR), str(PROJECT_DIR.parents[1])]

from lib.abs_merlin import merlin_state_table  # noqa: E402
from lib.circuits import get_platform, state_table  # noqa: E402
from lib.data import (  # noqa: E402
    digitized_kernels_path,
    make_nonlinear_1d,
    partition_indices,
)
from lib.kernels import quantum_kernel, states_for_assignment, u0_for_config  # noqa: E402

U0_POOL_SEED, PARTITION_SEED = 2, 88
ASSIGNMENT = [1, 0, 10, 9, 2, 6, 14, 13, 11, 12, 5, 3, 4, 8, 7]  # random_35 in metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(
        (PROJECT_DIR / "configs" / "alternative_kernels_1d_merlin.json").read_text()
    )
    spec = next(
        m for m in config["models"] if m["label"] == "B3_qutrit_alternative_merlin"
    )
    platform = get_platform("B3")
    u0, _ = u0_for_config(
        platform, U0_POOL_SEED, spec["u0"], digitized_kernels_path({})
    )
    labels = make_nonlinear_1d(15).labels
    train, test = partition_indices(15, 0.2, PARTITION_SEED)
    report = {"u0_pool_seed": U0_POOL_SEED, "partition_seed": PARTITION_SEED}
    kernels = {}
    for name, simulate in (("numpy", state_table), ("merlin", merlin_state_table)):
        states, _ = simulate(platform, u0)
        kernel = quantum_kernel(states_for_assignment(states, ASSIGNMENT), "overlap")
        kernels[name] = kernel
        model = SVC(kernel="precomputed", C=50).fit(
            kernel[np.ix_(train, train)], labels[train]
        )
        report[name] = {
            "decision_values": model.decision_function(
                kernel[np.ix_(test, train)]
            ).tolist(),
            "predictions": model.predict(kernel[np.ix_(test, train)]).tolist(),
        }
    report["true_labels"] = labels[test].tolist()
    report["max_abs_kernel_difference"] = float(
        np.max(np.abs(kernels["numpy"] - kernels["merlin"]))
    )
    report["libsvm_stopping_tolerance"] = 1e-3
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
