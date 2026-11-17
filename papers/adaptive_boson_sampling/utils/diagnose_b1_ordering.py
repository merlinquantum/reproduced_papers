"""Check whether any outcome ordering makes the simulated B1 kernel match Fig. 3e.

Usage::

    python utils/diagnose_b1_ordering.py --out results/b1_diagnosis.json

The B1 circuit (Fig. 3d) is fully published, but the order of the 15 outcomes
along the Fig. 3e axes is not. This script compares the ideal simulated kernel
with the digitised "theoretical" panel under:

1. the identity (lexicographic pair) order used by the runner;
2. every relabelling of the 6 detectors, in lexicographic and colexicographic
   pair order (2 x 720 orders);
3. a general search over all 15! orders (FAQ quadratic assignment, 200 random
   starts), against a Haar-random qubit-state control with the same search.

Simulated values are clipped to the digitised minimum before scoring, since the
shared Fig. 3e colorbar stops near 0.21 and the digitiser cannot read lower.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import quadratic_assignment

PROJECT_DIR = Path(__file__).resolve().parents[1]
# The paper folder provides lib/; the repository root provides runtime_lib/.
sys.path[:0] = [str(PROJECT_DIR), str(PROJECT_DIR.parents[1])]

from lib.circuits import get_platform, state_table  # noqa: E402
from lib.kernels import quantum_kernel, states_for_assignment  # noqa: E402

from runtime_lib.data_paths import paper_data_dir  # noqa: E402

UPPER = np.triu_indices(15, 1)


def score(kernel: np.ndarray, target: np.ndarray, order) -> dict[str, float]:
    """Return MAE and Pearson r of the reordered, clipped kernel vs the target."""
    floor = target[UPPER].min()
    reordered = np.clip(kernel[np.ix_(order, order)], floor, 1.0)
    return {
        "mean_abs_error": float(np.abs(reordered[UPPER] - target[UPPER]).mean()),
        "pearson_r": float(np.corrcoef(reordered[UPPER], target[UPPER])[0, 1]),
    }


def best_general_order(kernel, target, n_starts: int) -> dict:
    """Return the lowest-MAE order found by FAQ quadratic assignment."""
    centred_target = target - target[UPPER].mean()
    centred_kernel = kernel - kernel[UPPER].mean()
    best = None
    for start in range(n_starts):
        result = quadratic_assignment(
            centred_target,
            centred_kernel,
            method="faq",
            options={"maximize": True, "P0": "randomized", "rng": start},
        )
        candidate = {"order": result.col_ind.tolist()} | score(
            kernel, target, result.col_ind
        )
        if best is None or candidate["mean_abs_error"] < best["mean_abs_error"]:
            best = candidate
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-starts", type=int, default=200)
    args = parser.parse_args()

    digitized = np.load(
        paper_data_dir("adaptive_boson_sampling") / "digitized" / "figure_kernels.npz"
    )
    target = digitized["fig3e_B1_theoretical"]
    experimental = digitized["fig3e_B1_experimental"]

    platform = get_platform("B1")
    states, _ = state_table(platform, None)
    kernel = quantum_kernel(states_for_assignment(states, np.arange(15)), "overlap")

    pair_index = {
        tuple(i for i, count in enumerate(outcome) if count): position
        for position, outcome in enumerate(platform.outcomes)
    }
    lex = list(itertools.combinations(range(6), 2))
    pair_orders = {"lex": lex, "colex": sorted(lex, key=lambda p: (p[1], p[0]))}
    structured = []
    for order_name, pairs in pair_orders.items():
        for relabel in itertools.permutations(range(6)):
            order = [
                pair_index[tuple(sorted((relabel[a], relabel[b])))] for a, b in pairs
            ]
            structured.append(
                {"pair_order": order_name, "detector_relabel": list(relabel)}
                | score(kernel, target, order)
            )
    structured.sort(key=lambda row: row["mean_abs_error"])

    rng = np.random.default_rng(0)
    control_states = rng.normal(size=(15, 2)) + 1j * rng.normal(size=(15, 2))
    control_states /= np.linalg.norm(control_states, axis=1, keepdims=True)
    control_kernel = np.abs(control_states.conj() @ control_states.T) ** 2

    report = {
        "validity": "simulation of the published B1 circuit vs digitised Fig. 3e",
        "digitised_floor": float(target[UPPER].min()),
        "identity_order": score(kernel, target, np.arange(15)),
        "best_structured_orders": structured[:5],
        "best_general_order": best_general_order(kernel, target, args.n_starts),
        "random_qubit_control_general_order": best_general_order(
            control_kernel, target, args.n_starts
        ),
        "paper_theoretical_vs_experimental": {
            "mean_abs_error": float(np.abs(target[UPPER] - experimental[UPPER]).mean()),
            "pearson_r": float(np.corrcoef(target[UPPER], experimental[UPPER])[0, 1]),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "best_structured_orders"}
                     | {"best_structured": structured[0]}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
