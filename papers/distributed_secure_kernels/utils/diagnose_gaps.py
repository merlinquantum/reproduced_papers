"""Diagnostics for the two large gaps (Fig. 3 noise, Fig. 4 shots); writes results/gap_diagnostics.json.

Noise (Fig. 3):
  (a) a fully depolarized kernel (p0 = 1/2 for every pair, diagonal included)
      plus 1,024 shots: does it give the authors' Framingham kernel variance
      (0.000445 at both noise levels) and their 0.58 accuracy?
  (b) gate counts if ``initialize`` is decomposed into the noisy basis (older
      Qiskit behaviour) and the resulting ideal contraction (1 - rate)^gates.
Shots (Fig. 4):
  (c) spread of the 5-fold CV accuracy over the fold assignment
      (``StratifiedKFold`` random_state 0-19, and unshuffled folds), for each
      shot count and seed, on the authors' preprocessing.

Usage: ``.venv/bin/python papers/distributed_secure_kernels/utils/diagnose_gaps.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR.parent), str(PROJECT_DIR.parents[1])]

import numpy as np  # noqa: E402
from qiskit import transpile  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from distributed_secure_kernels.lib.data import build_dataset  # noqa: E402
from distributed_secure_kernels.lib.encoding import amplitude_encode  # noqa: E402
from distributed_secure_kernels.lib.evaluation import (  # noqa: E402
    paper_kernel_svm,
    stratified_folds,
)
from distributed_secure_kernels.lib.kernels import (  # noqa: E402
    estimate_from_zero_probability,
    finite_shot_kernel,
)
from distributed_secure_kernels.lib.noise import depolarizing_noise_model  # noqa: E402
from distributed_secure_kernels.lib.protocol import distributed_circuit  # noqa: E402
from runtime_lib.config import deep_update, load_config  # noqa: E402

AUTHORS_FRAMINGHAM_NOISY_VARIANCE = (
    0.000445  # heart_dataset_kernel_1060336.txt, levels 1 and 2
)
AUTHORS_FRAMINGHAM_NOISY_ACCURACY = (0.5807, 0.5814)


def _config(name: str) -> dict:
    defaults = load_config(PROJECT_DIR / "configs" / "defaults.json")
    return deep_update(defaults, load_config(PROJECT_DIR / "configs" / name))


def depolarized_kernel_check(seeds=range(5)) -> dict:
    """(a) Kernel of a fully depolarized server: every entry from p0 = 1/2."""

    results = []
    for seed in seeds:
        cfg = _config("accuracy_framingham_original.json")
        cfg["seed"] = seed
        data = build_dataset(cfg)
        n = len(data.labels)
        rng = np.random.default_rng(seed)
        kernel = estimate_from_zero_probability(np.full((n, n), 0.5), 1024, rng, True)
        folds = stratified_folds(data.labels, 5, 0)
        accuracy = paper_kernel_svm(kernel, data.labels, folds, "depolarized")[
            "accuracy"
        ]
        results.append(
            {
                "seed": seed,
                "variance": float(kernel.var()),
                "accuracy": accuracy["mean"],
            }
        )
    return {
        "authors_variance": AUTHORS_FRAMINGHAM_NOISY_VARIANCE,
        "authors_accuracy": AUTHORS_FRAMINGHAM_NOISY_ACCURACY,
        "per_seed": results,
        "variance_mean": float(np.mean([r["variance"] for r in results])),
        "accuracy_mean": float(np.mean([r["accuracy"] for r in results])),
    }


def decomposed_initialize_check(rates=(0.001, 0.01)) -> dict:
    """(b) Noisy gate counts with ``initialize`` decomposed, and the ideal contraction."""

    out = {}
    for n_qubits, dataset in ((4, "framingham"), (5, "parkinsons")):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=2**n_qubits), rng.normal(size=2**n_qubits)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        circuit = distributed_circuit(a, b)
        basis = ["u1", "u2", "u3", "cx", "measure", "reset", "if_else"]
        # As the authors run it here (Qiskit 2.5): transpiled against the noisy
        # simulator, ``initialize`` stays native and noiseless.
        native = transpile(
            circuit, AerSimulator(noise_model=depolarizing_noise_model(0.001))
        ).count_ops()
        decomposed = transpile(circuit, basis_gates=basis).count_ops()
        gates = {
            name: int(sum(c for g, c in ops.items() if g in ("u1", "u2", "u3", "cx")))
            for name, ops in (
                ("initialize_native", native),
                ("initialize_decomposed", decomposed),
            )
        }
        cx = {
            name: int(ops.get("cx", 0))
            for name, ops in (
                ("initialize_native", native),
                ("initialize_decomposed", decomposed),
            )
        }
        out[dataset] = {
            "qubits_per_party": n_qubits,
            "noisy_gates_outside_if_blocks": gates,
            "cx_outside_if_blocks": cx,
            "ideal_survival_(1-rate)^gates": {
                str(rate): {
                    name: float((1 - rate) ** count) for name, count in gates.items()
                }
                for rate in rates
            },
        }
    return out


def fold_assignment_check(seeds=range(5), fold_states=range(20)) -> dict:
    """(c) CV accuracy spread over fold assignments on the Digits subset."""

    cfg = _config("shots_digits_original.json")
    data = build_dataset(cfg)
    out = {}
    for shots in (128, 256, 512, 1024, None):
        values, unshuffled = [], []
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1, 1]))
            if shots is None:
                states = amplitude_encode(data.features)
                kernel = (states @ states.T) ** 2
            else:
                kernel = finite_shot_kernel(
                    data.features, data.features, shots, rng, True
                )
            for state in fold_states:
                folds = stratified_folds(data.labels, 5, state)
                values.append(
                    paper_kernel_svm(kernel, data.labels, folds, "diag")["accuracy"][
                        "mean"
                    ]
                )
            plain = list(
                StratifiedKFold(5).split(np.zeros(len(data.labels)), data.labels)
            )
            scores = []
            for train, test in plain:
                model = SVC(kernel="precomputed").fit(
                    kernel[np.ix_(train, train)], data.labels[train]
                )
                scores.append(
                    float(
                        np.mean(
                            model.predict(kernel[np.ix_(test, train)])
                            == data.labels[test]
                        )
                    )
                )
            unshuffled.append(float(np.mean(scores)))
        key = "exact" if shots is None else str(shots)
        out[key] = {
            "shuffled_folds": {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p05": float(np.quantile(values, 0.05)),
                "p95": float(np.quantile(values, 0.95)),
            },
            "unshuffled_folds_per_seed": unshuffled,
        }
    return out


def main() -> int:
    import logging

    logging.disable(logging.INFO)
    report = {
        "depolarized_kernel": depolarized_kernel_check(),
        "decomposed_initialize": decomposed_initialize_check(),
        "fold_assignment": fold_assignment_check(),
    }
    path = PROJECT_DIR / "results" / "gap_diagnostics.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
