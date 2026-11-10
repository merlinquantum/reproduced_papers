"""Baseline B4 (Phase 5 reference): linear models on Pauli expectations up to weight w.

For any fixed target observable O* = phi(H*), the regression target
Tr[O* rho] is linear in rho, and the paper's classification label is the sign
of a linear functional of rho. A linear model on enough Pauli expectations
<P>_rho therefore represents the target exactly (full tomography, w = n) or
approximately (w = 2). B4 is the Fair Baseline Plan's upper reference: it is
matched on data, not on parameters, and every row reports its feature count.

* Regression (squared loss): ridge regression with an unpenalized intercept,
  closed form.
* Classification (labels +-1): L2-regularized logistic regression with an
  unpenalized intercept, L-BFGS. Prediction: sign of the linear output.

Every regularization strength is fitted and evaluated; the Phase 5 rule
selects lambda later from the selection-split metric (``utils/select_p5.py``).
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def pauli_labels_up_to_weight(n_qubits: int, max_weight: int) -> list[str]:
    """Return every non-identity Pauli string on ``n_qubits`` with weight <= ``max_weight``.

    Raises
    ------
    ValueError
        If ``max_weight`` is not in [1, n_qubits].
    """
    if not 1 <= max_weight <= n_qubits:
        raise ValueError(f"max_weight must be in [1, {n_qubits}], got {max_weight}")
    labels = []
    for weight in range(1, max_weight + 1):
        for sites in itertools.combinations(range(n_qubits), weight):
            for letters in itertools.product("XYZ", repeat=weight):
                label = ["I"] * n_qubits
                for site, letter in zip(sites, letters):
                    label[site] = letter
                labels.append("".join(label))
    return labels


def _design(features: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(features), 1)), features])


def fit_ridge(features: np.ndarray, targets: np.ndarray, strength: float) -> np.ndarray:
    """Return ``[intercept, weights]`` minimizing mean squared error + strength * |w|^2."""
    design = _design(features)
    penalty = strength * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    n_states = len(targets)
    return np.linalg.solve(
        design.T @ design / n_states + penalty, design.T @ targets / n_states
    )


def logistic_loss(
    coefficients: np.ndarray, features: np.ndarray, labels: np.ndarray
) -> float:
    """Mean logistic loss log(1 + exp(-y (b + w.x)))."""
    margins = labels * (_design(features) @ coefficients)
    return float(np.mean(np.logaddexp(0.0, -margins)))


def fit_logistic(
    features: np.ndarray, labels: np.ndarray, strength: float
) -> np.ndarray:
    """Return ``[intercept, weights]`` minimizing mean logistic loss + strength / 2 * |w|^2.

    Raises
    ------
    RuntimeError
        If L-BFGS does not converge.
    """
    design = _design(features)

    def objective(coefficients):
        margins = labels * (design @ coefficients)
        value = np.mean(np.logaddexp(0.0, -margins))
        value += 0.5 * strength * np.sum(coefficients[1:] ** 2)
        weights = -labels / (1.0 + np.exp(margins))
        gradient = design.T @ weights / len(labels)
        gradient[1:] += strength * coefficients[1:]
        return value, gradient

    result = minimize(
        objective,
        np.zeros(design.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 5000},
    )
    if not result.success:
        raise RuntimeError(f"L-BFGS did not converge: {result.message}")
    return result.x


def evaluate_linear_reference(
    splits: dict,
    labels: list[str],
    terms: np.ndarray,
    task_kind: str,
    strengths: list[float],
) -> list[dict]:
    """Fit B4 for every regularization strength and evaluate every split.

    Parameters
    ----------
    splits : dict
        ``train``, ``selection`` and ``validation`` (test) datasets.
    labels, terms : list of str, numpy.ndarray
        Pauli strings and their dense matrices ``(F, d, d)``.
    task_kind : str
        ``"regression"`` or ``"classification"``.
    strengths : list of float
        Regularization grid.

    Returns
    -------
    list of dict
        One record per strength: per-split ``mse`` (regression) or
        ``logistic_loss`` and ``accuracy`` (classification).
    """
    features = {
        name: data.pauli_features(labels, terms) for name, data in splits.items()
    }
    records = []
    for strength in strengths:
        train = splits["train"]
        if task_kind == "regression":
            coefficients = fit_ridge(features["train"], train.targets, strength)
        else:
            coefficients = fit_logistic(features["train"], train.targets, strength)
        record = {"strength": strength, "n_features": len(labels)}
        for name, data in splits.items():
            outputs = _design(features[name]) @ coefficients
            if task_kind == "regression":
                record[f"{name}_mse"] = float(np.mean((outputs - data.targets) ** 2))
            else:
                record[f"{name}_logistic_loss"] = logistic_loss(
                    coefficients, features[name], data.targets
                )
                record[f"{name}_accuracy"] = float(
                    np.mean(np.where(outputs >= 0, 1.0, -1.0) == data.targets)
                )
        records.append(record)
    return records
