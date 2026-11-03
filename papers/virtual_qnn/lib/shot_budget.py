"""Finite-shot statistics of the quasi-probability (virtual) estimator, paper Eq. (6).

For a ±1-valued observable P measured on a virtual state σ = Σ_i c_i ρ_i with
γ = Σ_i |c_i|, each shot samples branch i with probability |c_i|/γ, measures
P on ρ_i (outcome o = ±1) and records X = γ·sign(c_i)·o. Then

    E[X] = Σ_i c_i Tr[P ρ_i] = Tr[P σ]      (unbiased)
    E[X²] = γ²                               (every sample has |X| = γ)

so the mean of S shots has variance (γ² − Tr[Pσ]²)/S. With γ = 1 this is
ordinary single-circuit estimation. These identities are the basis of the
Phase 5 equal-shot comparison (LOG.md, "Phase 5 Comparison Plan").
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}


def pauli_decomposition(
    matrix: np.ndarray, tolerance: float = 1e-12
) -> dict[str, float]:
    """Return ``{label: h}`` with ``matrix = Σ h · P_label`` over n-qubit Paulis.

    Raises
    ------
    ValueError
        If the matrix is not Hermitian (complex coefficients).
    """
    n_qubits = int(round(math.log2(matrix.shape[0])))
    terms = {}
    for labels in itertools.product("IXYZ", repeat=n_qubits):
        operator = _PAULI[labels[0]]
        for label in labels[1:]:
            operator = np.kron(operator, _PAULI[label])
        coefficient = np.trace(operator @ matrix) / 2**n_qubits
        if abs(coefficient.imag) > 1e-9:
            raise ValueError("Matrix is not Hermitian: complex Pauli coefficient")
        if abs(coefficient.real) > tolerance:
            terms["".join(labels)] = float(coefficient.real)
    return terms


def pauli_expectation(label: str, sigma: np.ndarray) -> float:
    """``Tr[P σ]`` for an n-qubit Pauli label such as ``"XZ"``."""
    operator = _PAULI[label[0]]
    for character in label[1:]:
        operator = np.kron(operator, _PAULI[character])
    return float(np.real(np.trace(operator @ sigma)))


def energy_estimator_rmse(
    hamiltonian: np.ndarray,
    sigma: np.ndarray,
    budget: float,
    total_shots: float,
    reference: float,
) -> dict[str, float]:
    """RMSE of the virtual energy estimator with ``total_shots`` split evenly over terms.

    Parameters
    ----------
    hamiltonian : numpy.ndarray
        Hermitian matrix ``H = Σ_k h_k P_k``.
    sigma : numpy.ndarray
        Virtual (or physical, ``budget = 1``) state.
    budget : float
        Sampling overhead ``γ = ‖c‖₁``.
    total_shots : float
        Total shots for one energy estimate; ``math.inf`` gives the exact value.
    reference : float
        Reference energy (exact ground-state energy) defining the bias.

    Returns
    -------
    dict[str, float]
        ``mean`` (Tr[Hσ]), ``bias``, ``variance``, ``rmse``.
    """
    terms = pauli_decomposition(hamiltonian)
    identity_label = "I" * len(next(iter(terms)))
    measured = {label: h for label, h in terms.items() if label != identity_label}
    mean = float(np.real(np.trace(hamiltonian @ sigma)))
    if math.isinf(total_shots):
        variance = 0.0
    else:
        shots_per_term = total_shots / len(measured)
        variance = sum(
            h**2 * (budget**2 - pauli_expectation(label, sigma) ** 2) / shots_per_term
            for label, h in measured.items()
        )
    bias = mean - reference
    return {
        "mean": mean,
        "bias": bias,
        "variance": variance,
        "rmse": math.sqrt(bias**2 + variance),
    }


def expected_accuracy(
    combined_expectation: np.ndarray, labels: np.ndarray, budget: float, shots: float
) -> float:
    """Expected accuracy of ``label = 1 iff m >= 0`` with ``m`` the mean of ``shots`` samples.

    ``combined_expectation`` is μ = c1 z1 + c2 z2 per point; each sample is ±γ
    with ``P(+γ) = (1 + μ/γ)/2``. Exact binomial probabilities; ties (m = 0)
    count as label 1. ``shots = math.inf`` uses the sign of μ.
    """
    mu = np.asarray(combined_expectation, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if math.isinf(shots):
        predicted_one = (mu >= 0).astype(float)
        return float(np.mean(predicted_one == labels))
    shots = int(shots)
    q = np.clip((1.0 + mu / budget) / 2.0, 0.0, 1.0)
    # m >= 0  <=>  number of +γ outcomes >= shots / 2.
    threshold = math.ceil(shots / 2)
    p_one = stats.binom.sf(threshold - 1, shots, q)
    p_correct = np.where(labels == 1.0, p_one, 1.0 - p_one)
    return float(np.mean(p_correct))
