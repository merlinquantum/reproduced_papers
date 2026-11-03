"""Constructive universality of App. C (experiment E5).

The theorem's model is ``f_m(x) = <Gamma| S(x)^dagger M S(x) |Gamma>`` with
``S(x) = exp(-i x H_m)``. The proof fixes ``|Gamma>`` to the uniform
superposition and reads every Fourier coefficient off one entry of ``M``.

Two notes recorded in LOG.md:

- The paper's family ``H_m = sum_i sigma^(i)`` (Eq. 28) has eigenvalue
  differences in steps of 2, so it only reaches even frequencies. The claim
  ``m = K`` holds for the Pauli-rotation generator of Section III,
  ``H_m = (1/2) sum_i Z_i``, which is used here.
- With ``S(x) = exp(-i x H)``, entry ``M_jk`` multiplies
  ``exp(i x (lambda_j - lambda_k))``. App. A writes ``lambda_k - lambda_j``;
  the sign only relabels pairs. The pairs below are chosen from the sign the
  simulation actually produces, and the tests check it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pauli_family_eigenvalues(n_qubits: int) -> np.ndarray:
    """Return the diagonal of ``H_m = (1/2) sum_i Z_i`` in the computational basis.

    Basis state ``j`` with ``p`` ones has eigenvalue ``(m - 2p) / 2``, so the
    frequency spectrum is ``{-m, ..., m}`` (paper Section III.B).
    """
    ones_count = np.array([bin(index).count("1") for index in range(2**n_qubits)])
    return (n_qubits - 2 * ones_count) / 2


def universal_observable(
    constant: float, coefficients: Sequence[complex], n_qubits: int
) -> np.ndarray:
    """Build the proof's observable for a target truncated Fourier series.

    For frequency ``n >= 1`` the pair ``(j, k) = (0, 2^n - 1)`` has
    ``lambda_j - lambda_k = n``. Setting ``M_jk = 2^m c_n`` (and its Hermitian
    partner) contributes ``c_n exp(inx) + conj(c_n) exp(-inx)``; ``M_00 = 2^m c_0``
    contributes the constant.

    Parameters
    ----------
    constant : float
        Real coefficient ``c_0``.
    coefficients : Sequence[complex]
        ``c_1 ... c_K``.
    n_qubits : int
        Number of subsystems ``m``. The construction needs ``m >= K``.

    Returns
    -------
    np.ndarray
        Hermitian observable of shape ``(2^m, 2^m)``.

    Raises
    ------
    ValueError
        If ``K > m``: the spectrum ``{-m..m}`` cannot contain frequency ``K``.
    """
    degree = len(coefficients)
    if degree > n_qubits:
        raise ValueError(f"degree K={degree} needs at least K qubits, got m={n_qubits}")
    dimension = 2**n_qubits
    observable = np.zeros((dimension, dimension), dtype=complex)
    observable[0, 0] = dimension * constant
    for frequency, coefficient in enumerate(coefficients, start=1):
        column = (1 << frequency) - 1
        observable[0, column] = dimension * coefficient
        observable[column, 0] = dimension * np.conj(coefficient)
    return observable


def universal_model_values(
    inputs: np.ndarray, observable: np.ndarray, n_qubits: int
) -> np.ndarray:
    """Evaluate ``f_m(x)`` by explicit state evolution from the uniform superposition.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of shape ``(N,)``.
    observable : np.ndarray
        Hermitian ``M`` of shape ``(2^m, 2^m)``.
    n_qubits : int
        Number of qubits ``m``.

    Returns
    -------
    np.ndarray
        Real values of shape ``(N,)``.
    """
    eigenvalues = pauli_family_eigenvalues(n_qubits)
    uniform = np.full(2**n_qubits, 2 ** (-n_qubits / 2), dtype=complex)
    states = uniform[None, :] * np.exp(-1j * inputs[:, None] * eigenvalues[None, :])
    values = np.einsum("ni,ij,nj->n", states.conj(), observable, states)
    return values.real


def truncated_series_values(
    inputs: np.ndarray, constant: float, coefficients: Sequence[complex]
) -> np.ndarray:
    """Return ``c_0 + sum_n (c_n exp(inx) + conj(c_n) exp(-inx))``."""
    values = np.full(inputs.shape, constant, dtype=complex)
    for frequency, coefficient in enumerate(coefficients, start=1):
        phase = np.exp(1j * frequency * inputs)
        values += coefficient * phase + np.conj(coefficient) * phase.conj()
    return values.real


def square_wave_coefficients(degree: int) -> tuple[float, list[complex]]:
    """Return the Fourier coefficients of ``sign(sin x)`` up to ``degree``.

    ``sign(sin x) = (4 / pi) sum_{n odd} sin(nx) / n``, so ``c_n = 2 / (i pi n)``
    for odd ``n`` and 0 otherwise; ``c_0 = 0``.
    """
    coefficients = [
        2 / (1j * np.pi * n) if n % 2 == 1 else 0j for n in range(1, degree + 1)
    ]
    return 0.0, coefficients


def square_wave_tail_rms(degree: int) -> float:
    """Return the exact RMS distance between ``sign(sin x)`` and its degree-``K`` truncation.

    By Parseval, ``mean(g^2) = 1 = sum_n |c_n|^2``, so the truncation error is
    ``sqrt(1 - c_0^2 - 2 sum_{n=1}^{K} |c_n|^2)``.
    """
    constant, coefficients = square_wave_coefficients(degree)
    retained = constant**2 + 2 * sum(abs(c) ** 2 for c in coefficients)
    return float(np.sqrt(max(0.0, 1.0 - retained)))
