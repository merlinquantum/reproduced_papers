"""Local eigentasks, NSR cutoff, REC and the paper's feature scalings.

Paper App. "Local eigentasks analysis" (after Hu et al. 2023): for each subset ``S`` the
generalized eigenproblem ``V h = beta^2 G h`` is solved with

- ``V = E_u[diag(p) - p p^T]``, the single-shot multinomial covariance of the POVM
  frequencies (so ``beta^2 / S`` is the noise-to-signal ratio at ``S`` shots);
- ``G = E_u[p p^T]``, the signal Gram matrix.

Both are estimated from the finite-shot training frequencies with the unbiased
corrections ``V_hat = S/(S-1) * mean(diag(f) - f f^T)`` and
``G_hat = mean(f f^T) - V_hat / S``. The POVM of a ``k``-qubit subset spans a
``4**k``-dimensional operator space, so ``G`` has rank ``4**k`` and the pencil has ``4**k``
finite eigenvalues. The constant function (``1^T f = 1``) has ``beta^2 = 0`` exactly and is
dropped because the ridge readout carries its own bias, leaving ``4**k - 1`` eigentasks
per subset (3 per qubit, 15 per pair; 930 = 62 x 15 at 124 qubits in the paper).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.linalg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EigentaskBasis:
    """Eigentasks of every subset, concatenated.

    Attributes
    ----------
    coefficients : list of numpy.ndarray
        Per subset, shape ``(n_effects, n_eigentasks)``; column ``l`` is ``h^(l)``.
    nsr : numpy.ndarray
        ``beta_l^2`` of every eigentask, concatenated over subsets.
    subset_index : numpy.ndarray
        Subset of each eigentask.
    """

    coefficients: list[np.ndarray]
    nsr: np.ndarray
    subset_index: np.ndarray

    def transform(self, frequencies: np.ndarray) -> np.ndarray:
        """Eigentask features ``y_l = h_l . f`` for frequencies ``(n_samples, n_subsets, n_effects)``."""
        features = [
            frequencies[:, subset] @ coeff
            for subset, coeff in enumerate(self.coefficients)
        ]
        return np.concatenate(features, axis=1)


def subset_eigentasks(
    train_frequencies: np.ndarray, shots: int, operator_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the eigentask problem of one subset.

    Parameters
    ----------
    train_frequencies : numpy.ndarray
        Finite-shot frequencies, shape ``(n_train, n_effects)``.
    shots : int
        Shots per sample used to obtain the frequencies.
    operator_dim : int
        ``4**k`` for a ``k``-qubit subset (rank of the POVM span).

    Returns
    -------
    tuple of numpy.ndarray
        ``(coefficients (n_effects, operator_dim - 1), nsr (operator_dim - 1,))`` sorted
        by increasing NSR; ``nsr`` is ``inf`` for directions below the noise floor.
    """
    if shots < 2:
        raise ValueError("Eigentask estimation needs at least 2 shots per sample")
    n_train, n_effects = train_frequencies.shape
    second_moment = train_frequencies.T @ train_frequencies / n_train
    noise = (
        (np.diag(train_frequencies.mean(axis=0)) - second_moment) * shots / (shots - 1)
    )
    gram = second_moment - noise / shots

    # The constant h = 1 solves the pencil with beta^2 = 0 (V 1 = 0). Writing
    # h = Q a + c 1 with Q an orthonormal basis of 1-perp, the remaining finite eigenpairs
    # solve Q^T V Q a = beta^2 G_c a with G_c the Schur complement of G on the constant
    # (the input covariance of the eigentask) and c = -1^T G Q a / 1^T G 1. Components of
    # h with no signal (null space of G) stay free, so each eigentask uses the
    # noise-minimising estimator of its function, as the full pencil V h = beta^2 G h requires.
    ones = np.ones(n_effects)
    basis = scipy.linalg.null_space(ones[None, :])
    gram_ones = gram @ ones
    gram_constant = ones @ gram_ones
    gram_centred = (
        basis.T @ gram @ basis
        - np.outer(basis.T @ gram_ones, basis.T @ gram_ones) / gram_constant
    )
    noise_reduced = basis.T @ noise @ basis
    # Relative 1e-12 diagonal guard only: Q^T V Q is positive definite unless an effect is
    # never observed in the whole training set.
    noise_reduced += (
        1e-12
        * np.trace(noise_reduced)
        / noise_reduced.shape[0]
        * np.eye(noise_reduced.shape[0])
    )

    n_tasks = operator_dim - 1
    size = noise_reduced.shape[0]
    snr, vectors = scipy.linalg.eigh(
        (gram_centred + gram_centred.T) / 2,
        (noise_reduced + noise_reduced.T) / 2,
        subset_by_index=[size - n_tasks, size - 1],
    )
    order = np.argsort(snr)[::-1]
    snr, vectors = snr[order], vectors[:, order]
    # With the unbiased G estimate, a direction whose signal is below the shot-noise
    # floor gets snr <= 0: it is kept as an eigentask with beta^2 = inf (removed by any
    # NSR cutoff, zero weight under NSR scaling, zero REC), matching the paper's
    # "all eigentasks" (ET) readout size.
    resolvable = snr > 0
    if not resolvable.all():
        logger.debug(
            "Eigentask subset has %d signal directions out of %d",
            resolvable.sum(),
            n_tasks,
        )
    # scipy normalises a^T V a = 1; rescale to the G-normalisation h^T G h = 1 (Hu et al.).
    scale = np.where(resolvable, 1.0 / np.sqrt(np.where(resolvable, snr, 1.0)), 1.0)
    vectors = vectors * scale
    constant_part = -(gram_ones @ basis @ vectors) / gram_constant
    coefficients = basis @ vectors + np.outer(ones, constant_part)
    nsr = np.where(resolvable, 1.0 / np.where(resolvable, snr, 1.0), np.inf)
    return coefficients, nsr


def fit_local_eigentasks(
    train_frequencies: np.ndarray, shots: int, subset_size: int
) -> EigentaskBasis:
    """Fit eigentasks for every subset of one collection (S1 or S2).

    Parameters
    ----------
    train_frequencies : numpy.ndarray
        Shape ``(n_train, n_subsets, n_effects)``.
    shots : int
        Shots per sample.
    subset_size : int
        1 for single qubits, 2 for disjoint pairs.

    Returns
    -------
    EigentaskBasis
        Concatenated eigentasks of all subsets.
    """
    coefficients, nsr_values, subset_ids = [], [], []
    for subset in range(train_frequencies.shape[1]):
        coeff, nsr = subset_eigentasks(
            train_frequencies[:, subset], shots, 4**subset_size
        )
        coefficients.append(coeff)
        nsr_values.append(nsr)
        subset_ids.append(np.full(nsr.shape, subset))
    return EigentaskBasis(
        coefficients=coefficients,
        nsr=np.concatenate(nsr_values),
        subset_index=np.concatenate(subset_ids),
    )


def nsr_cutoff_mask(nsr: np.ndarray, shots: int, cutoff: float) -> np.ndarray:
    """Retain eigentask ``l`` iff ``beta_l^2 / S <= cutoff`` (``cutoff = lambda``)."""
    return nsr / shots <= cutoff


def resolvable_expressive_capacity(
    nsr: np.ndarray, shots: int, cutoff: float | None = None
) -> float:
    """REC ``C = sum_l 1 / (1 + beta_l^2 / S)`` over eigentasks with ``beta_l^2 / S < cutoff``."""
    ratio = nsr / shots
    if cutoff is not None:
        ratio = ratio[ratio < cutoff]
    return float(np.sum(1.0 / (1.0 + ratio)))


def unit_scale(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-variance normalisation with training statistics.

    Returns ``(train_unit, test_unit, sigma)``. Features with zero training variance are
    kept at zero after centring.
    """
    mean = train.mean(axis=0)
    sigma = train.std(axis=0)
    safe_sigma = np.where(sigma > 0, sigma, 1.0)
    return (train - mean) / safe_sigma, (test - mean) / safe_sigma, sigma


def signal_scale(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signal-aware scaling ``(sigma_i / max sigma) * r_unit`` for Pauli/X features."""
    train_unit, test_unit, sigma = unit_scale(train, test)
    factor = sigma / sigma.max()
    return train_unit * factor, test_unit * factor


def nsr_scale(
    train: np.ndarray, test: np.ndarray, nsr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """NSR-aware scaling ``(beta_l^-1 / max beta^-1) * y_unit`` over the given eigentasks."""
    train_unit, test_unit, _ = unit_scale(train, test)
    inverse_beta = 1.0 / np.sqrt(np.maximum(nsr, np.finfo(float).tiny))
    factor = inverse_beta / inverse_beta.max()
    return train_unit * factor, test_unit * factor
