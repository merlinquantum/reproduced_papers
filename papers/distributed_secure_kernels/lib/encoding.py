"""Amplitude encoding as in the authors' ``library.py`` (paper Definition 4 in Sec. 2.3; clients encode in Sec. 6.2.2).

A sample ``x`` becomes the n-qubit state with amplitudes ``x / ||x||``, zero-padded
to ``2**n`` with ``n = ceil(log2(len(x)))``, then reordered by one fixed
permutation drawn from the clients' shared seed (Python ``random.seed(42)``,
``random.shuffle``). Both clients apply the same permutation, so overlaps are
unchanged; it only matters for the circuit amplitudes.

Amplitude index convention: entry ``j`` is the coefficient of the computational
basis state whose qubit ``q`` holds bit ``(j >> q) & 1`` (Qiskit little-endian,
as used by ``QuantumCircuit.initialize``).
"""

from __future__ import annotations

import math
import random

import numpy as np

# Seed of ``library.py::permute_arrays`` (the "pre-shared seed" of Sec. 6.2.2).
SHARED_PERMUTATION_SEED = 42


def qubits_per_sample(n_features: int) -> int:
    """Number of qubits ``ceil(log2(n_features))`` used by the authors' code."""

    if n_features < 2:
        raise ValueError(f"Need at least 2 features, got {n_features}")
    return math.ceil(math.log2(n_features))


def shared_permutation(
    dimension: int, seed: int = SHARED_PERMUTATION_SEED
) -> np.ndarray:
    """Permutation of ``range(dimension)`` produced by ``random.seed(seed); random.shuffle``.

    Uses a private ``random.Random`` instance, so the global ``random`` state is untouched.
    """

    indices = list(range(dimension))
    random.Random(seed).shuffle(indices)
    return np.asarray(indices)


def amplitude_encode(features: np.ndarray, permute: bool = True) -> np.ndarray:
    """Encode each row of ``features`` as a normalised, padded, permuted amplitude vector.

    Parameters
    ----------
    features : numpy.ndarray
        Shape ``(n_samples, n_features)``.
    permute : bool
        Apply the shared-seed permutation. Default value is True.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, 2**n)``, unit-norm rows.

    Raises
    ------
    ValueError
        If a row has zero norm (the authors' code raises in the same case).
    """

    features = np.atleast_2d(np.asarray(features, dtype=float))
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms == 0):
        raise ValueError(
            f"Zero-norm sample(s) at rows {np.flatnonzero(norms == 0).tolist()}"
        )
    dimension = 2 ** qubits_per_sample(features.shape[1])
    states = np.zeros((features.shape[0], dimension))
    states[:, : features.shape[1]] = features / norms[:, None]
    if permute:
        # library.py: permuted[i] = original[indices[i]]
        states = states[:, shared_permutation(dimension)]
    return states
