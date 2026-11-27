"""Classical and projected quantum kernel matrices (paper App. J.2, J.3, J.5).

Bandwidths follow the paper's normalization ``gamma = scale / (n * Var)``, where
``n`` is the system size (number of input features) and ``Var`` is the variance
of all coordinates of all data points: of the PCA features ``x_ik`` for
classical kernels, and of the Pauli expectations ``Tr(P rho_k(x_i))`` for the
projected quantum kernel.
"""

from __future__ import annotations

import numpy as np

# App. J.3: bandwidths scanned when minimizing the geometric difference.
GEOMETRY_GAMMA_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def squared_distances(points: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, clipped at zero."""
    norms = np.sum(points**2, axis=1)
    distances = norms[:, None] + norms[None, :] - 2.0 * points @ points.T
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, 0.0)
    return distances


def bandwidth_normalizer(points: np.ndarray, system_size: int) -> float:
    """Return ``1 / (n * Var)`` over all coordinates of ``points``."""
    variance = float(np.var(points))
    if variance <= 0.0:
        raise ValueError("Cannot normalize a bandwidth for zero-variance data")
    return 1.0 / (system_size * variance)


def linear_kernel(features: np.ndarray) -> np.ndarray:
    """Linear kernel ``x_i . x_j``."""
    return features @ features.T


def rbf_kernel_from_distances(distances: np.ndarray, gamma: float) -> np.ndarray:
    """Gaussian kernel ``exp(-gamma ||x_i - x_j||^2)`` from squared distances."""
    return np.exp(-gamma * distances)


def polynomial_kernel(features: np.ndarray, degree: int, gamma: float) -> np.ndarray:
    """Polynomial kernel ``(gamma x_i . x_j + 1)^degree`` (extension baseline)."""
    return (gamma * (features @ features.T) + 1.0) ** degree


def pqk_flat_features(pauli_features: np.ndarray) -> np.ndarray:
    """Flatten ``(N, n_qubits, 3)`` Pauli expectations to ``(N, 3 n_qubits)``."""
    return pauli_features.reshape(pauli_features.shape[0], -1)


def projected_quantum_kernel(
    pauli_distances: np.ndarray, gamma_scale: float, normalizer: float
) -> np.ndarray:
    """Projected quantum kernel (Eq. J.3).

    ``k(x_i, x_j) = exp(-gamma sum_k sum_P (Tr(P rho_k(x_i)) - Tr(P rho_k(x_j)))^2)``
    with ``gamma = gamma_scale * normalizer``.

    Parameters
    ----------
    pauli_distances : numpy.ndarray
        Squared distances between flattened Pauli expectation vectors.
    gamma_scale : float
        Grid value (e.g. 1.0).
    normalizer : float
        ``1 / (n * Var[Tr(P rho_k)])``.
    """
    return np.exp(-gamma_scale * normalizer * pauli_distances)


def normalize_trace(kernel: np.ndarray) -> np.ndarray:
    """Rescale a kernel matrix to ``Tr(K) = N`` (paper's normalization)."""
    trace = float(np.trace(kernel))
    if trace <= 0.0:
        raise ValueError("Kernel matrix has non-positive trace")
    return kernel * (kernel.shape[0] / trace)


def classical_geometry_kernels(
    features: np.ndarray,
    system_size: int,
    *,
    gamma_scales: tuple[float, ...] = GEOMETRY_GAMMA_SCALES,
    polynomial_degrees: tuple[int, ...] = (),
) -> dict[str, np.ndarray]:
    """Classical kernels scanned when minimizing the geometric difference.

    Paper set (App. J.3): the linear kernel and Gaussian kernels with
    ``gamma in {0.25, ..., 64} / (n Var[x])``. ``polynomial_degrees`` adds the
    polynomial extension baseline (not in the paper). Every matrix is
    normalized to ``Tr(K) = N``.

    Returns
    -------
    dict
        Kernel name to trace-normalized matrix.
    """
    normalizer = bandwidth_normalizer(features, system_size)
    distances = squared_distances(features)
    kernels = {"linear": normalize_trace(linear_kernel(features))}
    for scale in gamma_scales:
        kernels[f"rbf_{scale:g}"] = rbf_kernel_from_distances(
            distances, scale * normalizer
        )
    for degree in polynomial_degrees:
        kernels[f"poly_{degree}"] = normalize_trace(
            polynomial_kernel(features, degree, normalizer)
        )
    return kernels


__all__ = [
    "GEOMETRY_GAMMA_SCALES",
    "bandwidth_normalizer",
    "classical_geometry_kernels",
    "linear_kernel",
    "normalize_trace",
    "polynomial_kernel",
    "pqk_flat_features",
    "projected_quantum_kernel",
    "rbf_kernel_from_distances",
    "squared_distances",
]
