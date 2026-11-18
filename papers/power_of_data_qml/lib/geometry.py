"""Effective dimension, geometric difference, model complexity, engineered labels.

Formulas from arXiv:2011.01938 (journal SI equation numbers in brackets):

* Effective dimension (App. E.2, J.3 [S111]) with ``t_l`` the eigenvalues of
  ``K`` normalized to ``Tr K = N``, sorted decreasingly. As printed the sum
  divides by ``N - k`` and diverges at ``k = N`` (AMB-8); we use the tail mean
  ``d = sum_{k=1}^N (1/(N-k+1)) sum_{l=k}^N t_l``, which gives exactly 1 for a
  rank-one kernel and N for orthogonal states, as the paper states.
* Regularized geometric difference (Eq. E.15/J.3 [S112, S113]):
  ``g_gen = sqrt(|| sqrt(K^Q) sqrt(K^C) (K^C + lam I)^-2 sqrt(K^C) sqrt(K^Q) ||)``
  and ``g_tra = lam sqrt(|| sqrt(K^Q) (K^C + lam I)^-2 sqrt(K^Q) ||)``. For each
  classical kernel the largest ``lam`` with ``g_tra < 0.045`` is used, and the
  reported ``g`` is the minimum over classical kernels.
* Model complexity (App. E.1): ``s = y^T (K + lam)^-1 K (K + lam)^-1 y``.
* Prediction-error bound (Eq. K.1 [S133]):
  ``sqrt(lam^2 y^T (K+lam)^-2 y / N) + sqrt(y^T (K+lam)^-1 K (K+lam)^-1 y / N)``.
* Engineered labels (App. F, J.4 [S118, S119]): top eigenvector ``v`` of the
  ``g_gen`` matrix, ``y' = sqrt(K^Q) v``, labels ``sign(y')`` kept with
  probability 0.9 and replaced by a random sign otherwise.

``K^Q`` denotes the quantum-side kernel (original or projected) and ``K^C`` the
classical one; both are trace-normalized before use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh as scipy_eigh

from .kernels import normalize_trace

LOGGER = logging.getLogger(__name__)

# App. J.3 [S114].
LAMBDA_GRID = (0.00001, 0.0001, 0.001, 0.01, 0.025, 0.05, 0.1)
G_TRA_THRESHOLD = 0.045


def symmetric_eigh(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of a symmetric PSD matrix, negative eigenvalues clipped."""
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (kernel + kernel.T))
    return np.clip(eigenvalues, 0.0, None), eigenvectors


def psd_sqrt(kernel: np.ndarray) -> np.ndarray:
    """Principal square root of a symmetric PSD matrix."""
    eigenvalues, eigenvectors = symmetric_eigh(kernel)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def effective_dimension(kernel: np.ndarray) -> float:
    """Approximate dimension of the space spanned by the kernel's feature vectors.

    Parameters
    ----------
    kernel : numpy.ndarray
        Kernel matrix (normalized internally to ``Tr K = N``).

    Returns
    -------
    float
        Value in ``[1, N]``.
    """
    normalized = normalize_trace(kernel)
    eigenvalues = np.sort(symmetric_eigh(normalized)[0])[::-1]
    n_points = eigenvalues.shape[0]
    tail_sums = np.cumsum(eigenvalues[::-1])[::-1]  # tail_sums[k] = sum_{l>=k} t_l
    tail_lengths = np.arange(n_points, 0, -1)
    return float(np.sum(tail_sums / tail_lengths))


@dataclass
class ClassicalSpectrum:
    """Cached eigen-decomposition of a trace-normalized classical kernel."""

    name: str
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @classmethod
    def from_kernel(cls, name: str, kernel: np.ndarray) -> ClassicalSpectrum:
        eigenvalues, eigenvectors = symmetric_eigh(normalize_trace(kernel))
        return cls(name, eigenvalues, eigenvectors)


def _largest_eigenpair(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    # Only the top eigenpair is needed (spectral norm of a PSD matrix).
    size = matrix.shape[0]
    eigenvalues, eigenvectors = scipy_eigh(
        0.5 * (matrix + matrix.T), subset_by_index=[size - 1, size - 1]
    )
    return float(eigenvalues[-1]), eigenvectors[:, -1]


def _g_gen(sqrt_quantum: np.ndarray, spectrum: ClassicalSpectrum, lam: float) -> float:
    c = spectrum.eigenvalues
    return float(
        np.sqrt(
            max(
                _largest_eigenpair(
                    _projected(sqrt_quantum, spectrum, c / (c + lam) ** 2)
                )[0],
                0.0,
            )
        )
    )


def _g_tra(sqrt_quantum: np.ndarray, spectrum: ClassicalSpectrum, lam: float) -> float:
    c = spectrum.eigenvalues
    return float(
        lam
        * np.sqrt(
            max(
                _largest_eigenpair(
                    _projected(sqrt_quantum, spectrum, 1.0 / (c + lam) ** 2)
                )[0],
                0.0,
            )
        )
    )


def _projected(
    sqrt_quantum: np.ndarray, spectrum: ClassicalSpectrum, weights: np.ndarray
) -> np.ndarray:
    # sqrt(K^Q) U diag(weights) U^T sqrt(K^Q)
    left = sqrt_quantum @ spectrum.eigenvectors
    return (left * weights) @ left.T


def geometric_difference(
    sqrt_quantum: np.ndarray, spectrum: ClassicalSpectrum, lam: float
) -> tuple[float, float]:
    """Return ``(g_gen, g_tra)`` for one classical kernel and one ``lam``.

    Parameters
    ----------
    sqrt_quantum : numpy.ndarray
        ``sqrt(K^Q)`` of the trace-normalized quantum-side kernel.
    spectrum : ClassicalSpectrum
        Eigen-decomposition of the trace-normalized classical kernel.
    lam : float
        Regularization.
    """
    return _g_gen(sqrt_quantum, spectrum, lam), _g_tra(sqrt_quantum, spectrum, lam)


@dataclass
class GeometricDifferenceResult:
    """Minimum geometric difference over classical kernels (paper's reported g).

    Attributes
    ----------
    g : float
        Minimum ``g_gen`` over admissible classical kernels.
    kernel : str
        Classical kernel attaining the minimum.
    lam : float
        Regularization used for that kernel.
    g_tra : float
        Training-error term at the selected pair.
    per_kernel : dict
        For every classical kernel: selected ``lam``, ``g_gen``, ``g_tra``, or
        ``None`` when no ``lam`` satisfies the threshold.
    threshold_met : bool
        False when no (kernel, lam) pair satisfies ``g_tra < threshold``; the
        result then reports the pair with the smallest ``g_tra`` (AMB-18).
    """

    g: float
    kernel: str
    lam: float
    g_tra: float
    per_kernel: dict
    threshold_met: bool


def minimum_geometric_difference(
    quantum_kernel: np.ndarray,
    classical_spectra: list[ClassicalSpectrum],
    *,
    lambdas: tuple[float, ...] = LAMBDA_GRID,
    g_tra_threshold: float = G_TRA_THRESHOLD,
) -> GeometricDifferenceResult:
    """Paper's ``g``: per kernel the largest admissible ``lam``, then the minimum.

    When no (kernel, lam) pair satisfies ``g_tra < g_tra_threshold`` (the paper
    does not say what it did then; AMB-18), the pair with the smallest ``g_tra``
    is reported with ``threshold_met=False`` so these points can be counted and
    excluded downstream.
    """
    sqrt_quantum = psd_sqrt(normalize_trace(quantum_kernel))
    per_kernel: dict[str, dict | None] = {}
    best: tuple[float, str, float, float] | None = None
    least_training_error: tuple[float, str, float, float] | None = None
    for spectrum in classical_spectra:
        selected = None
        # g_tra grows with lambda, so scan from the largest lambda and stop at the
        # first admissible one; g_gen is only needed for the selected lambda.
        for lam in sorted(lambdas, reverse=True):
            g_tra = _g_tra(sqrt_quantum, spectrum, lam)
            if least_training_error is None or g_tra < least_training_error[3]:
                least_training_error = (None, spectrum.name, lam, g_tra)
            if g_tra < g_tra_threshold:
                selected = {
                    "lambda": lam,
                    "g_gen": _g_gen(sqrt_quantum, spectrum, lam),
                    "g_tra": g_tra,
                }
                break
        per_kernel[spectrum.name] = selected
        if selected is not None and (best is None or selected["g_gen"] < best[0]):
            best = (
                selected["g_gen"],
                spectrum.name,
                selected["lambda"],
                selected["g_tra"],
            )
    if best is None:
        _, name, lam, g_tra = least_training_error
        spectrum = next(sp for sp in classical_spectra if sp.name == name)
        least_training_error = (_g_gen(sqrt_quantum, spectrum, lam), name, lam, g_tra)
        LOGGER.warning(
            "No classical kernel meets g_tra < %.3g; reporting the smallest g_tra pair "
            "(kernel=%s, lambda=%g, g_tra=%.3f, g=%.3f)",
            g_tra_threshold,
            least_training_error[1],
            least_training_error[2],
            least_training_error[3],
            least_training_error[0],
        )
        return GeometricDifferenceResult(
            *least_training_error, per_kernel, threshold_met=False
        )
    return GeometricDifferenceResult(
        best[0], best[1], best[2], best[3], per_kernel, threshold_met=True
    )


def model_complexity(kernel: np.ndarray, labels: np.ndarray, lam: float) -> float:
    """``s_K = y^T (K + lam)^-1 K (K + lam)^-1 y`` on the trace-normalized kernel.

    With ``lam = 0`` this is the unregularized ``y^T K^-1 y`` (pseudo-inverse on
    the kernel's support).
    """
    eigenvalues, eigenvectors = symmetric_eigh(normalize_trace(kernel))
    projections = eigenvectors.T @ labels
    if lam == 0.0:
        support = eigenvalues > eigenvalues.max() * 1e-12
        return float(np.sum(projections[support] ** 2 / eigenvalues[support]))
    return float(np.sum(projections**2 * eigenvalues / (eigenvalues + lam) ** 2))


def prediction_error_bound(kernel: np.ndarray, labels: np.ndarray, lam: float) -> float:
    """Regularized bound ``s_{K,lam}(N)`` of Eq. K.1 (training plus generalization term)."""
    eigenvalues, eigenvectors = symmetric_eigh(normalize_trace(kernel))
    projections = eigenvectors.T @ labels
    n_points = labels.shape[0]
    training = lam**2 * np.sum(projections**2 / (eigenvalues + lam) ** 2)
    generalization = np.sum(projections**2 * eigenvalues / (eigenvalues + lam) ** 2)
    return float(np.sqrt(training / n_points) + np.sqrt(generalization / n_points))


def minimum_prediction_error_bound(
    classical_kernels: dict[str, np.ndarray],
    labels: np.ndarray,
    lambdas: tuple[float, ...] = LAMBDA_GRID,
) -> dict:
    """Smallest Eq. K.1 bound over classical kernels and ``lam`` (Fig. S3)."""
    best = None
    for name, kernel in classical_kernels.items():
        for lam in lambdas:
            value = prediction_error_bound(kernel, labels, lam)
            if best is None or value < best["bound"]:
                best = {"bound": value, "kernel": name, "lambda": lam}
    return best


@dataclass
class EngineeredLabels:
    """Labels engineered to saturate ``s_C = g^2 s_Q`` (App. F).

    Attributes
    ----------
    labels : numpy.ndarray
        ``+1/-1`` labels after the random relabelling.
    real_targets : numpy.ndarray
        ``y' = sqrt(K^Q) v`` before thresholding.
    g_gen : float
        Geometric difference achieved by the construction.
    flipped_fraction : float
        Fraction of labels that differ from the noiseless threshold.
    """

    labels: np.ndarray
    real_targets: np.ndarray
    g_gen: float
    flipped_fraction: float


def engineered_labels(
    quantum_kernel: np.ndarray,
    classical_kernel: np.ndarray,
    lam: float,
    rng: np.random.Generator,
    *,
    random_label_probability: float,
    threshold: str,
) -> EngineeredLabels:
    """Construct the maximally separating labels for one (K^Q, K^C, lam) triple.

    Parameters
    ----------
    quantum_kernel : numpy.ndarray
        Quantum-side kernel on all subsample points.
    classical_kernel : numpy.ndarray
        Classical kernel with the smallest geometric difference.
    lam : float
        Regularization of that classical kernel.
    rng : numpy.random.Generator
        Generator for the random relabelling.
    random_label_probability : float
        Probability that a label is replaced by a uniformly random sign
        (paper: 0.1).
    threshold : str
        ``"sign"`` (App. J, ``sign(y')``) or ``"median"`` (App. F, threshold at
        the median of ``y'``) (AMB-15).
    """
    sqrt_quantum = psd_sqrt(normalize_trace(quantum_kernel))
    spectrum = ClassicalSpectrum.from_kernel("classical", classical_kernel)
    c = spectrum.eigenvalues
    g_gen_sq, top_vector = _largest_eigenpair(
        _projected(sqrt_quantum, spectrum, c / (c + lam) ** 2)
    )
    # An eigenvector is defined up to sign, and the sign differs between LAPACK
    # drivers. Flipping it flips every clean label but not the randomly replaced
    # ones, which changes the task, so fix the sign: largest-magnitude entry > 0.
    top_vector = top_vector * np.sign(top_vector[np.argmax(np.abs(top_vector))])
    real_targets = sqrt_quantum @ top_vector
    if threshold == "sign":
        clean = np.where(real_targets > 0.0, 1, -1)
    elif threshold == "median":
        clean = np.where(real_targets > np.median(real_targets), 1, -1)
    else:
        raise ValueError(f"Unknown label threshold: {threshold}")
    if np.unique(clean).size < 2:
        # sign(y') is single-signed whenever the kernels are entrywise positive
        # (Perron-Frobenius), which makes the task trivial (AMB-15).
        raise ValueError(
            f"Engineered labels with threshold '{threshold}' contain a single class"
        )
    replace = rng.random(clean.shape[0]) < random_label_probability
    random_signs = np.where(rng.random(clean.shape[0]) < 0.5, 1, -1)
    labels = np.where(replace, random_signs, clean)
    return EngineeredLabels(
        labels=labels.astype(np.int64),
        real_targets=real_targets,
        g_gen=float(np.sqrt(max(g_gen_sq, 0.0))),
        flipped_fraction=float(np.mean(labels != clean)),
    )


__all__ = [
    "G_TRA_THRESHOLD",
    "LAMBDA_GRID",
    "ClassicalSpectrum",
    "EngineeredLabels",
    "GeometricDifferenceResult",
    "effective_dimension",
    "engineered_labels",
    "geometric_difference",
    "minimum_geometric_difference",
    "minimum_prediction_error_bound",
    "model_complexity",
    "prediction_error_bound",
    "psd_sqrt",
    "symmetric_eigh",
]
