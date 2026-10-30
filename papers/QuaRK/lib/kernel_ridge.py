"""Matérn kernel ridge regression readout (paper Sec. 3.3, App. B.3).

Follows the authors' ``src/models/kernel.py`` and ``qrc_matern_krr.py``:

* unit-variance Matérn kernel (scikit-learn convention) on standardized features;
* dual solution ``alpha = (K + reg I)^{-1} y``. The code adds ``reg`` without the
  factor N written in Sec. 3.3; we follow the code;
* per task: a grid over ``nu`` and a bounded Brent search over ``log(xi)`` that
  minimizes the validation MSE at ``reg = 1e-6``, on a shuffled 80/20 split of the
  training set.

Pairwise distances are computed once and reused for every ``xi``, which gives the
same kernel as ``sklearn.gaussian_process.kernels.Matern`` (checked in tests).
Linear systems use LU (``numpy.linalg.solve``), never an upper Cholesky: Accelerate
on macOS 27.0.x corrupts memory in upper ``potrf`` for n >= 1450.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from scipy.special import gamma as gamma_function
from scipy.special import kv

logger = logging.getLogger(__name__)


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Standardize with the training mean and standard deviation (``StandardScaler``)."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0.0] = 1.0
    return [(train - mean) / scale] + [(other - mean) / scale for other in others]


def matern_from_distances(distances: np.ndarray, xi: float, nu: float) -> np.ndarray:
    """Unit-variance Matérn kernel ``phi(d)`` of Eq. (10) from Euclidean distances."""
    # In-place operations keep at most two N x N temporaries alive.
    if nu == 0.5:
        kernel = distances * (-1.0 / xi)
        return np.exp(kernel, out=kernel)
    if nu == 1.5:
        scaled = distances * (math.sqrt(3.0) / xi)
        kernel = scaled + 1.0
        kernel *= np.exp(np.negative(scaled, out=scaled), out=scaled)
        return kernel
    if nu == 2.5:
        scaled = distances * (math.sqrt(5.0) / xi)
        kernel = scaled * scaled
        kernel /= 3.0
        kernel += scaled
        kernel += 1.0
        kernel *= np.exp(np.negative(scaled, out=scaled), out=scaled)
        return kernel
    argument = distances * (math.sqrt(2.0 * nu) / xi)
    argument[argument == 0.0] += np.finfo(float).eps  # as in scikit-learn
    kernel = _threaded_kv(nu, argument)
    kernel *= np.power(argument, nu, out=argument)
    kernel *= (2.0 ** (1.0 - nu)) / gamma_function(nu)
    return kernel


def _threaded_kv(nu: float, argument: np.ndarray) -> np.ndarray:
    """``scipy.special.kv`` over row blocks in threads (the ufunc releases the GIL).

    Element-wise, so the result is bit-identical to a single call.
    """
    if argument.ndim != 2 or argument.shape[0] < 64:
        return kv(nu, argument)
    out = np.empty_like(argument)
    blocks = np.array_split(np.arange(argument.shape[0]), os.cpu_count() or 1)

    def fill(rows):
        out[rows[0] : rows[-1] + 1] = kv(nu, argument[rows[0] : rows[-1] + 1])

    with ThreadPoolExecutor(len(blocks)) as pool:
        list(pool.map(fill, [b for b in blocks if b.size]))
    return out


def solve_dual(gram: np.ndarray, targets: np.ndarray, reg: float) -> np.ndarray:
    """Dual weights ``(K + reg I)^{-1} y`` by LU."""
    system = gram.copy()
    system[np.diag_indices_from(system)] += reg
    return np.linalg.solve(system, targets)


def inner_validation_split(num_train: int, val_ratio: float, seed: int):
    """Shuffled split of the training rows used for tuning (authors' convention)."""
    permutation = np.random.default_rng(seed).permutation(num_train)
    num_val = max(1, int(val_ratio * num_train))
    return permutation[num_val:], permutation[:num_val]


def tune_matern(features, targets, tuning: dict, *, task: str) -> dict:
    """Select ``(xi, nu)`` by validation MSE, as ``tune_matern_grid_train_val``.

    Parameters
    ----------
    features : numpy.ndarray
        Standardized training features ``(N, D)``.
    targets : numpy.ndarray
        Training labels ``(N,)``.
    tuning : dict
        ``nu_grid``, ``xi_bounds``, ``xi_maxiter``, ``val_ratio``, ``seed``, ``reg``.
    task : str
        Task name, for the log.

    Returns
    -------
    dict
        ``xi``, ``nu``, ``val_mse`` and the per-``nu`` results ``candidates``.
    """
    inner, val = inner_validation_split(
        features.shape[0], float(tuning["val_ratio"]), int(tuning["seed"])
    )
    d_inner = cdist(features[inner], features[inner])
    d_val = cdist(features[val], features[inner])
    reg = float(tuning["reg"])
    low, high = (math.log(float(b)) for b in tuning["xi_bounds"])

    candidates = []
    for nu in tuning["nu_grid"]:
        nu = float(nu)

        def validation_mse(log_xi: float, nu: float = nu) -> float:
            xi = math.exp(log_xi)
            weights = solve_dual(
                matern_from_distances(d_inner, xi, nu), targets[inner], reg
            )
            prediction = matern_from_distances(d_val, xi, nu) @ weights
            return float(np.mean((prediction - targets[val]) ** 2))

        result = minimize_scalar(
            validation_mse,
            bounds=(low, high),
            method="bounded",
            options={"maxiter": int(tuning["xi_maxiter"])},
        )
        candidate = {
            "nu": nu,
            "xi": math.exp(float(result.x)),
            "val_mse": float(result.fun),
            "evaluations": int(result.nfev),
        }
        candidates.append(candidate)
        logger.info(
            "TRAIN_TUNING_COMPLETED | task=%s | nu=%g | xi=%.6g | val_mse=%.6g | evaluations=%d",
            task,
            nu,
            candidate["xi"],
            candidate["val_mse"],
            candidate["evaluations"],
        )
    best = min(
        candidates, key=lambda c: c["val_mse"]
    )  # first wins on ties, as the code
    return {
        "xi": best["xi"],
        "nu": best["nu"],
        "val_mse": best["val_mse"],
        "candidates": candidates,
    }


def kernel_regularization_sweep(
    train, y_train, test, y_test, *, xi, nu, reg_grid, tuning
):
    """Train, validation and test MSE of Matérn KRR for every ``reg`` in the grid.

    Train and test MSE use the full training set (the paper's Fig. 2 and the
    authors' ``sweep_regularization``). Validation MSE fits on the inner training
    rows and scores the inner validation rows of the tuning split.

    Returns
    -------
    tuple[dict, numpy.ndarray, numpy.ndarray]
        Arrays over the grid (``mse_train``, ``mse_val``, ``mse_test``, ``rkhs_norm``),
        then the full training Gram matrix and the test-train kernel, for reuse.
    """
    inner, val = inner_validation_split(
        train.shape[0], float(tuning["val_ratio"]), int(tuning["seed"])
    )
    out = {
        k: np.empty(len(reg_grid))
        for k in ("mse_train", "mse_val", "mse_test", "rkhs_norm")
    }
    gram_inner = matern_from_distances(cdist(train[inner], train[inner]), xi, nu)
    cross_val = matern_from_distances(cdist(train[val], train[inner]), xi, nu)
    for i, reg in enumerate(reg_grid):
        weights = solve_dual(gram_inner, y_train[inner], reg)
        out["mse_val"][i] = np.mean((cross_val @ weights - y_train[val]) ** 2)
    del gram_inner, cross_val
    gram = matern_from_distances(cdist(train, train), xi, nu)
    cross_test = matern_from_distances(cdist(test, train), xi, nu)
    for i, reg in enumerate(reg_grid):
        weights = solve_dual(gram, y_train, reg)
        out["mse_train"][i] = np.mean((gram @ weights - y_train) ** 2)
        out["mse_test"][i] = np.mean((cross_test @ weights - y_test) ** 2)
        out["rkhs_norm"][i] = math.sqrt(max(float(weights @ gram @ weights), 0.0))
    return out, gram, cross_test


def linear_ridge_fit_predict(train, y_train, evaluation, reg):
    """Ridge regression with an unpenalized intercept on standardized features."""
    offset = y_train.mean()
    gram = train.T @ train
    gram[np.diag_indices_from(gram)] += reg
    coefficients = np.linalg.solve(gram, train.T @ (y_train - offset))
    return train @ coefficients + offset, evaluation @ coefficients + offset


def linear_regularization_sweep(train, y_train, test, y_test, *, reg_grid, tuning):
    """Train, validation and test MSE of linear ridge over the grid (same splits as KRR)."""
    inner, val = inner_validation_split(
        train.shape[0], float(tuning["val_ratio"]), int(tuning["seed"])
    )
    out = {k: np.empty(len(reg_grid)) for k in ("mse_train", "mse_val", "mse_test")}
    for i, reg in enumerate(reg_grid):
        _, pred_val = linear_ridge_fit_predict(
            train[inner], y_train[inner], train[val], reg
        )
        pred_train, pred_test = linear_ridge_fit_predict(train, y_train, test, reg)
        out["mse_val"][i] = np.mean((pred_val - y_train[val]) ** 2)
        out["mse_train"][i] = np.mean((pred_train - y_train) ** 2)
        out["mse_test"][i] = np.mean((pred_test - y_test) ** 2)
    return out


def decay_slope(sizes, test_mse, *, min_size: int) -> float:
    """Least-squares slope of log(test MSE) against log(N) for ``N >= min_size``."""
    sizes = np.asarray(sizes, dtype=float)
    mse = np.asarray(test_mse, dtype=float)
    keep = sizes >= min_size
    slope, _ = np.polyfit(np.log(sizes[keep]), np.log(mse[keep]), 1)
    return float(slope)
