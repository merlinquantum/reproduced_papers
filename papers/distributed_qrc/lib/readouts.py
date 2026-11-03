"""Ridge output layers: classical linear ridge, fidelity-kernel ridge, distributed readouts.

* Linear ridge (paper Eq. 5) with an unpenalized intercept.
* Kernel ridge with the fidelity kernel of :mod:`lib.kernels` (the paper's
  "quantum ridge", Eq. 6). The target is centered by its training mean (an
  intercept), which the paper does not specify.
* Regularization uses the n-normalized convention ``(A + n lambda I)`` so that a
  lambda selected on a training subsample transfers to the full training set.
  lambda is selected on the evaluation split from a predeclared grid:
  minimum evaluation RMSE, ties within 1 % relative
  going to the larger lambda.
* A distributed readout (SRMR, MRMR) splits the feature vector into m contiguous
  chunks, fits one independent ridge per chunk and averages the m predictions
  (our choice; ``merge = "sum"`` is the variant).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .kernels import (
    ShotNoise,
    bloch_vectors,
    contraction_factor,
    gram_matrix,
    kernel_times_vector,
)
from .metrics import rmse
from .noise import DeviceNoise

logger = logging.getLogger(__name__)

TIE_TOLERANCE = 0.01


def select_lambda(scores: dict[float, float]) -> float:
    """Smallest evaluation RMSE; candidates within 1 % relative count as ties -> largest lambda."""

    best = min(scores.values())
    tied = [lam for lam, score in scores.items() if score <= best * (1 + TIE_TOLERANCE)]
    return max(tied)


@dataclass
class FittedReadout:
    """Predictions and selection record of one ridge instance."""

    predictions: dict[str, np.ndarray]
    lambda_selected: float
    selection_scores: dict[float, float]
    details: dict[str, Any] = field(default_factory=dict)


def fit_linear_ridge(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    lambda_grid: list[float],
    intercept: bool = True,
) -> FittedReadout:
    """Closed-form linear ridge on ``features["train"]``; lambda chosen on ``"eval"``.

    With ``intercept=False`` this is exactly the paper's Eq. 5,
    ``W = Y R^T (R R^T + lambda I)^-1`` (no bias term; gap diagnosis C).
    """

    x, y = features["train"], targets["train"]
    if intercept:
        x_mean, y_mean = x.mean(axis=0), y.mean()
    else:
        x_mean, y_mean = np.zeros(x.shape[1]), 0.0
    xc, yc = x - x_mean, y - y_mean
    gram, moment = xc.T @ xc, xc.T @ yc
    n = x.shape[0]

    def solve(lam: float) -> np.ndarray:
        return np.linalg.solve(gram + n * lam * np.eye(gram.shape[0]), moment)

    def predict(weights: np.ndarray, split: str) -> np.ndarray:
        return (features[split] - x_mean) @ weights + y_mean

    scores = {
        lam: rmse(targets["eval"], predict(solve(lam), "eval")) for lam in lambda_grid
    }
    for lam, score in scores.items():
        logger.debug("ridge candidate | lambda=%.3g | eval_rmse=%.6f", lam, score)
    chosen = select_lambda(scores)
    weights = solve(chosen)
    return FittedReadout(
        predictions={split: predict(weights, split) for split in features},
        lambda_selected=chosen,
        selection_scores=scores,
    )


def _cholesky_solve(
    gram: torch.Tensor, rhs: torch.Tensor, shift: float
) -> torch.Tensor:
    """Solve ``(gram + shift I) a = rhs``, factorizing ``gram`` in place (it is consumed).

    torch's Cholesky is used on every device. ``scipy.linalg.cho_factor`` (SciPy
    1.18.1 with Accelerate, macOS / Python 3.14) was found to write outside its
    arrays and corrupt unrelated data on the development machine (an Accelerate
    upper-triangle Cholesky bug on macOS 27.0.x).
    """

    gram.diagonal().add_(shift)
    info = torch.empty((), dtype=torch.int32, device=gram.device)
    chol, info = torch.linalg.cholesky_ex(gram, out=(gram, info))
    if int(info) != 0:
        raise RuntimeError(
            f"Kernel matrix is not positive definite (Cholesky info={int(info)})"
        )
    return torch.cholesky_solve(rhs.reshape(-1, 1), chol)[:, 0]


def fit_kernel_ridge(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    lambda_grid: list[float],
    n_qubits: int,
    noise: DeviceNoise,
    device: str,
    selection_subsample: int | None,
    fit_subsample: int | None,
    shots: int | None = None,
    shot_seed: int = 0,
) -> FittedReadout:
    """Fidelity-kernel ridge regression.

    Parameters
    ----------
    features, targets : dict
        Per-split arrays (``"train"``, ``"eval"``, ``"test"``).
    lambda_grid : list of float
        n-normalized regularization candidates.
    n_qubits : int
        Width of the kernel feature map (5 or 10 in the paper).
    noise : DeviceNoise
        Device noise of the kernel circuit; ``IDEAL`` for the ideal simulator.
    device : str
        Torch device for Gram matrices and solves (``"cpu"`` or ``"cuda:0"``).
    selection_subsample : int or None
        Evenly spaced training points used to select lambda; None uses all of them.
    fit_subsample : int or None
        Evenly spaced training points used for the final fit; None (paper
        default) uses the full training split. A value is the RED-tier fallback.
    shots : int or None
        If set, every kernel entry is a finite-shot estimate (gap diagnosis).
        Default value is None (exact kernel).
    shot_seed : int
        Seed of the shot-noise generators. Default value is 0.

    Returns
    -------
    FittedReadout
        Predictions for every split and the lambda selection record.
    """

    n_features = features["train"].shape[1]
    factor = contraction_factor(noise, n_features, n_qubits)
    torch_device = torch.device(device)

    def to_bloch(split: str, index: np.ndarray | None = None) -> torch.Tensor:
        values = features[split] if index is None else features[split][index]
        return torch.from_numpy(bloch_vectors(values, n_qubits)).to(
            torch_device, torch.float64
        )

    def evenly_spaced(limit: int | None) -> np.ndarray:
        n = features["train"].shape[0]
        if limit is None or limit >= n:
            return np.arange(n)
        return np.unique(np.linspace(0, n - 1, limit).round().astype(int))

    def shot_noise(stream: int) -> ShotNoise | None:
        # One generator per measured kernel (selection Gram, eval kernel, ...),
        # re-seeded on each use so that every lambda sees the same estimates.
        if shots is None:
            return None
        generator = torch.Generator(device=torch_device)
        generator.manual_seed(
            int(np.random.SeedSequence([shot_seed, stream]).generate_state(1)[0])
        )
        return ShotNoise(shots, generator)

    y_train = targets["train"]
    y_mean = float(y_train.mean())
    eval_bloch = to_bloch("eval")

    sel_index = evenly_spaced(selection_subsample)
    sel_bloch = to_bloch("train", sel_index)
    sel_rhs = torch.from_numpy(y_train[sel_index] - y_mean).to(
        torch_device, torch.float64
    )
    sel_gram = gram_matrix(sel_bloch, factor, shot_noise=shot_noise(0))
    scores: dict[float, float] = {}
    for lam in lambda_grid:
        try:
            alpha = _cholesky_solve(sel_gram.clone(), sel_rhs, sel_index.size * lam)
        except RuntimeError:
            # A finite-shot Gram matrix need not be positive definite.
            scores[lam] = float("inf")
            logger.debug("kernel candidate | lambda=%.3g | not positive definite", lam)
            continue
        pred = (
            kernel_times_vector(
                eval_bloch,
                sel_bloch,
                factor,
                alpha.to(torch_device),
                shot_noise=shot_noise(1),
            )
            .cpu()
            .numpy()
            + y_mean
        )
        scores[lam] = rmse(targets["eval"], pred)
        logger.debug(
            "kernel candidate | lambda=%.3g | eval_rmse=%.6f", lam, scores[lam]
        )
    del sel_gram
    if not np.isfinite(min(scores.values())):
        raise RuntimeError(
            "No lambda in the grid gives a positive definite kernel matrix"
        )
    chosen = select_lambda(scores)

    fit_index = evenly_spaced(fit_subsample)
    fit_bloch = to_bloch("train", fit_index)
    fit_rhs = torch.from_numpy(y_train[fit_index] - y_mean).to(
        torch_device, torch.float64
    )
    # With finite shots the full Gram matrix may be indefinite at the chosen
    # lambda; then the next larger lambda of the grid is used (recorded).
    fit_lambdas = [lam for lam in sorted(lambda_grid) if lam >= chosen]
    for lam in fit_lambdas:
        gram = gram_matrix(fit_bloch, factor, shot_noise=shot_noise(2))
        try:
            alpha = _cholesky_solve(gram, fit_rhs, fit_index.size * lam).to(
                torch_device
            )
        except RuntimeError:
            del gram
            if shots is None:
                raise
            continue
        del gram
        break
    else:
        raise RuntimeError(
            "Full kernel matrix is not positive definite for any lambda >= the selected one"
        )
    fit_lambda = lam
    predictions = {}
    for stream, split in enumerate(features, start=3):
        rows = (
            fit_bloch
            if split == "train" and fit_index.size == y_train.size
            else to_bloch(split)
        )
        predictions[split] = (
            kernel_times_vector(
                rows, fit_bloch, factor, alpha, shot_noise=shot_noise(stream)
            )
            .cpu()
            .numpy()
            + y_mean
        )
    return FittedReadout(
        predictions=predictions,
        lambda_selected=chosen,
        selection_scores=scores,
        details={
            "kernel_factor": factor,
            "n_qubits": n_qubits,
            "layers": int(np.ceil(n_features / n_qubits)),
            "selection_points": int(sel_index.size),
            "fit_points": int(fit_index.size),
            "device": noise.name,
            "shots": shots,
            "fit_lambda": fit_lambda,
        },
    )


def partition_features(n_features: int, n_instances: int) -> list[np.ndarray]:
    """Contiguous, as-equal-as-possible feature chunks, one per ridge instance."""

    if n_instances > n_features:
        raise ValueError(
            f"Cannot split {n_features} features into {n_instances} instances"
        )
    return np.array_split(np.arange(n_features), n_instances)


def merge_predictions(parts: list[np.ndarray], merge: str) -> np.ndarray:
    if merge == "mean":
        return np.mean(parts, axis=0)
    if merge == "sum":
        return np.sum(parts, axis=0)
    raise ValueError(f"merge must be 'mean' or 'sum', got {merge!r}")


__all__ = [
    "FittedReadout",
    "fit_kernel_ridge",
    "fit_linear_ridge",
    "merge_predictions",
    "partition_features",
    "select_lambda",
]
