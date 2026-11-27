"""Additional fair classical baselines (README, *Fair Baselines*).

These are independent controls, not part of the paper's method:

* No-reservoir control: persistence (``y(t+1) = x(t)``, the newest lag of the window) and AR(4),
  which is the linear ridge of :mod:`lib.readouts` on the raw lags;
* Classical-kernel and no-reservoir controls: RBF kernel ridge, ``exp(-gamma ||x - x'||^2)``, with gamma and lambda
  selected jointly on the evaluation split from the same training subsample as
  the fidelity kernel (the paper's product kernel evaluated classically needs no
  separate baseline: it is the existing ``quantum_kernel_ideal`` readout);
* Tuned ESN: a ReservoirPy echo-state reservoir whose spectral radius, leak rate and
  input scaling are tuned on the evaluation split (the grid is enumerated here;
  the runner evaluates each setting with the linear readout).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import torch

from .metrics import rmse
from .readouts import TIE_TOLERANCE, FittedReadout, _cholesky_solve

logger = logging.getLogger(__name__)

# Column of the newest lag in a window (lib.data.make_windows: oldest lag first).
NEWEST_LAG = -1


def esn_settings(grid: dict[str, list[float]]) -> list[dict[str, float]]:
    """Tuned-ESN candidates in the declared nested order: spectral radius, leak rate, input scaling."""

    return [
        {"sr": float(sr), "lr": float(lr), "input_scaling": float(scale)}
        for sr, lr, scale in itertools.product(
            grid["sr"], grid["lr"], grid["input_scaling"]
        )
    ]


def select_first_within_tolerance(scores: list[float]) -> int:
    """Index of the first candidate within 1 % relative of the minimum score."""

    best = min(scores)
    return next(
        i for i, score in enumerate(scores) if score <= best * (1 + TIE_TOLERANCE)
    )


def fit_persistence(
    features: dict[str, np.ndarray], targets: dict[str, np.ndarray]
) -> FittedReadout:
    """Persistence forecast (no-reservoir control): the next hour equals the newest observed hour."""

    predictions = {
        split: values[:, NEWEST_LAG].copy() for split, values in features.items()
    }
    score = rmse(targets["eval"], predictions["eval"])
    return FittedReadout(
        predictions=predictions,
        lambda_selected=0.0,
        selection_scores={0.0: score},
        details={"readout": "persistence", "lag_column": NEWEST_LAG},
    )


def rbf_block(rows: torch.Tensor, cols: torch.Tensor, gamma: float) -> torch.Tensor:
    """``exp(-gamma ||r - c||^2)`` between two sets of feature vectors."""

    squared = (
        (rows * rows).sum(1)[:, None]
        + (cols * cols).sum(1)[None, :]
        - 2.0 * rows @ cols.T
    )
    return torch.exp(-gamma * squared.clamp_(min=0.0))


def _rbf_gram(
    points: torch.Tensor, gamma: float, chunk_size: int = 4096
) -> torch.Tensor:
    n = points.shape[0]
    gram = torch.empty(n, n, dtype=points.dtype, device=points.device)
    for start in range(0, n, chunk_size):
        gram[start : start + chunk_size] = rbf_block(
            points[start : start + chunk_size], points, gamma
        )
    return gram


def _rbf_times_vector(
    rows: torch.Tensor,
    cols: torch.Tensor,
    gamma: float,
    vector: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    out = torch.empty(rows.shape[0], dtype=rows.dtype, device=rows.device)
    for start in range(0, rows.shape[0], chunk_size):
        out[start : start + chunk_size] = (
            rbf_block(rows[start : start + chunk_size], cols, gamma) @ vector
        )
    return out


def fit_rbf_kernel_ridge(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    lambda_grid: list[float],
    gamma_factors: list[float],
    device: str,
    selection_subsample: int | None,
    fit_subsample: int | None,
) -> FittedReadout:
    """RBF kernel ridge (classical-kernel and no-reservoir controls) with the fidelity-kernel protocol of :mod:`lib.readouts`.

    Parameters
    ----------
    features, targets : dict
        Per-split arrays (``"train"``, ``"eval"``, ``"test"``).
    lambda_grid : list of float
        n-normalized regularization candidates (``K + n lambda I``).
    gamma_factors : list of float
        ``gamma = factor / d`` for ``d`` input features.
    device : str
        Torch device for Gram matrices and solves.
    selection_subsample, fit_subsample : int or None
        Evenly spaced training points for the selection and the final fit;
        None uses the full training split.

    Returns
    -------
    FittedReadout
        Predictions, the selected lambda, and the (gamma, lambda) selection record
        in ``details``. Candidates within 1 % relative of the minimum evaluation
        RMSE are tied; the tie goes to the smallest gamma, then the largest lambda.
    """

    torch_device = torch.device(device)
    n_features = features["train"].shape[1]

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(values)).to(
            torch_device, torch.float64
        )

    def evenly_spaced(limit: int | None) -> np.ndarray:
        n = features["train"].shape[0]
        if limit is None or limit >= n:
            return np.arange(n)
        return np.unique(np.linspace(0, n - 1, limit).round().astype(int))

    y_train = targets["train"]
    y_mean = float(y_train.mean())
    eval_x = tensor(features["eval"])
    sel_index = evenly_spaced(selection_subsample)
    sel_x = tensor(features["train"][sel_index])
    sel_rhs = tensor(y_train[sel_index] - y_mean)

    scores: dict[tuple[float, float], float] = {}
    for factor in gamma_factors:
        gamma = factor / n_features
        sel_gram = _rbf_gram(sel_x, gamma)
        for lam in lambda_grid:
            try:
                alpha = _cholesky_solve(sel_gram.clone(), sel_rhs, sel_index.size * lam)
            except RuntimeError:
                scores[(gamma, lam)] = float("inf")
                logger.debug(
                    "rbf candidate | gamma=%.4g | lambda=%.3g | not positive definite",
                    gamma,
                    lam,
                )
                continue
            pred = _rbf_times_vector(eval_x, sel_x, gamma, alpha).cpu().numpy() + y_mean
            scores[(gamma, lam)] = rmse(targets["eval"], pred)
            logger.debug(
                "rbf candidate | gamma=%.4g | lambda=%.3g | eval_rmse=%.6f",
                gamma,
                lam,
                scores[(gamma, lam)],
            )
        del sel_gram
    best = min(scores.values())
    if not np.isfinite(best):
        raise RuntimeError(
            "No (gamma, lambda) candidate gives a positive definite RBF system"
        )
    tied = [key for key, score in scores.items() if score <= best * (1 + TIE_TOLERANCE)]
    gamma, chosen = min(tied, key=lambda key: (key[0], -key[1]))

    fit_index = evenly_spaced(fit_subsample)
    fit_x = tensor(features["train"][fit_index])
    fit_rhs = tensor(y_train[fit_index] - y_mean)
    gram = _rbf_gram(fit_x, gamma)
    alpha = _cholesky_solve(gram, fit_rhs, fit_index.size * chosen)
    del gram
    predictions = {
        split: _rbf_times_vector(tensor(values), fit_x, gamma, alpha).cpu().numpy()
        + y_mean
        for split, values in features.items()
    }
    details: dict[str, Any] = {
        "kernel": "rbf",
        "gamma_selected": gamma,
        "gamma_factor_selected": gamma * n_features,
        "selection_eval_rmse_by_gamma": {
            f"{g:.4g}": {
                f"{lam:.3g}": score for (g2, lam), score in scores.items() if g2 == g
            }
            for g in sorted({g for g, _ in scores})
        },
        "selection_points": int(sel_index.size),
        "fit_points": int(fit_index.size),
    }
    return FittedReadout(
        predictions=predictions,
        lambda_selected=chosen,
        selection_scores={
            lam: score for (g, lam), score in scores.items() if g == gamma
        },
        details=details,
    )


__all__ = [
    "esn_settings",
    "fit_persistence",
    "fit_rbf_kernel_ridge",
    "rbf_block",
    "select_first_within_tolerance",
]
