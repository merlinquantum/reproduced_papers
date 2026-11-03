"""Fisher information, normalized Fisher, and effective dimension (paper Sec. 3, Eqs. 1-2).

Any model with ``probabilities(x, theta) -> (B, K)`` and an attribute ``d`` works:
the qubit models in ``lib.quantum``, the networks in ``lib.classical``, or a
photonic model later.

The Fisher of one (theta, x) pair is the expectation over the model's own label
distribution, as in the authors' code:

    F(theta, x) = sum_y  (d p_y)(d p_y)^T / p_y = sum_y p_y (d log p_y)(d log p_y)^T

The empirical Fisher at theta averages this over ``k`` inputs x_j (Eq. 1 with the
labels summed exactly instead of sampled).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


def summed_fisher(model, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Return the sum over rows of the per-sample Fishers, shape ``(d, d)``.

    Parameters
    ----------
    model : object
        Exposes ``probabilities(x, theta)`` and ``d``.
    x : torch.Tensor
        Inputs ``(B, s_in)``.
    theta : torch.Tensor
        Parameters ``(B, d)``; row ``b`` is used only with input row ``b``.

    Returns
    -------
    torch.Tensor
        ``sum_b sum_y (d p_y)(d p_y)^T / p_y``, detached. It is accumulated as
        ``J^T J`` with rows ``d p_y / sqrt(p_y)`` so no ``(B, d, d)`` tensor is built.
    """
    theta = theta.detach().clone().requires_grad_(True)
    probabilities = model.probabilities(x, theta)
    n_outputs = probabilities.shape[1]
    total = torch.zeros(model.d, model.d, dtype=theta.dtype, device=theta.device)
    for output in range(n_outputs):
        # Rows are independent, so the gradient of the column sum holds per-sample gradients.
        (gradient,) = torch.autograd.grad(
            probabilities[:, output].sum(), theta, retain_graph=output < n_outputs - 1
        )
        # A probability that underflows to 0 has a zero gradient; the floor keeps 0/0 out.
        weight = (
            probabilities[:, output].detach().clamp_min(torch.finfo(theta.dtype).tiny)
        )
        scaled = gradient / weight.sqrt().unsqueeze(1)
        total = total + scaled.T @ scaled
    return total


def empirical_fisher(
    model, thetas: torch.Tensor, inputs: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Average per-sample Fishers over all inputs for every parameter set.

    Parameters
    ----------
    model : object
        Exposes ``probabilities(x, theta)`` and ``d``.
    thetas : torch.Tensor
        Parameter sets ``(N_theta, d)``.
    inputs : torch.Tensor
        Inputs ``(N_x, s_in)`` shared by every parameter set.
    chunk_size : int
        Maximum number of (theta, x) pairs per autograd pass.

    Returns
    -------
    torch.Tensor
        Empirical Fisher matrices ``(N_theta, d, d)``.
    """
    n_thetas, n_inputs = thetas.shape[0], inputs.shape[0]
    fishers = torch.zeros(
        n_thetas, model.d, model.d, dtype=thetas.dtype, device=thetas.device
    )
    for theta_index in range(n_thetas):
        total = torch.zeros(model.d, model.d, dtype=thetas.dtype, device=thetas.device)
        for start in range(0, n_inputs, chunk_size):
            x_chunk = inputs[start : start + chunk_size]
            if getattr(model, "shared_parameters", False):
                # Models whose trainable phases are shared layer parameters (feed-forward
                # block) compute per-sample gradients themselves at one theta.
                total = total + model.summed_fisher_shared(x_chunk, thetas[theta_index])
            else:
                theta_chunk = thetas[theta_index].expand(x_chunk.shape[0], -1)
                total = total + summed_fisher(model, x_chunk, theta_chunk)
        fishers[theta_index] = total / n_inputs
        logger.debug("Fisher %d/%d done", theta_index + 1, n_thetas)
    return fishers


def normalize_fisher(fishers: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``F_hat = d F / E_theta[tr F]`` and the average trace.

    The average trace over the sampled parameter sets estimates
    ``(1 / V_Theta) int tr F dtheta``, so ``mean tr F_hat = d`` by construction.
    """
    d = fishers.shape[1]
    mean_trace = float(np.mean(np.trace(fishers, axis1=1, axis2=2)))
    if not mean_trace > 0.0:
        raise ValueError(f"Average Fisher trace must be positive; got {mean_trace}")
    return d * fishers / mean_trace, mean_trace


def effective_dimension(
    f_hat: np.ndarray, n_values: list[float], gamma: float = 1.0
) -> np.ndarray:
    """Monte Carlo estimate of the effective dimension ``d_{gamma,n}`` (Eq. 2).

    The integral over Theta is the average over the ``N_theta`` sampled F_hat. The
    square-root determinant is computed in log space and averaged with
    ``logsumexp``, as in the authors' ``EffectiveDimension.eff_dim``.

    Parameters
    ----------
    f_hat : numpy.ndarray
        Normalized Fishers ``(N_theta, d, d)``.
    n_values : list of float
        Numbers of data samples ``n > 1``.
    gamma : float
        Constant in ``(0, 1]``. Default value is 1.0 (the paper's Fig. 3a).

    Returns
    -------
    numpy.ndarray
        ``d_{gamma,n}`` for each ``n`` (not divided by ``d``).
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1]; got {gamma}")
    n_thetas, d, _ = f_hat.shape
    identity = np.eye(d)
    values = []
    for n in n_values:
        kappa = gamma * n / (2.0 * math.pi * math.log(n))
        # NumPy linked to Apple Accelerate raises spurious divide/overflow/invalid flags
        # in slogdet even for 2 * I at d >= 40 (checked 2026-10-20); the explicit sign and
        # finiteness checks below catch a genuinely bad determinant instead.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sign, log_det = np.linalg.slogdet(identity + kappa * f_hat)
        if np.any(sign <= 0) or not np.all(np.isfinite(log_det)):
            raise ValueError(
                "I + kappa F_hat must be positive definite with a finite determinant for every sampled theta"
            )
        half_log_det = log_det / 2.0
        values.append(
            2.0 * (logsumexp(half_log_det) - math.log(n_thetas)) / math.log(kappa)
        )
    return np.asarray(values)


def spectrum_summary(f_hat: np.ndarray, zero_tolerance: float) -> dict:
    """Summarize the eigenvalues of the sampled F_hat matrices.

    Parameters
    ----------
    f_hat : numpy.ndarray
        Normalized Fishers ``(N_theta, d, d)``.
    zero_tolerance : float
        Eigenvalues at or below this value count as zero (relative to the mean
        eigenvalue, which is 1 for F_hat).

    Returns
    -------
    dict
        Rank, near-zero fraction, largest eigenvalue, and condition statistics.
    """
    eigenvalues = np.clip(np.linalg.eigvalsh(f_hat), 0.0, None)
    ranks = (eigenvalues > zero_tolerance).sum(axis=1)
    largest = eigenvalues.max(axis=1)
    smallest_nonzero = np.array(
        [
            row[row > zero_tolerance].min() if np.any(row > zero_tolerance) else np.nan
            for row in eigenvalues
        ]
    )
    return {
        "zero_tolerance": zero_tolerance,
        "mean_rank": float(ranks.mean()),
        "max_rank": int(ranks.max()),
        "min_rank": int(ranks.min()),
        "fraction_near_zero": float((eigenvalues <= zero_tolerance).mean()),
        "mean_largest_eigenvalue": float(largest.mean()),
        "max_eigenvalue": float(largest.max()),
        "median_condition_number_on_support": float(
            np.nanmedian(largest / smallest_nonzero)
        ),
    }


def fisher_rao_norm(theta: np.ndarray, f_hat: np.ndarray) -> float:
    """Return ``theta^T F_hat theta`` (Table 1 uses F_hat normalized at the trained theta)."""
    return float(theta @ f_hat @ theta)
