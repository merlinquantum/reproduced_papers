"""Fit the unpublished B2/B3 ``U_0`` to a kernel digitised from Fig. 4.

The paper draws ``U_0`` at random and does not publish it. Here ``U_0`` is
recovered by gradient descent so that the platform's ideal kernel matches the
digitised "Simulated Kernel" panel. Quantities that were not fitted (the
alternative kernels of SI Note 4, the experimental panel, the classification
accuracies) are then predictions of the fitted circuit.

``U_0 = exp(i H)`` with ``H`` Hermitian, built from an unconstrained complex 8x8
matrix ``X`` as ``H = (X + X^dagger) / 2``. The loss is the mean squared error
over the upper triangle of the 15x15 kernel. Several random starts are run and
the lowest-loss one is kept; every start is recorded.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any

import numpy as np
import torch

from .circuits import AbsPlatform, unitary_from_operations

logger = logging.getLogger(__name__)


def _residual_rows(platform: AbsPlatform) -> tuple[torch.Tensor, torch.Tensor]:
    """Return output-mode rows of every (outcome, residual mode) amplitude.

    One residual photon is assumed (B2 and B3). Row ``[k, r]`` lists the output
    modes of outcome ``k`` with the residual photon in ``residual_modes[r]``; the
    amplitude is ``perm(U[rows][:, inputs]) / sqrt(prod(counts!))``.
    """
    if len(platform.input_modes) - sum(platform.outcomes[0]) != 1:
        raise ValueError("U_0 fitting supports platforms with one residual photon")
    rows, norms = [], []
    for outcome in platform.outcomes:
        detected = [
            mode
            for mode, count in zip(platform.detected_modes, outcome)
            for _ in range(count)
        ]
        rows.append([[*detected, residual] for residual in platform.residual_modes])
        norms.append(1 / math.sqrt(math.prod(math.factorial(c) for c in outcome)))
    return torch.tensor(rows), torch.tensor(norms, dtype=torch.float64)


def _permanent3(matrices: torch.Tensor) -> torch.Tensor:
    """Return the permanent of the trailing 3x3 block of ``matrices``."""
    return sum(
        matrices[..., 0, a] * matrices[..., 1, b] * matrices[..., 2, c]
        for a, b, c in itertools.permutations(range(3))
    )


def _unitary(params: torch.Tensor) -> torch.Tensor:
    generator = torch.complex(params[0], params[1])
    return torch.linalg.matrix_exp(0.5j * (generator + generator.conj().T))


def fit_u0_to_kernel(
    platform: AbsPlatform,
    target_kernel: np.ndarray,
    start_seeds: list[int],
    n_steps: int,
    learning_rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the ``U_0`` whose ideal B2/B3 kernel best matches ``target_kernel``.

    Parameters
    ----------
    platform : AbsPlatform
        B2 or B3 (three photons, one residual photon).
    target_kernel : numpy.ndarray
        15x15 digitised kernel to fit.
    start_seeds : list of int
        One random start per seed (torch generator seed of the initial ``X``).
    n_steps : int
        Adam steps per start.
    learning_rate : float
        Adam learning rate.

    Returns
    -------
    u0 : numpy.ndarray
        Best-fitting 8x8 unitary.
    record : dict
        Loss, MAE and Pearson r of every start and the chosen start seed.
    """
    if not platform.needs_random_u0 or len(platform.input_modes) != 3:
        raise ValueError("U_0 fitting is defined for B2 and B3 only")
    adaptive = torch.tensor(
        np.array(
            [
                unitary_from_operations(platform.n_modes, platform.operations_for(o))
                for o in platform.outcomes
            ]
        )
    )
    rows, norms = _residual_rows(platform)
    inputs = list(platform.input_modes)
    target = torch.tensor(target_kernel, dtype=torch.float64)
    upper = torch.triu_indices(len(platform.outcomes), len(platform.outcomes), 1)

    def kernel(params: torch.Tensor) -> torch.Tensor:
        columns = (adaptive @ _unitary(params))[:, :, inputs]
        blocks = torch.stack([columns[k][rows[k]] for k in range(len(rows))])
        amplitudes = _permanent3(blocks) * norms[:, None]
        states = amplitudes / amplitudes.norm(dim=1, keepdim=True)
        return (states.conj() @ states.T).abs() ** 2

    starts, best = [], None
    for seed in start_seeds:
        generator = torch.Generator().manual_seed(seed)
        params = torch.randn(
            2, platform.n_modes, platform.n_modes, generator=generator,
            dtype=torch.float64,
        ).requires_grad_()  # fmt: skip
        optimizer = torch.optim.Adam([params], lr=learning_rate)
        for step in range(n_steps):
            residual = (kernel(params) - target)[upper[0], upper[1]]
            loss = (residual**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 250 == 0:
                logger.debug("U0_FIT_STEP | start_seed=%d | step=%d | mse=%.3e",
                             seed, step, loss.item())  # fmt: skip
        with torch.no_grad():
            fitted = kernel(params).numpy()
        a = fitted[upper[0], upper[1]]
        b = target_kernel[upper[0], upper[1]]
        summary = {
            "start_seed": seed,
            "mse": float(np.mean((a - b) ** 2)),
            "mean_abs_error": float(np.mean(np.abs(a - b))),
            "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        }
        starts.append(summary)
        logger.info(
            "U0_FIT_START_COMPLETED | platform=%s | start_seed=%d | steps=%d | "
            "mse=%.3e | mae=%.4f | pearson_r=%.4f",
            platform.name, seed, n_steps, summary["mse"],
            summary["mean_abs_error"], summary["pearson_r"],
        )  # fmt: skip
        if best is None or summary["mse"] < best[0]["mse"]:
            best = (summary, _unitary(params.detach()).numpy())
    chosen, u0 = best
    return u0, {"chosen_start_seed": chosen["start_seed"], "starts": starts}
