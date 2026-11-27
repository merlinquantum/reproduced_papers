"""Classical baselines restricted to a quantum model's frequency set.

The paper claims expressivity, not advantage, so both baselines share the
quantum model's accessible spectrum ``{0, s, 2s, ..., rs}`` (LOG.md, "Fair
Baseline Plan"):

- B1, ``least_squares_oracle``: the closed-form best fit on the grid. Its MSE
  is the lowest error any model with that spectrum can reach.
- B2, ``FourierSeriesModel``: the same ``2r + 1`` real parameters, trained with
  the paper's optimizer protocol, to separate protocol effects from circuit
  effects.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def fourier_design_matrix(
    inputs: np.ndarray, frequencies: Sequence[int], scaling: float
) -> np.ndarray:
    """Return columns ``[1, cos(s n x), sin(s n x) for n in frequencies]``.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of shape ``(N,)``.
    frequencies : Sequence[int]
        Positive frequencies ``n``.
    scaling : float
        Input scaling ``s`` shared with the quantum model's encoding.

    Returns
    -------
    np.ndarray
        Design matrix of shape ``(N, 1 + 2 len(frequencies))``.
    """
    columns = [np.ones_like(inputs)]
    for frequency in frequencies:
        columns.append(np.cos(scaling * frequency * inputs))
        columns.append(np.sin(scaling * frequency * inputs))
    return np.stack(columns, axis=-1)


def least_squares_oracle(
    inputs: np.ndarray, targets: np.ndarray, frequencies: Sequence[int], scaling: float
) -> dict:
    """Fit baseline B1 by linear least squares on the given grid.

    Parameters
    ----------
    inputs : np.ndarray
        Grid of shape ``(N,)``.
    targets : np.ndarray
        Target values of shape ``(N,)``.
    frequencies : Sequence[int]
        Positive frequencies of the model class.
    scaling : float
        Input scaling ``s``.

    Returns
    -------
    dict
        ``mse`` (plain mean squared error on the grid), ``parameters`` (fitted
        coefficients in design-matrix order) and ``predictions``.
    """
    design = fourier_design_matrix(inputs, frequencies, scaling)
    parameters, *_ = np.linalg.lstsq(design, targets, rcond=None)
    predictions = design @ parameters
    return {
        "mse": float(np.mean((predictions - targets) ** 2)),
        "parameters": parameters.tolist(),
        "predictions": predictions,
    }


COEFFICIENT_INITS = ("uniform", "zero")


class FourierSeriesModel(nn.Module):
    """Baseline B2: ``a_0 + sum_n (a_n cos(s n x) + b_n sin(s n x))`` with trainable ``a, b``.

    Parameters
    ----------
    frequencies : Sequence[int]
        Positive frequencies ``n``.
    scaling : float
        Input scaling ``s``.
    coefficient_init : str
        ``"uniform"`` or ``"zero"`` (see ``reset_parameters``).
    dtype : torch.dtype
        Real dtype of the parameters.
    """

    def __init__(
        self,
        frequencies: Sequence[int],
        scaling: float,
        coefficient_init: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        if coefficient_init not in COEFFICIENT_INITS:
            raise ValueError(
                f"coefficient_init must be one of {COEFFICIENT_INITS}, got {coefficient_init!r}"
            )
        self.coefficient_init = coefficient_init
        self.register_buffer(
            "frequencies",
            torch.tensor(list(frequencies), dtype=dtype),
            persistent=False,
        )
        self.scaling = float(scaling)
        self.weights = nn.Parameter(torch.zeros(1 + 2 * len(frequencies), dtype=dtype))

    def reset_parameters(self, generator: torch.Generator, high: float) -> None:
        """Initialise the coefficients.

        ``"uniform"``: ``U[0, high)``, the circuits' angle range (Phase 1 plan).
        ``"zero"``: all zeros, as for the zero-initialised trainable readouts of
        the compared quantum and photonic models (Phase 5 amendment). The
        generator is still consumed in both cases, so seeds stay aligned.
        """
        with torch.no_grad():
            draw = torch.rand(
                self.weights.shape, generator=generator, dtype=self.weights.dtype
            )
            if self.coefficient_init == "uniform":
                self.weights.copy_(draw * high)
            else:
                self.weights.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate on inputs of shape ``(batch,)``; returns ``(batch,)``."""
        angles = self.scaling * inputs[:, None] * self.frequencies[None, :]
        basis = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).flatten(1)
        return self.weights[0] + basis @ self.weights[1:]
