"""Additional fair classical baselines of the Phase 5 comparison (B2-B5).

These are **not part of the paper**. They are independent controls for C7,
C10 and C11 (LOG.md, "Phase 5 Comparison Plan"). Every model maps the
compressed features x (Eq. (93), shape ``(batch, 6)``) to E[Y_k | x] for the
three players, like the paper's models, so the existing training loop,
metrics and runner apply unchanged.

- ``BiasEncodedFourier`` (B2): the surrogate's Fourier features z(x) with
  spectrum Omega_L (Sec. 8.3). Six logits per sample give a softmax mixture
  of the hexagon vertices v_1..v_6, so the zero-sum bias holds exactly and
  every behaviour in the hexagon is reachable.
- ``BiasedMLP`` (B3): 6 -> tanh(h) -> 6 softmax vertex logits (bias exact).
- ``UnbiasedMLP`` (B4): 6 -> tanh(h) -> 3 sigmoid outputs (no bias).
- ``QuadraticModel`` (B5): linear readout on every monomial of degree <= 2 of
  the 6 features. The true win probabilities are exactly quadratic in x
  (bilinear in the strategy rows), so the class contains the truth; it is
  the parameter-matched ceiling.
"""

from __future__ import annotations

import torch
from torch import nn

from .rps_game import HEXAGON_VERTICES
from .surrogate import FourierSurrogate

N_FEATURES = 6
# Keeps the polynomial model's probabilities strictly inside (0, 1) so the
# log-likelihood stays finite; the clamp only binds at the boundary.
QUADRATIC_EPSILON = 1e-6


def _vertex_expectations(logits: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """E[Y | x] of a softmax mixture over the hexagon vertices."""
    return 2.0 * torch.softmax(logits, dim=-1) @ vertices - 1.0


class BiasEncodedFourier(FourierSurrogate):
    """B2: Fourier surrogate features with a bias-preserving vertex readout.

    Parameters
    ----------
    spectrum : int
        Frequency cut-off L of Omega_L.
    generator : torch.Generator
        Initialisation source: U[-1, 1] / (2 (2L+1)^6), as for the surrogate.
    dtype : torch.dtype
        Parameter dtype.
    chunk_size : int
        Samples per factorised contraction. Default value is 500.
    """

    n_outputs = 6

    def __init__(
        self,
        spectrum: int,
        generator: torch.Generator,
        dtype: torch.dtype,
        chunk_size: int = 500,
    ) -> None:
        super().__init__(spectrum, generator, dtype, chunk_size)
        n_per_output = 2 * self.n_frequencies
        initial = (
            2.0
            * torch.rand(
                (self.n_outputs, n_per_output), generator=generator, dtype=dtype
            )
            - 1.0
        ) / n_per_output
        self.weights = nn.Parameter(initial)
        self.register_buffer("vertices", torch.as_tensor(HEXAGON_VERTICES, dtype=dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """E[Y_k | x] from a softmax over the six vertex logits, ``(batch, 3)``."""
        return _vertex_expectations(self.raw_output(features), self.vertices)


class BiasedMLP(nn.Module):
    """B3: one-hidden-layer tanh MLP with a bias-preserving vertex readout.

    Parameters
    ----------
    hidden : int
        Hidden width (7 gives 97 parameters, about 91 = the biased model).
    dtype : torch.dtype
        Parameter dtype.
    """

    def __init__(self, hidden: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, 6, dtype=dtype),
        )
        self.register_buffer("vertices", torch.as_tensor(HEXAGON_VERTICES, dtype=dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return _vertex_expectations(self.net(features), self.vertices)


class UnbiasedMLP(nn.Module):
    """B4: one-hidden-layer tanh MLP with three independent sigmoid outputs.

    Parameters
    ----------
    hidden : int
        Hidden width (9 gives 93 parameters).
    dtype : torch.dtype
        Parameter dtype.
    """

    def __init__(self, hidden: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden, 3, dtype=dtype),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.net(features)) - 1.0


def quadratic_monomials(features: torch.Tensor) -> torch.Tensor:
    """x_i and x_i x_j (i <= j): 6 + 21 = 27 columns (the constant is the bias)."""
    rows, cols = torch.triu_indices(N_FEATURES, N_FEATURES)
    products = features[:, rows] * features[:, cols]
    return torch.cat([features, products], dim=1)


class QuadraticModel(nn.Module):
    """B5: E[Y_k | x] = clamp(w_k . [x, x x^T]_{i<=j} + b_k) (84 parameters).

    Parameters
    ----------
    dtype : torch.dtype
        Parameter dtype.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.linear = nn.Linear(
            N_FEATURES + N_FEATURES * (N_FEATURES + 1) // 2, 3, dtype=dtype
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.linear(quadratic_monomials(features))
        return torch.clamp(raw, -1.0 + QUADRATIC_EPSILON, 1.0 - QUADRATIC_EPSILON)
