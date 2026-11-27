"""Classical Fourier surrogate of Sec. 8.3 (after Schreiber et al.).

For each player k the surrogate is a linear model in the Fourier features of
the quantum model,

    g_k(x) = sum_{omega in Omega_L} a_{k,omega} cos(omega . z) + b_{k,omega} sin(omega . z),

with z = (x1, x2, x3, x4 x5, x5 x6, x4 x6) and Omega_L = {-L, ..., L}^6, followed
by a HardTanh so that E[Y_k | x] = hardtanh(g_k(x)) lies in [-1, 1]. As in the
author code, both omega and -omega are kept, so each task has
2 (2L + 1)^6 parameters (1458 for L = 1, 31250 for L = 2) and the three tasks
together 3x that.

The dense feature matrix for 10000 test points and L = 2 would take 2.5 GB in
float64. ``g`` is therefore evaluated in factorised form,
g_k = Re sum_omega c_{k,omega} prod_d exp(i omega_d z_d) with c = a - i b,
contracting one input dimension at a time over chunks of samples. This is the
same function of the same parameters; ``tests/test_models.py`` checks it
against the dense features.
"""

from __future__ import annotations

import itertools

import torch
from torch import nn

N_VARIABLES = 6


def fourier_variables(features: torch.Tensor) -> torch.Tensor:
    """z(x) = (x1, x2, x3, x4 x5, x5 x6, x4 x6) for ``(batch, 6)`` features."""
    x4, x5, x6 = features[:, 3], features[:, 4], features[:, 5]
    return torch.stack(
        [features[:, 0], features[:, 1], features[:, 2], x4 * x5, x5 * x6, x4 * x6],
        dim=1,
    )


def frequency_grid(spectrum: int) -> torch.Tensor:
    """All omega in {-L..L}^6 in row-major order, shape ``((2L+1)^6, 6)``."""
    values = range(-spectrum, spectrum + 1)
    return torch.tensor(
        list(itertools.product(values, repeat=N_VARIABLES)), dtype=torch.float64
    )


def dense_features(features: torch.Tensor, spectrum: int) -> torch.Tensor:
    """``[cos(Omega z), sin(Omega z)]`` of shape ``(batch, 2 (2L+1)^6)`` (tests only)."""
    omega = frequency_grid(spectrum).to(features)
    phases = fourier_variables(features) @ omega.T
    return torch.cat([torch.cos(phases), torch.sin(phases)], dim=1)


class FourierSurrogate(nn.Module):
    """Three linear Fourier models with HardTanh outputs (Sec. 8.3).

    Parameters
    ----------
    spectrum : int
        Frequency cut-off L of Omega_L.
    generator : torch.Generator
        Source of the initialisation U[-1, 1] / (2 (2L+1)^6) (App. F).
    dtype : torch.dtype
        Parameter dtype.
    chunk_size : int
        Samples per factorised contraction; bounds peak memory at about
        ``chunk_size * 3 * (2L+1)^5`` complex numbers. Default value is 500.
    """

    def __init__(
        self,
        spectrum: int,
        generator: torch.Generator,
        dtype: torch.dtype,
        chunk_size: int = 500,
    ) -> None:
        super().__init__()
        if spectrum < 1:
            raise ValueError(f"spectrum must be >= 1, got {spectrum}")
        self.spectrum = spectrum
        self.chunk_size = chunk_size
        self.side = 2 * spectrum + 1
        self.n_frequencies = self.side**N_VARIABLES
        n_per_task = 2 * self.n_frequencies
        # App. F: "uniformly in [-1, 1] then multiplying by 1/#params" with
        # #params = 2 (2L+1)^6, the per-task count used by the author code.
        initial = (
            2.0 * torch.rand((3, n_per_task), generator=generator, dtype=dtype) - 1.0
        ) / n_per_task
        self.weights = nn.Parameter(initial)

    def raw_output(self, features: torch.Tensor) -> torch.Tensor:
        """Linear model outputs before any output map, shape ``(batch, n_outputs)``.

        ``n_outputs`` is the number of weight rows: 3 for the surrogate g_k(x),
        6 vertex logits for the bias-encoded subclass (Phase 5, B2).
        """
        cos_weights = self.weights[:, : self.n_frequencies]
        sin_weights = self.weights[:, self.n_frequencies :]
        coefficients = torch.complex(cos_weights, -sin_weights).reshape(
            self.weights.shape[0], *([self.side] * N_VARIABLES)
        )
        offsets = torch.arange(
            -self.spectrum,
            self.spectrum + 1,
            dtype=features.dtype,
            device=features.device,
        )
        outputs = []
        for start in range(0, features.shape[0], self.chunk_size):
            z = fourier_variables(features[start : start + self.chunk_size])
            # factors[d][n, i] = exp(i * (i - L) * z_{n, d})
            factors = [
                torch.exp(1j * z[:, d : d + 1] * offsets) for d in range(N_VARIABLES)
            ]
            # Contract the last frequency axis first: (3, K^6) -> (n, 3, K^5) -> ... -> (n, 3).
            partial = torch.einsum("k...i,ni->nk...", coefficients, factors[-1])
            for factor in reversed(factors[:-1]):
                partial = torch.einsum("nk...i,ni->nk...", partial, factor)
            outputs.append(partial.real)
        return torch.cat(outputs, dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """E[Y_k | x] = hardtanh(g_k(x)), shape ``(batch, 3)``."""
        return torch.clamp(self.raw_output(features), -1.0, 1.0)
