"""Learning random Fourier series, the task behind paper Section 3.1.

The paper's predictive claim is that the Fourier coefficient correlation ranks
ansatzes by how well they learn: models with lower FCC reach lower mean squared
error when fitting a random Fourier series. This module provides the pieces
needed to test that claim on photonic circuits.

Targets follow paper Eq. 7. Coefficients are drawn as ``sqrt(r) * exp(-i 2 pi p)``
with ``r, p`` uniform on [0, 1], so ``|c| <= 1``, and the frequency set of the
target is set equal to the frequency set the model can express, which keeps the
target inside the learnable domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .fourier import AVAILABLE_CIRCUITS, PhotonicSpectralModel, compute_fingerprint

# Draws used only to discover which frequencies a circuit can reach. Lower than
# the fingerprint budget because the active set is far less noisy than the
# correlations between coefficients.
FREQUENCY_DISCOVERY_SAMPLES = 48


@dataclass
class FourierTarget:
    """A real-valued random Fourier series sampled on a regular grid."""

    x: torch.Tensor
    y: torch.Tensor
    max_frequency: int
    n_points: int


def reachable_max_frequency(
    encoding: str, n_samples: int = FREQUENCY_DISCOVERY_SAMPLES, seed: int = 0
) -> int:
    """Highest frequency any circuit reaches under ``encoding``.

    The active set is taken as the union across circuit topologies, so every
    circuit sharing an encoding is compared against the same target spectrum.
    """
    highest = 0
    for circuit_index in AVAILABLE_CIRCUITS.values():
        torch.manual_seed(seed)
        model = PhotonicSpectralModel(
            dimension=1, encoding=encoding, circuit_index=circuit_index
        )
        _, _, labels, _ = compute_fingerprint(model, n_samples=n_samples)
        for label in labels:
            highest = max(highest, int(label.split("=")[1]))
    return highest


def random_fourier_target(
    max_frequency: int, n_points: int, seed: int
) -> FourierTarget:
    """Build a real random Fourier series with frequencies 0..``max_frequency``.

    The series is standardised to zero mean and unit variance so that the mean
    squared error is comparable across targets of different spectral content.
    """
    if n_points < 2 * max_frequency + 1:
        raise ValueError(
            f"n_points={n_points} is below the Nyquist requirement "
            f"{2 * max_frequency + 1} for max_frequency={max_frequency}"
        )

    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    signal = np.full_like(grid, np.sqrt(rng.uniform(0.0, 1.0)))
    for frequency in range(1, max_frequency + 1):
        amplitude = np.sqrt(rng.uniform(0.0, 1.0))
        phase = 2 * np.pi * rng.uniform(0.0, 1.0)
        signal += 2 * amplitude * np.cos(frequency * grid - phase)

    signal = (signal - signal.mean()) / signal.std()
    return FourierTarget(
        x=torch.tensor(grid, dtype=torch.float32).unsqueeze(1),
        y=torch.tensor(signal, dtype=torch.float32),
        max_frequency=max_frequency,
        n_points=n_points,
    )


class SpectralRegressor(nn.Module):
    """A photonic model with a trainable affine read-out head.

    The circuit measures a probability in [0, 1] while the target is a
    standardised Fourier series, so a scale and offset are needed to place the
    output on the target's range. The head is affine and therefore cannot add
    frequencies: which frequencies the model can express is still decided
    entirely by the circuit and its encoding.
    """

    def __init__(self, encoding: str, circuit_index: int):
        super().__init__()
        self.core = PhotonicSpectralModel(
            dimension=1, encoding=encoding, circuit_index=circuit_index
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = self.core.mode0_occupancy(self.core(x))
        return self.scale * signal + self.offset


def train_regressor(
    model: SpectralRegressor,
    target: FourierTarget,
    epochs: int = 200,
    lr: float = 0.05,
) -> tuple[float, list[float]]:
    """Fit ``model`` to ``target`` under MSE and return the best loss reached."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    best = float("inf")

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(target.x) - target.y) ** 2)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        history.append(value)
        best = min(best, value)

    return best, history


__all__ = [
    "FourierTarget",
    "SpectralRegressor",
    "random_fourier_target",
    "reachable_max_frequency",
    "train_regressor",
]
