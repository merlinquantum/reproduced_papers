"""Model factory: QNNs and the classical context references (LOG.md, B4)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .qnn import TrainableFrequencyQNN
from .qnn_merlin import PhotonicFrequencyModel


class MLP(nn.Module):
    """Fully connected tanh network mapping ``(batch, n_inputs)`` to ``(batch,)``.

    Parameters
    ----------
    n_inputs : int
        Input dimension.
    hidden_sizes : list of int
        Width of each hidden layer.
    dtype : torch.dtype
        Parameter dtype.
    generator : torch.Generator
        Seeded generator for Xavier-normal weight initialisation; biases start at 0.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_sizes: list[int],
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        widths = [n_inputs, *hidden_sizes, 1]
        layers: list[nn.Module] = []
        for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
            linear = nn.Linear(fan_in, fan_out, dtype=dtype)
            nn.init.xavier_normal_(linear.weight, generator=generator)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if index < len(widths) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class FourierRegressor(nn.Module):
    """Classical trainable-frequency Fourier series for one input.

    ``f(x) = c + sum_k a_k cos(w_k x) + b_k sin(w_k x)``. With
    ``trainable_frequencies`` the ``w_k`` are learned, which is the classical
    analogue of a TF model. Frequencies start at the given values, amplitudes are
    drawn from N(0, 0.1**2) with the seeded generator.
    """

    def __init__(
        self,
        initial_frequencies: list[float],
        trainable_frequencies: bool,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        n_terms = len(initial_frequencies)
        frequencies = torch.tensor(initial_frequencies, dtype=dtype)
        if trainable_frequencies:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)
        self.cos_amplitudes = nn.Parameter(
            0.1 * torch.randn(n_terms, generator=generator, dtype=dtype)
        )
        self.sin_amplitudes = nn.Parameter(
            0.1 * torch.randn(n_terms, generator=generator, dtype=dtype)
        )
        self.offset = nn.Parameter(torch.zeros((), dtype=dtype))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        phases = inputs[:, :1] * self.frequencies
        return (
            torch.cos(phases) @ self.cos_amplitudes
            + torch.sin(phases) @ self.sin_amplitudes
            + self.offset
        )


def build_network(
    model_cfg: dict[str, Any],
    n_inputs: int,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> nn.Module:
    """Build one network from its config block.

    Parameters
    ----------
    model_cfg : dict
        Block with ``family`` in ``{"qnn", "merlin_qnn", "mlp", "fourier"}`` and
        its keys. ``merlin_qnn`` is the MerLin photonic adaptation (Phase 4).
    n_inputs : int
        Number of input coordinates.
    dtype : torch.dtype
        Real parameter dtype.
    generator : torch.Generator
        Seeded generator used for initialisation.

    Returns
    -------
    torch.nn.Module
        Network mapping ``(batch, n_inputs)`` to ``(batch,)``.
    """
    family = model_cfg["family"]
    if family == "qnn":
        return TrainableFrequencyQNN(
            n_qubits=model_cfg["n_qubits"],
            n_inputs=n_inputs,
            layout=model_cfg["layout"],
            n_layers=model_cfg["n_layers"],
            entangler=model_cfg["entangler"],
            feature_map=model_cfg["feature_map"],
            frequency_mode=model_cfg["frequency_mode"],
            frequency_init=float(model_cfg["frequency_init"]),
            dtype=dtype,
            generator=generator,
        )
    if family == "merlin_qnn":
        return PhotonicFrequencyModel(
            n_modes=model_cfg["n_modes"],
            n_photons=model_cfg["n_photons"],
            n_inputs=n_inputs,
            layout=model_cfg["layout"],
            encoding_modes_per_input=model_cfg["encoding_modes_per_input"],
            feature_map=model_cfg["feature_map"],
            frequency_mode=model_cfg["frequency_mode"],
            frequency_init=float(model_cfg["frequency_init"]),
            readout=model_cfg["readout"],
            dtype=dtype,
            generator=generator,
        )
    if family == "mlp":
        return MLP(n_inputs, list(model_cfg["hidden_sizes"]), dtype, generator)
    if family == "fourier":
        if n_inputs != 1:
            raise ValueError("The Fourier regressor reference supports a single input.")
        return FourierRegressor(
            [float(w) for w in model_cfg["initial_frequencies"]],
            bool(model_cfg["trainable_frequencies"]),
            dtype,
            generator,
        )
    raise ValueError(
        f"Unknown model family {family!r}; expected 'qnn', 'merlin_qnn', 'mlp' or 'fourier'."
    )


def parameter_summary(network: nn.Module) -> dict[str, int]:
    """Count trainable parameters, split into ansatz and frequency parts for QNNs."""
    total = sum(p.numel() for p in network.parameters() if p.requires_grad)
    if isinstance(network, (TrainableFrequencyQNN, PhotonicFrequencyModel)):
        return {
            "total": total,
            "ansatz": network.n_ansatz_parameters(),
            "frequency": network.n_frequency_parameters(),
        }
    return {"total": total, "ansatz": 0, "frequency": 0}


def circuit_evaluation_cost_factor(summaries: list[dict[str, int]]) -> float:
    """Return the paper's ``C_f = (|theta_F| + |theta_A|) / |theta_A|`` (Sec. VI)."""
    ansatz = sum(s["ansatz"] for s in summaries)
    frequency = sum(s["frequency"] for s in summaries)
    if ansatz == 0:
        return math.nan
    return (frequency + ansatz) / ansatz
