"""Shot-based training of photonic circuits with the generalised parameter-shift rule.

Forward model: a MerLin ``QuantumLayer`` over the Perceval circuit of a Clements
mesh (Fig. 7) or a cascaded loop processor (Fig. 9), returning exact
photon-number-resolved output probabilities for a batch of parameter vectors.

Gradient (paper Eq. 143, App. F): for a parameter whose loss is a trigonometric
polynomial of degree at most ``R``,

    d l / d theta_k = sum_{j=1}^{2R} l(theta + kappa_j e_k) (-1)^{j+1}
                      / (4 R sin^2(kappa_j / 2)),   kappa_j = (2j - 1) pi / (2R).

With exact losses this is the exact derivative; with ``N``-shot estimates it is
unbiased with variance ``sum_j coefficient_j^2 Var[O | shifted] / N``
(paper Eq. 144). Shot noise is simulated by multinomial sampling from the exact
output distribution, which is statistically identical to drawing ``N``
photon-number-resolved samples.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import perceval as pcvl
import torch

from .clements import parameter_name
from .merlin_hardware import build_layer, outcome_mask

LossFunction = Callable[[np.ndarray], np.ndarray]


def gpsr_shifts_and_coefficients(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Shifts ``kappa_j`` and weights of paper Eq. 143 for frequency bound ``R``.

    Parameters
    ----------
    order : int
        Frequency bound ``R >= 1``.

    Returns
    -------
    shifts : numpy.ndarray
        ``(2R,)`` shifts.
    coefficients : numpy.ndarray
        ``(2R,)`` weights; ``sum coefficients^2 = (2R^2 + 1) / 6``.
    """
    if order < 1:
        raise ValueError(f"GPSR order must be at least 1, got {order}")
    index = np.arange(1, 2 * order + 1)
    shifts = (2 * index - 1) * np.pi / (2 * order)
    coefficients = (-1.0) ** (index + 1) / (4 * order * np.sin(shifts / 2) ** 2)
    return shifts, coefficients


def shifted_parameter_batch(parameters: np.ndarray, order: int) -> np.ndarray:
    """All ``P * 2R`` shifted parameter vectors, parameter-major."""
    shifts, _ = gpsr_shifts_and_coefficients(order)
    count = len(parameters)
    batch = np.repeat(parameters[None, :], count * len(shifts), axis=0)
    rows = np.arange(count * len(shifts))
    batch[rows, np.repeat(np.arange(count), len(shifts))] += np.tile(shifts, count)
    return batch


def gpsr_gradient(
    losses_of_batch: LossFunction, parameters: np.ndarray, order: int
) -> np.ndarray:
    """GPSR gradient from a batched loss function (exact or shot-based).

    Parameters
    ----------
    losses_of_batch : callable
        Maps a ``(B, P)`` parameter batch to ``(B,)`` loss values or estimates.
    parameters : numpy.ndarray
        Current ``(P,)`` parameters.
    order : int
        Frequency bound ``R``.

    Returns
    -------
    numpy.ndarray
        ``(P,)`` gradient estimate.
    """
    _, coefficients = gpsr_shifts_and_coefficients(order)
    losses = losses_of_batch(shifted_parameter_batch(parameters, order))
    return losses.reshape(len(parameters), len(coefficients)) @ coefficients


class PhotonicExpectation:
    """Exact and shot-estimated expectation of a diagonal observable via MerLin.

    Parameters
    ----------
    circuit : perceval.Circuit
        Circuit whose phases are named by :func:`clements.parameter_name`.
    occupation : tuple of int
        Fock input state.
    observable : callable
        Maps ``(D, M)`` photon patterns to ``(D,)`` values.
    parameter_scale : float
        Factor applied to parameters before they reach Perceval, e.g. 2 for
        beam-splitter angles (Perceval's ``BS(theta)`` uses ``theta / 2``).
        Default value is 1.0.
    device : str
        Torch device for MerLin. Default value is ``"cpu"``.
    chunk_size : int
        Largest parameter batch sent to MerLin at once. Default value is 512.
    hardware : Mapping[str, Any] | None
        Hardware condition of :mod:`merlin_hardware` (Phase 04 adaptation);
        ``None`` is the ideal PNR model of the reproduction. With
        post-selection, probabilities are restricted to the kept outcomes and
        renormalised. Default value is None.
    """

    def __init__(
        self,
        circuit: pcvl.Circuit,
        occupation: tuple[int, ...],
        observable: Callable[[np.ndarray], np.ndarray],
        parameter_scale: float = 1.0,
        device: str = "cpu",
        chunk_size: int = 512,
        hardware: Mapping[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.hardware = hardware
        self.layer = build_layer(circuit, tuple(occupation), hardware, self.device)
        all_patterns = np.array([tuple(key) for key in self.layer.output_keys])
        mask = outcome_mask(all_patterns, tuple(occupation), hardware)
        self.kept = None if mask is None else torch.tensor(mask, device=self.device)
        self.patterns = all_patterns if mask is None else all_patterns[mask]
        self.values = np.asarray(observable(self.patterns), dtype=np.float64)
        self.values_tensor = torch.tensor(
            self.values, dtype=torch.float64, device=self.device
        )
        self.parameter_scale = parameter_scale
        self.chunk_size = chunk_size
        self.parameter_count = len(circuit.get_parameters())
        expected = [parameter_name(i) for i in range(self.parameter_count)]
        names = sorted(parameter.name for parameter in circuit.get_parameters())
        if names != expected:
            raise ValueError("Circuit parameters must be phi_00000, phi_00001, ...")

    def probabilities(self, parameter_batch: np.ndarray) -> np.ndarray:
        """``(B, D)`` exact output probabilities, aligned with ``self.patterns``."""
        outputs = []
        with torch.no_grad():
            for start in range(0, len(parameter_batch), self.chunk_size):
                chunk = torch.tensor(
                    self.parameter_scale
                    * parameter_batch[start : start + self.chunk_size],
                    dtype=torch.float64,
                    device=self.device,
                )
                outputs.append(
                    self._kept_probabilities(self.layer(chunk)).cpu().numpy()
                )
        return np.concatenate(outputs)

    def _kept_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Restrict to post-selected outcomes and renormalise (no-op otherwise)."""
        if self.kept is None:
            return probabilities
        kept = probabilities[:, self.kept]
        return kept / kept.sum(dim=1, keepdim=True)

    def exact_losses(self, parameter_batch: np.ndarray) -> np.ndarray:
        """Exact (infinite-shot) ``<O>`` under the layer's hardware model."""
        return self.probabilities(parameter_batch) @ self.values

    def shot_losses(
        self, parameter_batch: np.ndarray, shots: int, rng: np.random.Generator
    ) -> np.ndarray:
        """``shots``-sample estimates of ``<O>`` for each parameter vector."""
        probabilities = np.clip(self.probabilities(parameter_batch), 0.0, None)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        estimates = np.empty(len(parameter_batch))
        for row, distribution in enumerate(probabilities):
            counts = rng.multinomial(shots, distribution)
            estimates[row] = counts @ self.values / shots
        return estimates

    def shot_variances(self, parameter_batch: np.ndarray) -> np.ndarray:
        """Single-shot variances ``Var[O | theta]`` for each parameter vector."""
        probabilities = self.probabilities(parameter_batch)
        means = probabilities @ self.values
        return probabilities @ self.values**2 - means**2

    def autograd_gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Exact gradient by backpropagation through MerLin (reference only)."""
        tensor = torch.tensor(
            parameters[None, :],
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        probabilities = self._kept_probabilities(
            self.layer(self.parameter_scale * tensor)
        )
        loss = (probabilities @ self.values_tensor).sum()
        loss.backward()
        return tensor.grad[0].cpu().numpy()
