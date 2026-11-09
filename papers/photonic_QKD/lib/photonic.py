"""MerLin simulation of the paper's loop-based photonic sampler and feature map.

Circuit (Supp. Note 7, "Circuit family and theta-length constraint"): an
``N``-mode time-bin interferometer with loop length 1. One "tiling" is a
sequence of ``N - 1`` beam splitters coupling modes ``(i, i + 1)`` for
``i = 0 .. N-2`` in order, which is the parameterisation of the CUDA-Q ORCA
backend (``N - 1`` angles per loop). ``dim(theta)`` angles are padded with
zeros or truncated to ``(N - 1) * ceil(dim(theta) / (N - 1))``.

Measurement (Supp. Note 7, "Measurement and sample format"): threshold
detection, ``b_j = 1[n_j > 0]``. The exact output distribution is computed by
MerLin in the full Fock space; MerLin's default ``UNBUNCHED`` space would drop
the bunched outcomes that threshold detectors also click on. Finite-shot
samples are drawn from that distribution with ``torch.multinomial``.

Feature map Phi (Supp. Note 7, "Feature construction"): split each 16-bit
click pattern into two 8-bit halves, index each half most-significant bit
first, and concatenate the two normalised 256-bin histograms (d = 512).

Only the MerLin 0.4.2 public API is used: ``QuantumLayer``,
``MeasurementStrategy.probs`` and ``ComputationSpace.FOCK``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import merlin as ML
import perceval as pcvl
import torch

logger = logging.getLogger(__name__)

# Above this many Fock states the exact distribution no longer fits a laptop
# (16 modes / 12 photons ~ 1.7e7 states was killed at 7.5 GB RAM).
MAX_FOCK_STATES = 5_000_000


def fock_space_size(n_modes: int, n_photons: int) -> int:
    """Number of Fock states of ``n_photons`` in ``n_modes``."""
    return math.comb(n_modes + n_photons - 1, n_photons)


def n_tilings_for(theta_dim: int, n_modes: int) -> int:
    """Number of tilings consumed by ``theta_dim`` angles (Supp. Eq. S-ntiling)."""
    if theta_dim < 1:
        raise ValueError(f"theta_dim must be positive, got {theta_dim}")
    return math.ceil(theta_dim / (n_modes - 1))


def fit_angles(theta: torch.Tensor, n_angles: int) -> torch.Tensor:
    """Zero-pad or truncate ``theta`` to the sampler's ``n_angles`` (Supp. theta_fit rule)."""
    if theta.numel() >= n_angles:
        return theta[:n_angles]
    return torch.cat([theta, theta.new_zeros(n_angles - theta.numel())])


def time_bin_circuit(n_modes: int, n_tilings: int) -> pcvl.Circuit:
    """Build ``n_tilings`` nearest-neighbour beam-splitter chains on ``n_modes`` modes.

    Parameters are named ``theta_000``, ``theta_001``, ... in application
    order, so the lexicographic order MerLin uses for the ``theta`` vector is
    also the physical order.
    """
    circuit = pcvl.Circuit(n_modes)
    index = 0
    for _ in range(n_tilings):
        for mode in range(n_modes - 1):
            circuit.add(mode, pcvl.BS(theta=pcvl.P(f"theta_{index:03d}")))
            index += 1
    return circuit


class TimeBinFeatureMap:
    """Shot-limited photonic feature ``z~(theta) = [h1; h2]`` (before standardisation).

    Parameters
    ----------
    n_modes : int
        Number of modes ``N`` (paper: 16). Must be even.
    theta_dim : int
        Photonic parameter budget ``dim(theta)`` (paper: 15, 30 or 45).
    input_state : sequence of int
        Photon occupation per input mode. The paper only says "fixed input
        pattern"; the reproduction default is 8 photons in alternating modes.
    """

    def __init__(
        self, n_modes: int, theta_dim: int, input_state: Sequence[int]
    ) -> None:
        if n_modes % 2:
            raise ValueError(
                f"n_modes must be even to split the pattern in halves, got {n_modes}"
            )
        if len(input_state) != n_modes:
            raise ValueError(
                f"input_state has {len(input_state)} modes, expected {n_modes}"
            )
        n_photons = int(sum(input_state))
        n_states = fock_space_size(n_modes, n_photons)
        if n_states > MAX_FOCK_STATES:
            raise ValueError(
                f"{n_photons} photons in {n_modes} modes give {n_states} Fock states "
                f"(> {MAX_FOCK_STATES}); use fewer photons for exact simulation"
            )
        self.n_modes = n_modes
        self.theta_dim = theta_dim
        self.n_photons = n_photons
        self.input_state = tuple(int(count) for count in input_state)
        self.n_tilings = n_tilings_for(theta_dim, n_modes)
        self.n_angles = self.n_tilings * (n_modes - 1)
        self.half_modes = n_modes // 2
        self.bins_per_half = 2**self.half_modes
        self.feature_dim = 2 * self.bins_per_half

        self.layer = ML.QuantumLayer(
            input_size=0,
            circuit=time_bin_circuit(n_modes, self.n_tilings),
            input_state=list(self.input_state),
            trainable_parameters=["theta"],
            measurement_strategy=ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK),
            dtype=torch.float64,
        )
        self.layer.requires_grad_(False)
        angle_parameters = [
            parameter
            for parameter in self.layer.parameters()
            if parameter.numel() == self.n_angles
        ]
        if len(angle_parameters) != 1:
            raise RuntimeError(
                f"Expected one MerLin parameter tensor with {self.n_angles} angles, "
                f"found shapes {[tuple(p.shape) for p in self.layer.parameters()]}"
            )
        self._angles = angle_parameters[0]

        occupations = torch.tensor(self.layer.output_keys, dtype=torch.long)
        if occupations.shape != (n_states, n_modes):
            raise RuntimeError(
                f"Unexpected MerLin output keys shape {tuple(occupations.shape)}"
            )
        clicks = (occupations > 0).to(torch.long)
        place_values = 2 ** torch.arange(self.half_modes - 1, -1, -1)
        self.first_half_index = clicks[:, : self.half_modes] @ place_values
        self.second_half_index = clicks[:, self.half_modes :] @ place_values
        self.click_patterns = clicks.to(torch.bool)
        logger.info(
            "PHOTONIC_SAMPLER_READY | modes=%d | photons=%d | tilings=%d | angles=%d | fock_states=%d",
            n_modes,
            n_photons,
            self.n_tilings,
            self.n_angles,
            n_states,
        )

    @torch.no_grad()
    def outcome_probabilities(self, theta: torch.Tensor) -> torch.Tensor:
        """Exact Fock-state distribution ``p_theta`` (float64, sums to 1)."""
        self._angles.copy_(
            fit_angles(theta.detach().to(torch.float64).cpu(), self.n_angles)
        )
        probabilities = self.layer().reshape(-1).clamp_min(0.0)
        return probabilities / probabilities.sum()

    def _histograms(
        self, first: torch.Tensor, second: torch.Tensor, weights: torch.Tensor | None
    ) -> torch.Tensor:
        first_hist = torch.bincount(
            first, weights=weights, minlength=self.bins_per_half
        )
        second_hist = torch.bincount(
            second, weights=weights, minlength=self.bins_per_half
        )
        return torch.cat([first_hist, second_hist]).to(torch.float64)

    @torch.no_grad()
    def raw_feature(
        self, theta: torch.Tensor, shots: int | None, generator: torch.Generator
    ) -> torch.Tensor:
        """Concatenated marginal histograms ``[h1; h2]`` of length ``feature_dim``.

        Parameters
        ----------
        theta : torch.Tensor
            Photonic parameters, shape ``(theta_dim,)``.
        shots : int or None
            Shot budget ``S``. ``None`` returns the infinite-shot marginals.
        generator : torch.Generator
            CPU generator for the multinomial shot sampling.
        """
        probabilities = self.outcome_probabilities(theta)
        if shots is None:
            return self._histograms(
                self.first_half_index, self.second_half_index, probabilities
            )
        samples = torch.multinomial(
            probabilities, shots, replacement=True, generator=generator
        )
        counts = self._histograms(
            self.first_half_index[samples], self.second_half_index[samples], None
        )
        return counts / shots

    @property
    def angles(self) -> torch.nn.Parameter:
        """The MerLin layer's beam-splitter angle parameter (length ``n_angles``)."""
        return self._angles

    def differentiable_exact_feature(self) -> torch.Tensor:
        """Infinite-shot ``[h1; h2]`` at the layer's current angles, inside autograd.

        Used by the MerLin-native adaptation: the marginals are sums of exact
        Fock-state probabilities (``index_add``), so gradients flow from ``z``
        back to the angles through MerLin's differentiable simulation.
        """
        probabilities = self.layer().reshape(-1)
        first = probabilities.new_zeros(self.bins_per_half).index_add(
            0, self.first_half_index, probabilities
        )
        second = probabilities.new_zeros(self.bins_per_half).index_add(
            0, self.second_half_index, probabilities
        )
        return torch.cat([first, second])

    @torch.no_grad()
    def mode_click_probabilities(self, theta: torch.Tensor) -> torch.Tensor:
        """Per-mode click probability ``P(b_j = 1)`` under ``p_theta``."""
        probabilities = self.outcome_probabilities(theta)
        return probabilities @ self.click_patterns.to(torch.float64)

    def raw_feature_from_bits(self, bits: torch.Tensor) -> torch.Tensor:
        """Apply Phi to explicit click patterns ``bits`` of shape ``(S, N)``.

        Used by the classical-histogram baseline (B6) so it shares the exact
        same feature map as the photonic source.
        """
        place_values = 2 ** torch.arange(self.half_modes - 1, -1, -1)
        bits = bits.to(torch.long)
        first = bits[:, : self.half_modes] @ place_values
        second = bits[:, self.half_modes :] @ place_values
        return self._histograms(first, second, None) / bits.shape[0]
