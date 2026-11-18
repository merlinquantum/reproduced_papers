"""MerLin photonic adaptation of the paper's quantum models (Phase 4, an adaptation, not the reproduction).

Photonic counterpart of the paper's model ``f(x) = <0|U(x)^dagger M U(x)|0>``:

    U(x) = W^(L+1) S(x) W^(L) ... S(x) W^(1),   S(x) = exp(i s x sum_j k_j n_{E_j})

- ``W`` is a trainable MZI mesh on all modes (``CircuitBuilder.add_entangling_layer``).
  The bare MerLin 0.2.3 mesh has only ``m(m-1)`` independent phases, short of
  the ``m^2 - 1`` of a general mode unitary. It lacks the input-side relative
  phases. With ``phase_screens``, every mesh that follows an encoding is
  preceded by a trainable phase screen on modes ``1..m-1`` (``add_rotations``),
  and screen plus mesh is a universal mode unitary (Jacobian rank ``m^2 - 1``,
  tested). The input mesh needs no screen: input-side phases on a Fock product
  state are a global phase. Phases just before detection do not change
  probabilities.
- ``S(x)`` puts phase ``s k_j x`` on encoded mode ``E_j``. Its generator is
  ``s sum_j k_j n_{E_j}``, the photonic version of the paper's encoding
  Hamiltonian and of its continuous-variable phase-shift remark (Section
  III.C). With ``k_j = 1`` on a single mode, ``n`` photons give eigenvalues
  ``{0, ..., n}`` and ``L`` layers give degree ``n L``. With one photon and
  ``k_j = j`` on modes ``1..m-1``, the generator is ``diag(0, 1, ..., m-1)``:
  the setting of the paper's universality theorem.
- MerLin 0.2.3 mishandles several ``add_angle_encoding`` calls with non-unit
  ``scale`` (``IndexError``, or silently wrong spectra; LOG.md, Phase 4
  review). Each layer therefore uses one call at scale 1, and the inputs are
  pre-scaled to ``s k_j x``.
- The input is ``n`` single photons in the first ``n`` modes.

Readouts:

- ``"threshold_mode0"``: ``f(x) = 2 P(no click in mode 0) - 1``, a fixed
  ``+-1`` observable, the counterpart of the paper's ``<Z>``;
- ``"trainable_linear"``: a zero-initialised linear map over all click-pattern
  probabilities, i.e. a trainable observable (paper Section V, practical
  insight 5).

FOCK probabilities are grouped by click pattern, which emulates threshold
detection exactly. MerLin's ``UNBUNCHED`` space is supported only as a
contrast: it postselects collision-free outcomes and renormalises, so its
outputs are ratios of trigonometric polynomials and are not band-limited.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import merlin as ml
import torch
from torch import nn

from .fourier import predicted_spectrum

COMPUTATION_SPACES = {
    "fock": ml.ComputationSpace.FOCK,
    "unbunched": ml.ComputationSpace.UNBUNCHED,
}
READOUTS = ("threshold_mode0", "trainable_linear")


def single_photon_input_state(n_photons: int, n_modes: int) -> list[int]:
    """Return ``[1] * n + [0] * (m - n)``: one photon in each of the first ``n`` modes.

    Raises
    ------
    ValueError
        If ``n_photons < 1`` or there are not more modes than photons. One
        empty mode is required so that mode 0 can be found empty.
    """
    if n_photons < 1:
        raise ValueError("n_photons must be at least 1")
    if n_modes <= n_photons:
        raise ValueError(
            f"need n_modes > n_photons, got n_modes={n_modes}, n_photons={n_photons}"
        )
    return [1] * n_photons + [0] * (n_modes - n_photons)


def photonic_generator_eigenvalues(
    n_photons: int,
    n_modes: int,
    encoded_modes: Sequence[int],
    mode_scales: Sequence[float],
    scaling: float,
) -> list[float]:
    """Return the distinct eigenvalues of ``s sum_j k_j n_{E_j}`` on the ``n``-photon Fock space.

    The generator is diagonal in the Fock basis, so its eigenvalues are its
    values on every occupation with ``sum_k n_k = n``.
    """
    values = set()
    for occupation in itertools.product(range(n_photons + 1), repeat=n_modes):
        if sum(occupation) == n_photons:
            value = scaling * sum(
                k * occupation[mode] for mode, k in zip(encoded_modes, mode_scales)
            )
            values.add(round(value, 9))
    return sorted(values)


def photonic_predicted_spectrum(model_cfg: Mapping[str, Any]) -> list[float]:
    """Return the predicted spectrum ``{Lambda_k - Lambda_j}`` of a photonic model config."""
    eigenvalues = photonic_generator_eigenvalues(
        int(model_cfg["n_photons"]),
        int(model_cfg["n_modes"]),
        list(model_cfg["encoded_modes"]),
        [float(k) for k in model_cfg["mode_scales"]],
        float(model_cfg["encoding_scaling"]),
    )
    return predicted_spectrum(eigenvalues, int(model_cfg["encoding_layers"])).tolist()


def photonic_positive_frequencies(model_cfg: Mapping[str, Any]) -> list[int]:
    """Return the positive integer frequencies (in units of ``s``) of the predicted spectrum.

    Raises
    ------
    ValueError
        If the predicted spectrum, in units of ``s``, is not integer.
    """
    scaling = float(model_cfg["encoding_scaling"])
    frequencies = []
    for value in photonic_predicted_spectrum(model_cfg):
        in_units = value / scaling
        if abs(in_units - round(in_units)) > 1e-9:
            raise ValueError(
                f"non-integer frequency {in_units} in units of the scaling"
            )
        if round(in_units) > 0:
            frequencies.append(round(in_units))
    return frequencies


def _click_grouping(
    output_keys: Sequence[tuple[int, ...]], dtype: torch.dtype
) -> tuple[torch.Tensor, list]:
    """Return the matrix summing outcome probabilities into click patterns, and the patterns."""
    patterns = sorted({tuple(int(n > 0) for n in key) for key in output_keys})
    grouping = torch.zeros(len(output_keys), len(patterns), dtype=dtype)
    for row, key in enumerate(output_keys):
        grouping[row, patterns.index(tuple(int(n > 0) for n in key))] = 1.0
    return grouping, patterns


class PhotonicFourierModel(nn.Module):
    """MerLin model with ``L`` phase-encoding layers between trainable meshes.

    Parameters
    ----------
    n_photons : int
        Number of single photons ``n`` in the input state.
    n_modes : int
        Number of modes ``m`` (must exceed ``n``).
    encoded_modes : Sequence[int]
        Modes ``E`` that receive the data phase in every encoding layer.
    mode_scales : Sequence[float]
        Scale ``k_j`` of each encoded mode (phase ``s k_j x``).
    encoding_layers : int
        Number of encoding layers ``L`` (serial re-uploading).
    encoding_scaling : float
        Global input scaling ``s``.
    phase_screens : bool
        Add a trainable phase screen on modes ``1..m-1`` after every mesh.
    readout : str
        ``"threshold_mode0"`` or ``"trainable_linear"``.
    computation_space : str
        ``"fock"`` (main mapping) or ``"unbunched"`` (postselected contrast).
    dtype : torch.dtype
        Real dtype of the simulation.
    """

    def __init__(
        self,
        n_photons: int,
        n_modes: int,
        encoded_modes: Sequence[int],
        mode_scales: Sequence[float],
        encoding_layers: int,
        encoding_scaling: float,
        phase_screens: bool,
        readout: str,
        computation_space: str,
        dtype: torch.dtype,
    ):
        super().__init__()
        if encoding_layers < 1:
            raise ValueError("encoding_layers must be at least 1")
        if computation_space not in COMPUTATION_SPACES:
            raise ValueError(
                f"computation_space must be one of {sorted(COMPUTATION_SPACES)}"
            )
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}")
        encoded_modes = [int(mode) for mode in encoded_modes]
        if not encoded_modes or any(not 0 <= mode < n_modes for mode in encoded_modes):
            raise ValueError(
                f"encoded_modes must be a non-empty subset of range({n_modes})"
            )
        if len(set(encoded_modes)) != len(encoded_modes):
            raise ValueError("encoded_modes must not repeat a mode")
        if len(mode_scales) != len(encoded_modes):
            raise ValueError("mode_scales needs one scale per encoded mode")
        self.n_photons = n_photons
        self.n_modes = n_modes
        self.encoded_modes = encoded_modes
        self.mode_scales = [float(k) for k in mode_scales]
        self.encoding_layers = encoding_layers
        self.encoding_scaling = float(encoding_scaling)
        self.phase_screens = bool(phase_screens)
        self.readout = readout
        self.computation_space = computation_space
        self.input_state = single_photon_input_state(n_photons, n_modes)

        builder = ml.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer()
        for layer in range(encoding_layers):
            builder.add_angle_encoding(
                modes=encoded_modes, name=f"x{layer}_", scale=1.0
            )
            self._add_screened_mesh(builder, f"l{layer}")
        self.layer = ml.QuantumLayer(
            input_size=encoding_layers * len(encoded_modes),
            builder=builder,
            input_state=self.input_state,
            n_photons=n_photons,
            computation_space=COMPUTATION_SPACES[computation_space],
            dtype=dtype,
        )
        input_scales = torch.tensor(self.mode_scales * encoding_layers, dtype=dtype)
        self.register_buffer(
            "input_scales", self.encoding_scaling * input_scales, persistent=False
        )
        self.output_keys = [
            tuple(int(n) for n in key) for key in self.layer.output_keys
        ]
        grouping, patterns = _click_grouping(self.output_keys, dtype)
        self.click_patterns = patterns
        self.register_buffer("click_grouping", grouping, persistent=False)
        mode0_empty = torch.tensor(
            [pattern[0] == 0 for pattern in patterns], dtype=dtype
        )
        self.register_buffer("mode0_empty", mode0_empty, persistent=False)
        self.readout_layer = (
            nn.Linear(len(patterns), 1, dtype=dtype)
            if readout == "trainable_linear"
            else None
        )

    def _add_screened_mesh(self, builder: ml.CircuitBuilder, tag: str) -> None:
        """Add an optional input-side phase screen, then a mesh (universal together)."""
        if self.phase_screens and self.n_modes > 1:
            builder.add_rotations(
                modes=list(range(1, self.n_modes)),
                trainable=True,
                name=f"screen_{tag}_",
            )
        builder.add_entangling_layer()

    def reset_parameters(self, generator: torch.Generator, high: float) -> None:
        """Draw every circuit phase from ``U[0, high)`` (same range as the qubit models); zero the readout."""
        with torch.no_grad():
            for parameter in self.layer.parameters():
                parameter.copy_(
                    torch.rand(
                        parameter.shape, generator=generator, dtype=parameter.dtype
                    )
                    * high
                )
            if self.readout_layer is not None:
                self.readout_layer.weight.zero_()
                self.readout_layer.bias.zero_()

    def probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return outcome probabilities ``(batch, len(output_keys))`` for inputs ``(batch,)``."""
        return self.layer(inputs[:, None] * self.input_scales[None, :])

    def click_probabilities(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return threshold-detector click-pattern probabilities ``(batch, len(click_patterns))``."""
        return self.probabilities(inputs) @ self.click_grouping

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the readout ``f(x)`` for inputs ``(batch,)``."""
        clicks = self.click_probabilities(inputs)
        if self.readout_layer is None:
            return 2 * (clicks @ self.mode0_empty) - 1
        return self.readout_layer(clicks).squeeze(-1)


def build_photonic_model(
    model_cfg: Mapping[str, Any], dtype: torch.dtype
) -> PhotonicFourierModel:
    """Build a ``PhotonicFourierModel`` from a config section.

    Raises
    ------
    KeyError
        If a required key is missing (``n_photons``, ``n_modes``,
        ``encoded_modes``, ``mode_scales``, ``encoding_layers``,
        ``encoding_scaling``, ``phase_screens``, ``readout``,
        ``computation_space``).
    """
    return PhotonicFourierModel(
        int(model_cfg["n_photons"]),
        int(model_cfg["n_modes"]),
        list(model_cfg["encoded_modes"]),
        list(model_cfg["mode_scales"]),
        int(model_cfg["encoding_layers"]),
        float(model_cfg["encoding_scaling"]),
        bool(model_cfg["phase_screens"]),
        str(model_cfg["readout"]),
        str(model_cfg["computation_space"]),
        dtype,
    )
