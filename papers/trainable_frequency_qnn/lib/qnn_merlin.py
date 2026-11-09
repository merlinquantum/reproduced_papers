"""MerLin photonic adaptation of the fixed- / trainable-frequency QNNs.

This is an **adaptation**, not the paper's model. The paper encodes data with
qubit rotations ``R_y(gamma_m w_m x)``; here each encoding mode carries a phase
shifter ``exp(i phi n)`` with ``phi = gamma_m w_m x``. The output is then a
Fourier series in ``x`` whose frequencies are photon-number differences weighted
by ``gamma_m w_m``: with ``w = 1`` they are integers (fixed-frequency, as in Gan
et al.), with trainable ``w`` they are learnable and non-integer (trainable
frequency). See LOG.md, "Phase 4 Adaptation Plan".

Circuit (``CircuitBuilder``), per network:

- ``toy``         : mesh, encoding, mesh
- ``ns_parallel`` : mesh, encoding, mesh, encoding (re-upload, shared w), mesh

A trainable MZI mesh must precede the first encoding so photons reach the
encoding modes. Readout is a fixed observable on threshold-detector clicks:
``o(s) = (#clicks on even modes) - (#clicks on odd modes)``, the analogue of the
paper's total magnetization, with no trainable readout.

Detection models (``readout``):

- ``threshold_clicks`` : FOCK probabilities grouped by click pattern, i.e.
  threshold detectors **without** post-selection. Keeps the finite Fourier
  spectrum of the circuit (primary).
- ``unbunched_clicks`` : MerLin's UNBUNCHED space, i.e. collision-free
  post-selection with renormalisation. The output becomes a ratio of
  trigonometric polynomials and leaks power above the circuit's spectrum
  (readout ablation only).
"""

from __future__ import annotations

import itertools
import math

import merlin as ml
import torch
from torch import nn

from .qnn import FREQUENCY_MODES, generator_coefficients

PHOTONIC_LAYOUTS = ("toy", "ns_parallel")
READOUTS = {
    "threshold_clicks": ml.ComputationSpace.FOCK,
    "unbunched_clicks": ml.ComputationSpace.UNBUNCHED,
}


def spread_input_state(n_modes: int, n_photons: int) -> list[int]:
    """Return one photon on every other mode, ``[1, 0, 1, 0, ...]``.

    Raises
    ------
    ValueError
        If the photons do not fit on the even modes.
    """
    if 2 * n_photons > n_modes:
        raise ValueError(
            f"{n_photons} photons cannot be spread over the even modes of {n_modes} modes."
        )
    return [
        1 if mode % 2 == 0 and mode // 2 < n_photons else 0 for mode in range(n_modes)
    ]


class PhotonicFrequencyModel(nn.Module):
    """Photonic FF / TF / FF+S model with a fixed threshold-click observable.

    Parameters
    ----------
    n_modes : int
        Number of optical modes.
    n_photons : int
        Number of photons (at least 2 for any reported result).
    n_inputs : int
        Number of input coordinates (1 for the toy task, 3 for (x, y, t)).
    layout : str
        ``"toy"`` or ``"ns_parallel"``.
    encoding_modes_per_input : int
        Number of phase-shifter modes encoding each input coordinate.
    feature_map : str
        ``"simple"``, ``"tower"`` or ``"exponential"`` fixed coefficients gamma_m.
    frequency_mode : str
        ``"fixed"``, ``"trainable"`` or ``"global_scale"``.
    frequency_init : float
        Initial theta_F / global scale, or the fixed multiplier for FF.
    readout : str
        ``"threshold_clicks"`` (primary) or ``"unbunched_clicks"`` (ablation).
    dtype : torch.dtype
        Real dtype (float64 recommended).
    generator : torch.Generator
        Seeded generator. Mesh phases are redrawn uniformly in [0, 2 pi) from it,
        so FF, FF+S and TF models with the same seed share their mesh
        initialisation (paired seeds); frequency parameters are not drawn.
    """

    def __init__(
        self,
        *,
        n_modes: int,
        n_photons: int,
        n_inputs: int,
        layout: str,
        encoding_modes_per_input: int,
        feature_map: str,
        frequency_mode: str,
        frequency_init: float,
        readout: str,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if layout not in PHOTONIC_LAYOUTS:
            raise ValueError(
                f"Unknown layout {layout!r}; expected one of {PHOTONIC_LAYOUTS}."
            )
        if frequency_mode not in FREQUENCY_MODES:
            raise ValueError(
                f"Unknown frequency_mode {frequency_mode!r}; expected one of {FREQUENCY_MODES}."
            )
        if readout not in READOUTS:
            raise ValueError(
                f"Unknown readout {readout!r}; expected one of {sorted(READOUTS)}."
            )
        if layout == "toy" and n_inputs != 1:
            raise ValueError("The toy layout encodes a single input coordinate.")
        n_encoding_modes = n_inputs * encoding_modes_per_input
        if n_encoding_modes > n_modes:
            raise ValueError(
                f"{n_encoding_modes} encoding modes exceed {n_modes} modes."
            )

        self.n_modes = n_modes
        self.n_photons = n_photons
        self.n_inputs = n_inputs
        self.layout = layout
        self.frequency_mode = frequency_mode
        self.readout = readout
        self.n_blocks = 1 if layout == "toy" else 2
        self.input_state = spread_input_state(n_modes, n_photons)

        # Encoding mode j encodes input j // encoding_modes_per_input, as the
        # qubit model assigns 2 qubits per coordinate in its parallel layout.
        self.encoding_input_index = [
            d for d in range(n_inputs) for _ in range(encoding_modes_per_input)
        ]
        gamma = generator_coefficients(feature_map, encoding_modes_per_input) * n_inputs
        self.register_buffer("gamma", torch.tensor(gamma, dtype=dtype))
        self.register_buffer(
            "encoding_input_index_tensor", torch.tensor(self.encoding_input_index)
        )
        encoding_modes = list(range(n_encoding_modes))

        builder = ml.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer()
        for block in range(self.n_blocks):
            builder.add_angle_encoding(modes=encoding_modes, name=f"enc{block}")
            builder.add_entangling_layer()
        self.layer = ml.QuantumLayer(
            input_size=n_encoding_modes * self.n_blocks,
            builder=builder,
            input_state=self.input_state,
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=READOUTS[readout]
            ),
            dtype=dtype,
        )
        with torch.no_grad():
            for parameter in self.layer.parameters():
                parameter.copy_(
                    torch.rand(
                        parameter.shape, generator=generator, dtype=parameter.dtype
                    )
                    * (2 * math.pi)
                )

        if frequency_mode == "trainable":
            self.frequency_weights = nn.Parameter(
                torch.full((n_encoding_modes,), frequency_init, dtype=dtype)
            )
        elif frequency_mode == "global_scale":
            self.input_scales = nn.Parameter(
                torch.full((n_inputs,), frequency_init, dtype=dtype)
            )
        else:
            self.register_buffer(
                "fixed_weight", torch.tensor(frequency_init, dtype=dtype)
            )

        signs = [1.0 if mode % 2 == 0 else -1.0 for mode in range(n_modes)]
        observable = [
            sum(sign for sign, count in zip(signs, key) if count > 0)
            for key in self.layer.output_keys
        ]
        self.register_buffer("click_observable", torch.tensor(observable, dtype=dtype))

    def encoding_frequencies(self) -> torch.Tensor:
        """Return the effective per-mode phase multipliers ``gamma_m * w_m``."""
        if self.frequency_mode == "trainable":
            return self.gamma * self.frequency_weights
        if self.frequency_mode == "global_scale":
            return self.gamma * self.input_scales[self.encoding_input_index_tensor]
        return self.gamma * self.fixed_weight

    def accessible_frequencies(self, decimals: int = 9) -> list[float]:
        """Return the positive frequencies of a single-input model.

        Phase shifters contribute ``exp(i x sum_m mu_m n_m)`` with ``mu_m``
        the encoding multipliers, so the gaps are ``|sum_m mu_m (n_m - n'_m)|``
        over photon-number patterns with at most ``n_photons`` photons in the
        encoding modes (one encoding block).
        """
        if self.n_inputs != 1:
            raise ValueError("accessible_frequencies is defined for a single input.")
        multipliers = [float(w) for w in self.encoding_frequencies().detach().cpu()]
        energies = set()
        for pattern in itertools.product(
            range(self.n_photons + 1), repeat=len(multipliers)
        ):
            if sum(pattern) <= self.n_photons:
                energies.add(sum(m * n for m, n in zip(multipliers, pattern)))
        gaps = {round(abs(a - b), decimals) for a in energies for b in energies}
        return sorted(g for g in gaps if g > 0)

    def n_frequency_parameters(self) -> int:
        """Number of trainable feature-map parameters (|theta_F| or scales)."""
        if self.frequency_mode == "trainable":
            return self.frequency_weights.numel()
        if self.frequency_mode == "global_scale":
            return self.input_scales.numel()
        return 0

    def n_ansatz_parameters(self) -> int:
        """Number of trainable mesh phases."""
        return sum(p.numel() for p in self.layer.parameters())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the click observable for a batch of inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Real tensor of shape ``(batch, n_inputs)``.

        Returns
        -------
        torch.Tensor
            Outputs of shape ``(batch,)``.
        """
        if inputs.dim() != 2 or inputs.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected inputs of shape (batch, {self.n_inputs}), got {tuple(inputs.shape)}."
            )
        phases = (
            inputs[:, self.encoding_input_index_tensor] * self.encoding_frequencies()
        )
        # Every encoding block re-uploads the same phases (shared theta_F).
        probabilities = self.layer(phases.repeat(1, self.n_blocks))
        return probabilities @ self.click_observable
