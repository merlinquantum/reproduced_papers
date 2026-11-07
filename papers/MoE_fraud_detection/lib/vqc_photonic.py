"""MerLin photonic adaptation of the gate-model VQC (Phase 4 photonic
translation, see MERLIN_COOKBOOK.md and PAPER_REPRODUCTION_INSTRUCTIONS.md
Section 5).

Translation principle: preserve the *role* the VQC plays in the GQC pipeline
(angle-encode the 6-dimensional autoencoder latent, apply one trainable
entangling block, produce a single scalar fed to the classification head) —
not a literal gate-for-gate mapping of the 6-qubit Alternating Layered Ansatz.
Per the photonic-translation policy this reproduction is scoped under, the
photonic circuit uses >=2 photons (never a single-photon trivial baseline).

Design (see LOG.md "MerLin API Notes" for the full hardware-aware record):

- ``n_modes = n_qubits`` (default 6) - one photonic mode per latent feature,
  mirroring the gate model's one-qubit-per-feature layout.
- ``n_photons`` (default 3, configurable, must be >= 2) photons placed by
  :func:`spread_input_state`, which spreads them as evenly as possible across
  the modes rather than hardcoding a fixed pattern (photon placement is a
  modelling choice, cookbook Sec 0.5 point 1) so that changing ``n_modes`` /
  ``n_photons`` via config does not silently break.
- One trainable entangling layer (``add_entangling_layer``) before AND after
  a single angle-encoding pass (matching the gate model's "encoding occurs
  only once" choice) - a single trainable mesh is already a universal
  photonic transform (cookbook Sec 0.5 point 2), so we do not stack multiple
  copies.
- Measurement: ``MeasurementStrategy.probs(ComputationSpace.UNBUNCHED)``
  (threshold detection, the repo's hardware-aligned default) producing a
  ``C(n_modes, n_photons)``-dimensional probability vector, then
  ``LexGrouping`` down to 2 buckets; the probability mass of bucket 1 is used
  as the scalar fed into the same ``Head`` used by the gate-model branch, so
  the rest of the GQC pipeline (autoencoder, head, calibration, MoE router)
  is byte-for-byte shared between the gate and photonic branches.

MerLin API note: this reproduction runs against merlinquantum==0.4.1,
installed in this container. Two calls used by MERLIN_COOKBOOK.md's Pattern A
(written against 0.3.2) are deprecated in 0.4.x:
``ml.MeasurementStrategy.PROBABILITIES`` -> use
``ml.MeasurementStrategy.probs(computation_space)``; and
``ml.LexGrouping(output_size, num_classes)`` -> use
``ml.LexGrouping(input_size=..., output_size=...)`` (keyword names changed,
and it no longer takes the *upstream* quantum layer's output_size implicitly
- pass it explicitly). Recorded here and in
``ACTION_REQUIRED_SEND_TO_MERLIN_TEAM.md``.
"""

from __future__ import annotations

import merlin as ml
import torch
from torch import nn


def spread_input_state(n_modes: int, n_photons: int) -> list[int]:
    """Place ``n_photons`` photons as evenly as possible across ``n_modes``
    modes (one photon per mode until modes run out, spaced by
    ``n_modes // n_photons``), so every mode stays close to the photon
    light-cone rather than clustering photons in a fixed prefix.

    Raises ``ValueError`` if ``n_photons < 2`` (this reproduction's photonic
    circuits must use at least two photons - see module docstring) or if
    ``n_photons > n_modes``.
    """
    if n_photons < 2:
        raise ValueError(
            f"n_photons={n_photons} < 2: a single-photon circuit is a trivial "
            "linear-optical baseline and must not be used as the photonic "
            "implementation (see PAPER_REPRODUCTION_INSTRUCTIONS.md Sec 5)."
        )
    if n_photons > n_modes:
        raise ValueError(f"n_photons={n_photons} cannot exceed n_modes={n_modes}")

    state = [0] * n_modes
    step = n_modes / n_photons
    for k in range(n_photons):
        mode = int(round(k * step))
        mode = min(mode, n_modes - 1)
        while state[mode] == 1:  # extremely small n_modes edge case
            mode = (mode + 1) % n_modes
        state[mode] = 1
    return state


class PhotonicVQCLayer(nn.Module):
    """MerLin ``QuantumLayer`` wrapper standing in for the gate-model VQC.

    Interface-compatible with :class:`lib.vqc_gate.VQCLayer`: takes a
    ``(batch, n_modes)`` tensor of angle-encoding features (the autoencoder
    latent) and returns a ``(batch,)`` scalar in ``[0, 1]`` fed to the same
    :class:`lib.head.Head`.

    ``readout`` selects how the quantum layer's ``C(n_modes, n_photons)``-
    dimensional probability vector is collapsed to that scalar:

    - ``"fixed"`` (default): ``merlin.LexGrouping`` down to 2 buckets, a
      fixed (non-trainable) lexicographic grouping rule.
    - ``"trainable"``: a single trainable ``nn.Linear(output_size, 1)`` +
      sigmoid directly on the probability vector -- addresses the "fixed
      grouping versus trainable readout" pitfall in MERLIN_COOKBOOK.md's
      photonic debugging checklist (fixed grouping can discard exactly the
      class-discriminative structure the circuit learns to produce).
    """

    def __init__(
        self, n_qubits: int = 6, n_photons: int = 3, readout: str = "fixed"
    ) -> None:
        super().__init__()
        if readout not in {"fixed", "trainable"}:
            raise ValueError(f"readout must be 'fixed' or 'trainable', got {readout!r}")
        n_modes = n_qubits
        input_modes = list(range(n_modes))
        input_state = spread_input_state(n_modes, n_photons)

        builder = ml.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer()
        builder.add_angle_encoding(modes=input_modes, scale=1.0)
        builder.add_entangling_layer()

        self.quantum_layer = ml.QuantumLayer(
            input_size=len(input_modes),
            builder=builder,
            input_state=input_state,
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                ml.ComputationSpace.UNBUNCHED
            ),
        )

        output_size = self.quantum_layer.output_size
        if readout == "fixed" and output_size % 2 != 0:
            raise ValueError(
                f"PhotonicVQCLayer: output_size={output_size} (n_modes={n_modes}, "
                f"n_photons={n_photons}, UNBUNCHED) is not divisible by 2 - "
                "LexGrouping(2) would drop outcomes. Choose a different "
                "n_modes/n_photons pair."
            )
        self.readout = readout
        if readout == "fixed":
            self.grouping: nn.Module = ml.LexGrouping(
                input_size=output_size, output_size=2
            )
        else:
            self.grouping = nn.Sequential(nn.Linear(output_size, 1), nn.Sigmoid())

        self.n_modes = n_modes
        self.n_photons = n_photons
        self.input_state = input_state
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        probs = self.quantum_layer(x)
        if self.readout == "trainable":
            return self.grouping(probs).squeeze(-1)
        grouped = self.grouping(probs)
        return grouped[..., 1]


__all__ = ["PhotonicVQCLayer", "spread_input_state"]
