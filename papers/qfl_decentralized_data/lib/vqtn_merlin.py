"""MerLin photonic adaptation of the VQTN classifier (Phase 4).

This is an **adaptation**, not the paper's model. It keeps the scientific role of the
VQTN in the paper's claims: a trainable quantum classifier on the same 16 downsampled
pixels whose parameters are averaged by FedAvg. The gate-level tree layout is not
kept. See LOG.md, "Phase 4 Adaptation Plan".

Circuit (8 modes, photons spread over the modes, threshold detection modelled by the
collision-free ``UNBUNCHED`` space):

* a fixed superposition layer first: MerLin's default cascade of 50:50 beam splitters
  on the neighbouring pairs (0, 1) to (6, 7), untrained;
* ``front`` placement: superposition, trainable mesh, encode even pixels (stage
  ``xa``), trainable mesh, encode odd pixels (``xb``), trainable mesh;
* ``replace`` placement: the superposition takes the place of the first trainable
  mesh, i.e. superposition, ``xa``, mesh, ``xb``, mesh.

Each pixel ``x`` in [0, 1] becomes one phase ``encoding_scale * pi * x`` on its mode.
The input columns feed the encoding stages in order (first 8 columns to ``xa``,
checked in ``tests/test_merlin.py``), so pixels are reordered even-then-odd, mirroring
the gate model's two pixels per qubit (pixel ``2j`` and ``2j + 1`` on mode ``j``).

Read-outs: ``linear_head`` maps the 8 post-selected click probabilities
(``mode_expectations`` in ``UNBUNCHED``) through a trainable linear layer, the analogue
of the hybrid VQTN's eight-<Z> head; ``lex_grouping`` folds the outcome distribution
into the classes with a fixed ``LexGrouping`` and has no classical parameters.
"""

from __future__ import annotations

import math

import merlin as ml
import torch
from torch import nn

N_PIXELS = 16
N_MODES = 8
# Photons spread over the modes so every trainable mode lies in their light cone.
INPUT_STATES = {2: [1, 0, 0, 0, 1, 0, 0, 0], 4: [1, 0, 1, 0, 1, 0, 1, 0]}
SUPERPOSITION_PLACEMENTS = ("front", "replace")
READOUTS = ("linear_head", "lex_grouping")
COMPUTATION_SPACES = {"unbunched": ml.ComputationSpace.UNBUNCHED}
LOG_PROBABILITY_FLOOR = 1e-8


def stage_inputs(pixels: torch.Tensor) -> torch.Tensor:
    """Reorder ``(batch, 16)`` pixels as even pixels (stage ``xa``) then odd (``xb``)."""
    return torch.cat([pixels[:, 0::2], pixels[:, 1::2]], dim=1)


class PhotonicClassifier(nn.Module):
    """Photonic classifier returning class log-probabilities.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    n_photons : int
        Photon number, 2 or 4 (see ``INPUT_STATES``).
    encoding_scale : float
        Phase per pixel is ``encoding_scale * pi * pixel``.
    superposition : str
        ``front`` (superposition before the first trainable mesh) or ``replace``
        (superposition instead of the first trainable mesh).
    readout : str
        ``linear_head`` or ``lex_grouping``.
    computation_space : str
        Only ``unbunched`` (threshold detection, collision-free post-selection).
        Default value is ``unbunched``.
    dtype : torch.dtype
        Real dtype of the layer. Default value is ``torch.float32``.

    Raises
    ------
    ValueError
        If an option is unsupported, or ``lex_grouping`` cannot split the outcomes
        evenly into ``n_classes`` groups.
    """

    def __init__(
        self,
        n_classes: int,
        n_photons: int,
        encoding_scale: float,
        superposition: str,
        readout: str,
        computation_space: str = "unbunched",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if n_photons not in INPUT_STATES:
            raise ValueError(
                f"n_photons must be one of {sorted(INPUT_STATES)}, got {n_photons}"
            )
        if superposition not in SUPERPOSITION_PLACEMENTS:
            raise ValueError(
                f"superposition must be one of {SUPERPOSITION_PLACEMENTS}, got {superposition!r}"
            )
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        if computation_space not in COMPUTATION_SPACES:
            raise ValueError(f"Unsupported computation_space {computation_space!r}")
        self.n_photons = n_photons
        self.encoding_scale = encoding_scale
        self.superposition = superposition
        self.readout = readout
        self.computation_space = COMPUTATION_SPACES[computation_space]

        builder = ml.CircuitBuilder(n_modes=N_MODES)
        builder.add_superpositions(name="sup")
        if superposition == "front":
            builder.add_entangling_layer(name="mesha")
        # Stage names must be letters only: a trailing digit makes MerLin ignore scale.
        builder.add_angle_encoding(
            modes=list(range(N_MODES)), name="xa", scale=encoding_scale * math.pi
        )
        builder.add_entangling_layer(name="meshb")
        builder.add_angle_encoding(
            modes=list(range(N_MODES)), name="xb", scale=encoding_scale * math.pi
        )
        builder.add_entangling_layer(name="meshc")
        if readout == "linear_head":
            strategy = ml.MeasurementStrategy.mode_expectations(self.computation_space)
        else:
            strategy = ml.MeasurementStrategy.probs(self.computation_space)
        self.layer = ml.QuantumLayer(
            input_size=N_PIXELS,
            builder=builder,
            input_state=INPUT_STATES[n_photons],
            n_photons=n_photons,
            measurement_strategy=strategy,
            dtype=dtype,
        )
        if readout == "linear_head":
            self.head = nn.Linear(N_MODES, n_classes, dtype=dtype)
        else:
            outcomes = self.layer.output_size
            if outcomes % n_classes:
                raise ValueError(
                    f"lex_grouping needs n_classes to divide the {outcomes} outcomes; "
                    f"{outcomes} % {n_classes} = {outcomes % n_classes}"
                )
            self.grouping = ml.LexGrouping(outcomes, n_classes)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return class log-probabilities, shape ``(batch, n_classes)``."""
        measured = self.layer(stage_inputs(pixels))
        if self.readout == "linear_head":
            return torch.log_softmax(self.head(measured), dim=1)
        return torch.log(self.grouping(measured) + LOG_PROBABILITY_FLOOR)
