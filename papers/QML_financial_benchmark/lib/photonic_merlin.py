"""MerLin photonic adaptations (Phase 4): a multi-photon fidelity kernel and a photonic QNN.

These are adaptations, not the paper's method (which is gate-based, see
``models.py`` and ``kernels.py``). Both pass phases explicitly: MerLin 0.4.1
applies the raw input of ``add_angle_encoding`` as the phase and ignores its
``scale`` argument (known-answer test in ``LOG.md``, Phase 4 plan).

Photonic conventions used here:

- Input state: ``n`` single photons spread evenly over ``m`` modes (photon at
  mode ``round(k * m / n)``), so every mesh acts on photons from the start.
- Kernel: amplitudes in the FOCK space give the exact overlap
  ``|<psi(x)|psi(x')>|^2``. On hardware this equals the probability of
  detecting the (collision-free) input pattern after ``U(x')^dagger U(x)``,
  which threshold detectors resolve.
- Classifier: UNBUNCHED probabilities (threshold detection).
"""

from __future__ import annotations

import merlin as ml
import numpy as np
import torch
from torch import nn


def spread_input_state(n_modes: int, n_photons: int) -> list[int]:
    """``n_photons`` single photons evenly spread over ``n_modes`` modes.

    Raises
    ------
    ValueError
        If the photons do not fit as single photons or fewer than 2 are requested
        (single-photon circuits are linear-optical baselines, not adaptations).
    """
    if n_photons < 2 or n_photons > n_modes:
        raise ValueError(
            f"Need 2 <= n_photons <= n_modes, got n_photons={n_photons}, n_modes={n_modes}"
        )
    state = [0] * n_modes
    for photon in range(n_photons):
        state[int(round(photon * n_modes / n_photons))] = 1
    if sum(state) != n_photons:
        raise ValueError(
            f"Photon placement collided for n_modes={n_modes}, n_photons={n_photons}"
        )
    return state


def reuploading_builder(
    n_modes: int, layers: int, trainable: bool, seed: int | None
) -> ml.CircuitBuilder:
    """``[mesh, angle encoding] x layers`` (+ a final trainable mesh when ``trainable``).

    Encoding blocks are named ``x0, x1, ...``; each takes one phase per mode.
    Fixed meshes use ``seed + layer`` so the kernel feature map is reproducible.
    """
    builder = ml.CircuitBuilder(n_modes=n_modes)
    for layer in range(layers):
        builder.add_entangling_layer(
            trainable=trainable,
            name=f"U{layer}",
            seed=None if seed is None else seed + layer,
        )
        builder.add_angle_encoding(modes=list(range(n_modes)), name=f"x{layer}")
    if trainable:
        builder.add_entangling_layer(trainable=True, name=f"U{layers}")
    return builder


def photonic_feature_states(
    features: np.ndarray,
    n_photons: int,
    layers: int,
    angle_scale: float,
    mesh_seed: int,
    batch_size: int = 1024,
) -> torch.Tensor:
    """FOCK-space amplitudes of the re-uploading photonic feature map, one row per sample.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n, d)`` inputs scaled to ``[0, 1]``; the circuit has ``m = d`` modes.
    n_photons : int
        Number of photons (>= 2).
    layers : int
        Number of ``fixed mesh -> encoding`` blocks (>= 1; with 0 the encoding
        acts on a Fock state and only adds a global phase).
    angle_scale : float
        Phase per unit feature.
    mesh_seed : int
        Seed of the fixed random meshes.
    batch_size : int
        Rows per MerLin forward call. Default value is 1024.
    """
    if layers < 1:
        raise ValueError("Photonic kernel needs layers >= 1")
    n_modes = features.shape[1]
    layer = ml.QuantumLayer(
        input_size=n_modes * layers,
        builder=reuploading_builder(n_modes, layers, trainable=False, seed=mesh_seed),
        input_state=spread_input_state(n_modes, n_photons),
        n_photons=n_photons,
        measurement_strategy=ml.MeasurementStrategy.amplitudes(
            computation_space=ml.ComputationSpace.FOCK
        ),
        dtype=torch.float64,
    )
    phases = (
        torch.as_tensor(features, dtype=torch.float64).repeat(1, layers) * angle_scale
    )
    with torch.no_grad():
        return torch.cat(
            [
                layer(phases[start : start + batch_size])
                for start in range(0, len(phases), batch_size)
            ]
        )


class PhotonicQNNClassifier(nn.Module):
    """Photonic analogue of the four QNN families: {non-hybrid, hybrid} x {SQ', MQ'} readout.

    Parameters
    ----------
    n_features : int
        Input dimension ``d``.
    n_modes : int
        Circuit modes ``m``; non-hybrid models need ``m == d``.
    n_photons : int
        Photons (>= 2).
    layers : int
        Data re-uploads ``L`` between trainable meshes.
    hybrid : bool
        If True, ``Linear(d, m)`` produces the phases; otherwise phases are
        ``angle_scale * x`` for ``[0, 1]``-scaled ``x``.
    multi_mode_readout : bool
        If True, ``Linear(outputs, 1)`` on the UNBUNCHED distribution (MQ');
        otherwise the probability that the last mode is occupied (SQ').
    angle_scale : float
        Phase per unit feature for non-hybrid models.
    """

    def __init__(
        self,
        n_features: int,
        n_modes: int,
        n_photons: int,
        layers: int,
        hybrid: bool,
        multi_mode_readout: bool,
        angle_scale: float,
    ):
        super().__init__()
        if not hybrid and n_modes != n_features:
            raise ValueError(
                f"Non-hybrid photonic QNN needs n_modes == n_features ({n_features}), got {n_modes}"
            )
        self.layers = layers
        self.angle_scale = angle_scale
        self.preprocess = nn.Linear(n_features, n_modes) if hybrid else None
        self.circuit = ml.QuantumLayer(
            input_size=n_modes * layers,
            builder=reuploading_builder(n_modes, layers, trainable=True, seed=None),
            input_state=spread_input_state(n_modes, n_photons),
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.UNBUNCHED
            ),
        )
        last_mode_occupied = [
            index for index, key in enumerate(self.circuit.output_keys) if key[-1] == 1
        ]
        self.register_buffer("last_mode_mask", torch.zeros(self.circuit.output_size))
        self.last_mode_mask[last_mode_occupied] = 1.0
        self.readout = (
            nn.Linear(self.circuit.output_size, 1) if multi_mode_readout else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        phases = (
            self.preprocess(inputs)
            if self.preprocess is not None
            else self.angle_scale * inputs
        )
        probabilities = self.circuit(phases.repeat(1, self.layers)).to(inputs.dtype)
        if self.readout is not None:
            return self.readout(probabilities).squeeze(-1)
        occupied = (probabilities @ self.last_mode_mask.to(inputs.dtype)).clamp(
            1e-6, 1 - 1e-6
        )
        return torch.log(occupied) - torch.log1p(-occupied)
