"""Photonic HQ-CNN variant using the MerLin photonic library.

The CNN backbone is identical to :class:`lib.models.HQCNN` so that
parameter-count and embedding pipelines stay comparable. Only the PQC is
swapped for a photonic ``merlin.QuantumLayer`` whose interferometer mixes
``n_modes`` modes with ``n_photons`` photons.

Architecture
------------

1. ``CNNBackbone`` (Conv -> BN -> ReLU -> MaxPool, 4 blocks)  ->  fc_dim
2. ``encoder`` :  Linear(fc_dim -> n_modes)
3. ``scale``   :  per-mode learnable scaling (``ScaleLayer``); replaces the
   ``pi * sigmoid`` analog used by the qubit variant.  The photonic layer
   already maps phases to ``[0, 2 pi)`` via the perceval ``PS`` parameters.
4. ``QuantumLayer`` :  Generic rectangular interferometer with trainable
   beam-splitter / phase-shifter parameters, mode-encoded input phase
   shifters, and a readout interferometer.  Configured via the
   :class:`merlin.ComputationSpace.UNBUNCHED` Fock-output projection.
5. ``head`` :  Linear(output_size -> num_classes)

This mirrors the structure used by
``papers/HQNN_MythOrReality/models/hqnn.py``.
"""

from __future__ import annotations

import math

import torch
from lib.models import CNNBackbone
from torch import nn


def _build_photonic_circuit(modes: int, num_features: int):
    """Generic rectangular interferometer + per-mode input PS + readout."""
    import perceval as pcvl

    interferometer = pcvl.GenericInterferometer(
        modes,
        lambda i: (
            pcvl.BS(theta=pcvl.P(f"bs_1_{i}"))
            // pcvl.PS(pcvl.P(f"ps_1_{i}"))
            // pcvl.BS(theta=pcvl.P(f"bs_2_{i}"))
            // pcvl.PS(pcvl.P(f"ps_2_{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    circuit = pcvl.Circuit(modes)
    circuit.add(0, interferometer, merge=True)

    variable_layers = pcvl.Circuit(modes)
    for index in range(num_features):
        px = pcvl.P(f"px-{index + 1}")
        variable_layers.add(index % modes, pcvl.PS(px))
    circuit.add(0, variable_layers, merge=True)

    readout = pcvl.GenericInterferometer(
        modes,
        lambda i: (
            pcvl.BS()
            // pcvl.PS(pcvl.P(f"ps_3_{i}"))
            // pcvl.BS()
            // pcvl.PS(pcvl.P(f"ps_4_{i}"))
        ),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    circuit.add(0, readout, merge=True)
    return circuit


class ScaleLayer(nn.Module):
    """Element-wise learnable scaling layer (initial uniform in [0, 2π))."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(2.0 * math.pi * torch.rand(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class HQCNNMerLin(nn.Module):
    """MerLin-backed photonic variant of HQ-CNN.

    Parameters
    ----------
    num_classes : int
    in_channels : int
    fc_dim : int
        CNN bottleneck width.
    image_size : int
    n_modes : int
        Photonic interferometer width.
    n_photons : int
        Number of photons (n_modes / 2 is a sensible upper bound).
    no_bunching : bool
        If True, use the unbunched (collision-free) computation space — this
        keeps the photonic output dimension at ``C(n_modes, n_photons)``.
    layer_factory : callable, optional
        Override for unit tests / mocking.
    device : torch.device or str
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        fc_dim: int = 4096,
        image_size: int = 224,
        n_modes: int = 6,
        n_photons: int = 2,
        no_bunching: bool = True,
        layer_factory=None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.backbone = CNNBackbone(in_channels, fc_dim, image_size)
        self.encoder = nn.Linear(fc_dim, n_modes)
        self.scale = ScaleLayer(n_modes)
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.no_bunching = no_bunching

        self.quantum_layer, self.output_size = self._build_layer(
            n_modes, n_photons, no_bunching, layer_factory, device
        )
        self.head = nn.Linear(self.output_size, num_classes)
        nn.init.constant_(self.head.bias, 0.0)

    @staticmethod
    def _build_layer(
        n_modes: int,
        n_photons: int,
        no_bunching: bool,
        layer_factory,
        device,
    ):
        if layer_factory is not None:
            layer = layer_factory(n_modes=n_modes, n_photons=n_photons)
            out = getattr(layer, "output_size", n_modes)
            return layer, out
        try:
            from merlin import ComputationSpace, QuantumLayer
        except ImportError as exc:
            raise ImportError(
                "MerLin photonic variant requires the 'merlinquantum' package. "
                "Install it with: pip install merlinquantum"
            ) from exc

        circuit = _build_photonic_circuit(n_modes, num_features=n_modes)

        # Input state: alternating-mode single photons (paper HQNN convention).
        input_state = [0] * n_modes
        for index in range(n_photons):
            input_state[2 * index] = 1

        comp_space = ComputationSpace.UNBUNCHED if no_bunching else ComputationSpace.FOCK
        layer = QuantumLayer(
            input_size=n_modes,
            circuit=circuit,
            trainable_parameters=[
                p.name for p in circuit.get_parameters() if not p.name.startswith("px")
            ],
            input_parameters=["px"],
            input_state=input_state,
            computation_space=comp_space,
            device=torch.device(device) if isinstance(device, str) else device,
        )
        return layer, layer.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        # Encoding interface: map CNN embedding to n_modes phase parameters.
        encoded = torch.sigmoid(self.encoder(h))  # in [0, 1]
        encoded = self.scale(encoded)             # in [0, 2pi) effectively
        q = self.quantum_layer(encoded)
        return self.head(q)
