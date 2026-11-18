"""Photonic (MerLin) Q-function: a linear-optical adaptation of the paper's PQC.

This is a Phase 4 *adaptation*, not part of the reproduction. It keeps the
scientific role of the paper's circuit (bounded per-action outputs whose range
is matched by output weights, data re-uploading with trainable input scaling)
and replaces the qubit gates by linear optics (LOG.md, Phase 4 Photonic
Assessment and Adaptation Plan).

Circuit on ``n_modes`` modes with ``n_photons`` photons (default 6 modes,
3 photons, input state ``[1, 0, 1, 0, 1, 0]``)::

    layer 0, then for l = 1..L: [encoding phases], layer l

* layer = brickwork of Mach-Zehnder interferometers on neighbouring modes,
  first on (0,1), (2,3), (4,5), then on (1,2), (3,4); each MZI is
  ``BS, PS(phi_a), BS, PS(phi_b)``. This is the linear-optical counterpart of
  the hardware-efficient layer (local trainable operations + nearest-neighbour
  coupling). A layer must precede the first encoding because phase shifters
  do not change an input Fock state, and one must follow the last encoding
  because phases right before detection do not change click probabilities.
* encoding = phase shifters on modes ``0..n_inputs-1``. Cart Pole: phase
  ``activation(x_i * w_d[l, i])`` (arctan by default, paper eq. 10), in every
  layer with data re-uploading, otherwise only before layer 1. Frozen Lake:
  phase ``pi * bit_i`` (bit 0 = most significant, as the qubit basis state),
  encoded once as in the paper; ``data_reuploading=True`` re-injects the same
  phases before every layer (Phase 4 diagnostic, not the paper's model).
  ``state_encoding="dual_rail_input"`` (Phase 4 design study) encodes the
  Frozen Lake state in the *input photons* instead: bit i puts photon i in
  mode 2i (bit 0) or 2i+1 (bit 1), the photonic twin of the qubit basis state.
  MerLin 0.4 cannot combine a per-sample input state with per-sample phase
  inputs, so the state is prepared exactly from the fixed input
  ``[1,0,1,0,...]`` by one MZI per pair (``BS, PS(x_i), BS``) with
  ``x_i = pi * (1 - bit_i)``: phase 0 swaps the photon, phase pi leaves it.
  The preparation replaces layer 0; ``n_layers + 1`` brickwork layers follow
  and the state is encoded once.
* readout: the click probability of a threshold detector is a probability in
  [0, 1], the role of ``(<Z> + 1) / 2`` in the qubit model:
  ``Q(s, a) = P(click on readout_modes[a]) * w_o[a] * factor``.
  ``readout="threshold_click"`` (default) simulates the full Fock space and
  collapses each outcome to click / no-click per mode
  (``MeasurementStrategy.probs(FOCK, occupancy_readout=True)``): threshold
  detectors with no post-selection. ``readout="unbunched_click"`` uses
  MerLin's unbunched space, which keeps only collision-free outcomes and
  renormalises them, i.e. post-selects on ``n_photons`` clicks (checked:
  expected clicks then always sum to ``n_photons``).

All circuit phases are passed to MerLin as *input* parameters, one row per
(agent, sample). The trainable tensors therefore keep the leading agent
dimension used everywhere else in this package, so the masked optimizer,
target-network copy and lockstep training loop are reused unchanged, and all
agents are simulated in one batched MerLin call.
"""

from __future__ import annotations

import math
from typing import Any

import merlin as ml
import perceval as pcvl
import torch
from torch import nn

from .config_access import require, require_choice
from .models import ENCODING_ACTIVATIONS, ParameterGroup, _uniform

STATE_ENCODINGS_MERLIN = ("continuous_phase", "basis_phase", "dual_rail_input")
READOUTS_MERLIN = ("threshold_click", "unbunched_click")
MZI_PHASES = 2


def brickwork_pairs(n_modes: int) -> list[int]:
    """Upper mode of every MZI in one brickwork layer (even pairs, then odd pairs)."""
    return list(range(0, n_modes - 1, 2)) + list(range(1, n_modes - 1, 2))


def phases_per_layer(n_modes: int) -> int:
    """Trainable phases in one brickwork layer."""
    return MZI_PHASES * len(brickwork_pairs(n_modes))


def build_circuit(
    n_modes: int,
    n_layers: int,
    n_inputs: int,
    data_reuploading: bool,
    dual_rail_input: bool = False,
) -> tuple[pcvl.Circuit, int, int]:
    """Perceval circuit with mesh phases ``w*`` and encoding phases ``x*``.

    With ``dual_rail_input=True`` the encoding is one preparation MZI per mode
    pair (internal phase ``x_i``) in place of layer 0, followed by
    ``n_layers + 1`` brickwork layers (default value is False).

    Parameter names are zero-padded so that their lexicographic order equals
    their creation order (mesh phases layer by layer, encoding phases layer by
    layer, mode by mode); ``MerlinQFunction`` builds its input tensors in the
    same order.

    Returns
    -------
    tuple
        ``(circuit, n_mesh_phases, n_encoding_phases)``.
    """
    circuit = pcvl.Circuit(n_modes)
    mesh_index = 0
    encoding_index = 0

    def add_layer() -> None:
        nonlocal mesh_index
        for upper_mode in brickwork_pairs(n_modes):
            first = pcvl.P(f"w{mesh_index:05d}")
            second = pcvl.P(f"w{mesh_index + 1:05d}")
            mzi = (
                pcvl.Circuit(2)
                // pcvl.BS()
                // (0, pcvl.PS(first))
                // pcvl.BS()
                // (0, pcvl.PS(second))
            )
            circuit.add(upper_mode, mzi)
            mesh_index += MZI_PHASES

    if dual_rail_input:
        for pair in range(n_inputs):
            preparation = (
                pcvl.Circuit(2)
                // pcvl.BS()
                // (0, pcvl.PS(pcvl.P(f"x{pair:05d}")))
                // pcvl.BS()
            )
            circuit.add(2 * pair, preparation)
        for _ in range(n_layers + 1):
            add_layer()
        return circuit, mesh_index, n_inputs

    add_layer()
    for layer in range(1, n_layers + 1):
        if data_reuploading or layer == 1:
            for mode in range(n_inputs):
                circuit.add(mode, pcvl.PS(pcvl.P(f"x{encoding_index:05d}")))
                encoding_index += 1
        add_layer()
    return circuit, mesh_index, encoding_index


class MerlinQFunction(nn.Module):
    """Linear-optical Q-function, one independent circuit per agent.

    Parameters
    ----------
    cfg : dict
        Resolved configuration. Reads ``model.n_modes``, ``model.n_photons``,
        ``model.input_state``, ``model.n_layers``, ``model.state_encoding``,
        ``model.encoding_activation`` (continuous only),
        ``model.data_reuploading``, ``model.train_input_weights``,
        ``model.train_output_weights``, ``model.output_factor``,
        ``model.readout``, ``model.readout_modes`` and
        ``model.theta_init.{low,high}``.
    seeds : list of int
        One seed per agent; agent ``a`` draws its phases from ``seeds[a]``.
    n_actions : int
        Number of Q-values per state; must equal ``len(readout_modes)``.
    real_dtype : torch.dtype
        Real precision of parameters and outputs.

    Raises
    ------
    ValueError
        If the configuration is inconsistent (input state, readout modes,
        number of encoded inputs, encoding options).
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        seeds: list[int],
        n_actions: int,
        real_dtype: torch.dtype,
    ):
        super().__init__()
        self.n_agents = len(seeds)
        self.n_modes = int(require(cfg, "model.n_modes"))
        self.n_photons = int(require(cfg, "model.n_photons"))
        self.n_layers = int(require(cfg, "model.n_layers"))
        self.state_encoding = require_choice(
            cfg, "model.state_encoding", STATE_ENCODINGS_MERLIN
        )
        self.output_factor = float(require(cfg, "model.output_factor"))
        self.real_dtype = real_dtype

        input_state = [int(value) for value in require(cfg, "model.input_state")]
        if len(input_state) != self.n_modes or sum(input_state) != self.n_photons:
            raise ValueError(
                f"model.input_state {input_state} does not match n_modes={self.n_modes}, "
                f"n_photons={self.n_photons}"
            )
        if self.n_photons < 2:
            raise ValueError(
                "A photonic adaptation needs at least two photons (single-photon "
                "circuits are a linear-optical baseline)"
            )
        self.input_state = input_state

        readout_modes = [int(mode) for mode in require(cfg, "model.readout_modes")]
        if len(readout_modes) != n_actions:
            raise ValueError(
                f"model.readout_modes has {len(readout_modes)} modes, environment has "
                f"{n_actions} actions"
            )
        if len(set(readout_modes)) != len(readout_modes) or not all(
            0 <= mode < self.n_modes for mode in readout_modes
        ):
            raise ValueError(f"Invalid model.readout_modes {readout_modes}")
        self.register_buffer(
            "readout_index", torch.tensor(readout_modes, dtype=torch.long)
        )

        if self.state_encoding == "continuous_phase":
            self.encoding_activation = require_choice(
                cfg, "model.encoding_activation", ENCODING_ACTIVATIONS
            )
            self.data_reuploading = bool(require(cfg, "model.data_reuploading"))
            train_input_weights = bool(require(cfg, "model.train_input_weights"))
            self.n_inputs = 4
        else:
            self.encoding_activation = None
            # Re-uploading the basis phases is not in the paper's Frozen Lake model; it is
            # the Phase 4 diagnostic for the expressivity limit of a single phase encoding
            # in linear optics (LOG.md, Phase 4 amendment).
            self.data_reuploading = bool(require(cfg, "model.data_reuploading"))
            train_input_weights = False
            if bool(require(cfg, "model.train_input_weights")):
                raise ValueError(f"{self.state_encoding} encoding has no input weights")
            self.n_inputs = 4
            if self.state_encoding == "dual_rail_input":
                if self.data_reuploading:
                    raise ValueError("dual_rail_input encodes the state once")
                if (
                    self.n_modes != 2 * self.n_inputs
                    or input_state != [1, 0] * self.n_inputs
                ):
                    raise ValueError(
                        "dual_rail_input needs n_modes = 8 and input_state [1,0,1,0,1,0,1,0]"
                    )
        if self.n_inputs > self.n_modes:
            raise ValueError("More encoded inputs than modes")
        train_output_weights = bool(require(cfg, "model.train_output_weights"))

        self.readout = require_choice(cfg, "model.readout", READOUTS_MERLIN)
        if self.readout == "threshold_click":
            self.computation_space = ml.ComputationSpace.FOCK
            strategy = ml.MeasurementStrategy.probs(
                computation_space=self.computation_space, occupancy_readout=True
            )
        else:
            self.computation_space = ml.ComputationSpace.UNBUNCHED
            strategy = ml.MeasurementStrategy.mode_expectations(
                computation_space=self.computation_space
            )

        circuit, n_mesh_phases, n_encoding_phases = build_circuit(
            self.n_modes,
            self.n_layers,
            self.n_inputs,
            self.data_reuploading,
            dual_rail_input=self.state_encoding == "dual_rail_input",
        )
        self.n_mesh_phases = n_mesh_phases
        self.n_encoding_layers = n_encoding_phases // self.n_inputs
        # All phases are inputs: no MerLin-owned trainable parameters (see module docstring).
        self.layer = ml.QuantumLayer(
            circuit=circuit,
            input_state=input_state,
            n_photons=self.n_photons,
            trainable_parameters=[],
            input_parameters=["w", "x"],
            measurement_strategy=strategy,
            dtype=real_dtype,
        )
        if self.readout == "threshold_click":
            # Binary occupancy keys -> per-mode click matrix (outcomes x modes).
            keys = [tuple(int(v) for v in key) for key in self.layer.output_keys]
            if any(len(key) != self.n_modes or max(key) > 1 for key in keys):
                raise ValueError("Unexpected occupancy keys from the threshold readout")
            self.register_buffer(
                "click_matrix", torch.tensor(keys, dtype=real_dtype), persistent=False
            )

        theta_low = float(require(cfg, "model.theta_init.low"))
        theta_high = float(require(cfg, "model.theta_init.high"))
        self.theta = nn.Parameter(
            _uniform((n_mesh_phases,), theta_low, theta_high, seeds, real_dtype)
        )
        self.input_weights = nn.Parameter(
            torch.ones(
                self.n_agents,
                max(self.n_encoding_layers, 1),
                self.n_inputs,
                dtype=real_dtype,
            ),
            requires_grad=train_input_weights,
        )
        self.output_weights = nn.Parameter(
            torch.ones(self.n_agents, n_actions, dtype=real_dtype),
            requires_grad=train_output_weights,
        )
        self.register_buffer("bit_shifts", torch.arange(self.n_inputs - 1, -1, -1))

    def parameter_groups(self) -> list[ParameterGroup]:
        """Trainable tensors grouped by learning rate, as for ``PQCQFunction``."""
        groups = [ParameterGroup("mesh_phases", [self.theta], "main", amsgrad=True)]
        if self.input_weights.requires_grad and self.n_encoding_layers > 0:
            groups.append(
                ParameterGroup(
                    "input_weights", [self.input_weights], "input_weights", False
                )
            )
        if self.output_weights.requires_grad:
            groups.append(
                ParameterGroup(
                    "output_weights", [self.output_weights], "output_weights", False
                )
            )
        return groups

    def count_trainable_parameters(self) -> int:
        """Trainable parameters of one agent."""
        count = self.theta[0].numel()
        if self.input_weights.requires_grad and self.n_encoding_layers > 0:
            count += self.input_weights[0].numel()
        if self.output_weights.requires_grad:
            count += self.output_weights[0].numel()
        return count

    def encoding_phases(self, observations: torch.Tensor) -> torch.Tensor:
        """Encoding phases ``(A, B, n_encoding_layers * n_inputs)``, layer-major."""
        if self.state_encoding in ("basis_phase", "dual_rail_input"):
            indices = observations[..., 0].round().long()
            bits = ((indices[..., None] >> self.bit_shifts) & 1).to(self.real_dtype)
            if self.state_encoding == "dual_rail_input":
                return math.pi * (1.0 - bits)
            return (math.pi * bits).repeat(1, 1, self.n_encoding_layers)
        inputs = observations.to(self.real_dtype)
        scaled = inputs[:, :, None, :] * self.input_weights[:, None, :, :]
        if self.encoding_activation == "arctan":
            phases = torch.atan(scaled)
        else:
            phases = torch.tanh(scaled)
        return phases.reshape(phases.shape[0], phases.shape[1], -1)

    def click_probabilities(self, observations: torch.Tensor) -> torch.Tensor:
        """Click probability of every mode ``(A, B, n_modes)``, each in [0, 1]."""
        n_agents, batch = observations.shape[0], observations.shape[1]
        mesh = self.theta[:, None, :].expand(n_agents, batch, self.n_mesh_phases)
        encoding = self.encoding_phases(observations)
        output = self.layer(
            mesh.reshape(n_agents * batch, -1),
            encoding.reshape(n_agents * batch, -1),
        )
        if self.readout == "threshold_click":
            output = output @ self.click_matrix
        return output.reshape(n_agents, batch, self.n_modes)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Q-values ``(A, B, n_actions)`` from ``(A, B, 4)`` or ``(A, B, 1)`` inputs."""
        clicks = self.click_probabilities(observations)[..., self.readout_index]
        return clicks * self.output_weights[:, None, :] * self.output_factor
