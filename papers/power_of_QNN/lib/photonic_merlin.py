"""Native linear-optical adaptation of the paper's encoding study (Phase 04, MerLin).

This is an adaptation, not the paper's model. It keeps the paper's question (how the
data encoding shapes the Fisher spectrum, effective dimension and trainability at
matched d) without mapping qubits or gates one to one; dual-rail CNOTs would cost
extra modes and photons (user direction, LOG.md Phase 4 Adaptation Plan).

Chip: ``m = s_in + extra_modes`` modes, ``n = m // 2`` photons, input ``[1, 0, 1, 0, ...]``.

- Easy encoding: fixed balanced mixer ``B``, then ``PS(2 x_i)`` on modes ``0..s_in-1``.
- Hard encoding: ``[B, PS(2 x_i), B, PS(2 (pi - x_i)(pi - x_(i+1))) on mode i]`` repeated
  twice. This is the paper's ZZFeatureMap data map (first and second order terms,
  depth 2), with interference between fixed mixers in place of ZZ gates; the linear
  neighbour pattern follows the paper's hardware circuit (App. C.4).
- Trainable mesh ``W(theta)``: rectangular brick of MZIs (``BS(theta_k) // PS(theta_(k+1))``
  on modes ``(i, i+1)``), wrapping over columns until exactly ``d`` parameters.
- Readout: class 1 = odd number of threshold clicks on the odd modes. ``threshold``
  simulates in the FOCK space and keeps bunched events as clicks; ``unbunched`` uses
  MerLin's UNBUNCHED space, which postselects collision-free outcomes and renormalizes.

Data phases and trainable phases enter as MerLin ``input_parameters`` groups, so every
row of a batch carries its own parameter vector (the per-sample Fisher needs this).
Parameter names are zero-padded so MerLin's name order equals the circuit order.
"""

from __future__ import annotations

import math

import merlin as ML
import perceval as pcvl
import torch

ENCODINGS = ("easy", "hard")
DETECTIONS = ("threshold", "unbunched")
# Amendment 2 design-study options (LOG.md). The first value of each is the planned design.
MESH_ORDERS = ("bs_ps", "ps_bs")
LAYOUTS = ("end", "interleaved")
INPUTS = ("fock", "cluster", "ghz")
READOUTS = ("click_parity", "pnr_parity")


def mzi_positions(n_modes: int, n_mzis: int) -> list[int]:
    """Return the upper mode of each MZI in rectangular brick order, wrapping over columns."""
    positions: list[int] = []
    column = 0
    while len(positions) < n_mzis:
        for upper in range(column % 2, n_modes - 1, 2):
            positions.append(upper)
            if len(positions) == n_mzis:
                break
        column += 1
    return positions


def _balanced_mixer(circuit: pcvl.Circuit, n_modes: int) -> None:
    """Two brick columns of fixed 50:50 beam splitters (Perceval's default ``BS()``)."""
    for start in (0, 1):
        for upper in range(start, n_modes - 1, 2):
            circuit.add(upper, pcvl.BS())


class PhotonicEncodingModel:
    """Photonic classifier ``p(y | x; theta)`` with an easy or hard data encoding.

    Parameters
    ----------
    n_inputs : int
        Input size ``s_in`` (at least 2).
    encoding : str
        ``"easy"`` or ``"hard"``.
    d : int
        Number of trainable phases; must be even (two phases per MZI).
    detection : str
        ``"threshold"`` (FOCK simulation, clicks, no postselection) or ``"unbunched"``
        (MerLin UNBUNCHED postselection).
    extra_modes : int
        Modes beyond ``s_in``. Default value is 2.
    mesh_order : str
        ``"bs_ps"``: each MZI is BS(theta) then PS(phi) (the planned design; a phase
        after the last beam splitter on its mode is dead). ``"ps_bs"``: PS(phi) then
        BS(theta), so every phase reaches a later beam splitter. Default value is
        ``"bs_ps"``.
    layout : str
        ``"end"``: every encoding repetition, then the whole mesh (planned design).
        ``"interleaved"``: W_1 E W_2 ... E W_(reps+1), data re-uploading between
        trainable blocks, with the MZIs split as evenly as possible. Default value is
        ``"end"``.
    input_kind : str
        ``"fock"``: ``[1, 0, 1, 0, ...]``. ``"cluster"`` / ``"ghz"``: photons from one
        spin emitter (``EmitterState.caterpillar``, photon k on modes (2k, 2k+1),
        emitter measured in the X basis, giving the pure entangled state). Emitter inputs
        need the photonic_engines fork. Default value is ``"fock"``.
    readout : str
        ``"click_parity"``: parity of threshold clicks on odd modes. ``"pnr_parity"``:
        parity of the photon count on odd modes (photon-number-resolving detectors).
        Default value is ``"click_parity"``.
    data_reps : int or None
        Number of encoding repetitions. None uses the paper's depth (2 for hard, 1 for
        easy). Default value is None.
    """

    output_size = 2

    def __init__(
        self,
        n_inputs: int,
        encoding: str,
        d: int,
        detection: str,
        extra_modes: int = 2,
        mesh_order: str = "bs_ps",
        layout: str = "end",
        input_kind: str = "fock",
        readout: str = "click_parity",
        data_reps: int | None = None,
    ):
        for value, allowed, label in (
            (mesh_order, MESH_ORDERS, "mesh_order"),
            (layout, LAYOUTS, "layout"),
            (input_kind, INPUTS, "input_kind"),
            (readout, READOUTS, "readout"),
        ):
            if value not in allowed:
                raise ValueError(
                    f"Unsupported {label} {value!r}; expected one of {allowed}"
                )
        if readout == "pnr_parity" and detection != "threshold":
            raise ValueError(
                "pnr_parity needs the FOCK simulation (detection='threshold')"
            )
        if encoding not in ENCODINGS:
            raise ValueError(
                f"Unsupported encoding {encoding!r}; expected one of {ENCODINGS}"
            )
        if detection not in DETECTIONS:
            raise ValueError(
                f"Unsupported detection {detection!r}; expected one of {DETECTIONS}"
            )
        if d % 2 or d <= 0:
            raise ValueError(
                f"d must be a positive even number (two phases per MZI); got {d}"
            )
        if n_inputs < 2:
            raise ValueError("The photonic model needs at least two inputs")
        self.input_size = n_inputs
        self.encoding = encoding
        self.detection = detection
        self.d = d
        self.n_modes = n_inputs + extra_modes
        self.n_photons = self.n_modes // 2
        self.input_state = [1 if mode % 2 == 0 else 0 for mode in range(self.n_modes)]
        # The paper's depths: hard 2, easy 1. data_reps overrides them (Amendment 4).
        self.repetitions = (
            data_reps if data_reps is not None else (2 if encoding == "hard" else 1)
        )
        if self.repetitions < 1:
            raise ValueError(f"data_reps must be >= 1; got {data_reps}")
        self.mesh_order = mesh_order
        self.layout = layout
        self.input_kind = input_kind
        self.readout = readout
        if input_kind != "fock" and self.n_modes != 2 * self.n_photons:
            raise ValueError(
                "Emitter inputs need an even number of modes (one photon per mode pair)"
            )
        self.circuit = self._build_circuit()
        self.input_groups = ["a", "b", "t"] if encoding == "hard" else ["a", "t"]
        self._layers: dict[tuple, ML.QuantumLayer] = {}

    def _add_mesh(self, circuit: pcvl.Circuit, first_index: int, n_mzis: int) -> int:
        """Append ``n_mzis`` MZIs in brick order; return the next parameter index."""
        index = first_index
        for upper in mzi_positions(self.n_modes, n_mzis):
            # Name the first-placed element first so circuit order equals name order.
            if self.mesh_order == "bs_ps":
                circuit.add(upper, pcvl.BS(pcvl.P(f"t{index:04d}")))
                circuit.add(upper, pcvl.PS(pcvl.P(f"t{index + 1:04d}")))
            else:
                circuit.add(upper, pcvl.PS(pcvl.P(f"t{index:04d}")))
                circuit.add(upper, pcvl.BS(pcvl.P(f"t{index + 1:04d}")))
            index += 2
        return index

    def _mesh_blocks(self) -> list[int]:
        """MZI counts per trainable block: one block (end) or reps + 1 blocks (interleaved)."""
        n_mzis = self.d // 2
        blocks = 1 if self.layout == "end" else self.repetitions + 1
        base, extra = divmod(n_mzis, blocks)
        return [base + (1 if block < extra else 0) for block in range(blocks)]

    def _build_circuit(self) -> pcvl.Circuit:
        circuit = pcvl.Circuit(self.n_modes)
        blocks = self._mesh_blocks()
        parameter_index = 0
        if self.layout == "interleaved":
            parameter_index = self._add_mesh(circuit, parameter_index, blocks[0])
        for repetition in range(self.repetitions):
            _balanced_mixer(circuit, self.n_modes)
            for feature in range(self.input_size):
                circuit.add(feature, pcvl.PS(pcvl.P(f"a{repetition}{feature:03d}")))
            if self.encoding == "hard":
                _balanced_mixer(circuit, self.n_modes)
                for feature in range(self.input_size - 1):
                    circuit.add(feature, pcvl.PS(pcvl.P(f"b{repetition}{feature:03d}")))
            if self.layout == "interleaved":
                parameter_index = self._add_mesh(
                    circuit, parameter_index, blocks[repetition + 1]
                )
        if self.layout == "end":
            parameter_index = self._add_mesh(circuit, parameter_index, blocks[0])
        assert parameter_index == self.d
        return circuit

    def _input_state(self):
        if self.input_kind == "fock":
            return pcvl.BasicState(self.input_state)
        # Fork-only API (photonic_engines); the planned FOCK designs never reach this line.
        # A spin emits photon k into modes (2k, 2k+1). Linear cluster: a pi/2 spin rotation
        # before every emission; GHZ: one pi/2 preparation, then no rotation. The emitter is
        # finally measured in the X basis: the "+" outcome leaves the pure entangled photon
        # state; "-" differs by a known phase on one photon, correctable by a fixed phase
        # shifter, so the state is deterministic in principle. Tracing the emitter out instead
        # would leave a classical mixture (checked: traced GHZ = 50/50 mix of two Fock states).
        rotation = ML.EmitterState.spin_gate(math.pi / 2, 0.0)
        identity = ML.EmitterState.spin_gate(0.0, 0.0)
        gates = (
            [rotation] * self.n_photons
            if self.input_kind == "cluster"
            else [rotation] + [identity] * (self.n_photons - 1)
        )
        x_plus = (1 / math.sqrt(2), 1 / math.sqrt(2))
        return ML.EmitterState.caterpillar(
            self.n_photons, gates, n_modes=self.n_modes, final=x_plus
        )

    def layer(
        self, device: torch.device, dtype: torch.dtype, space: str | None = None
    ) -> ML.QuantumLayer:
        """Return (and cache) the MerLin layer for a device, dtype and computation space."""
        space = space or ("FOCK" if self.detection == "threshold" else "UNBUNCHED")
        key = (str(device), dtype, space)
        if key not in self._layers:
            self._layers[key] = ML.QuantumLayer(
                circuit=self.circuit.copy(),
                input_parameters=self.input_groups,
                input_state=self._input_state(),
                measurement_strategy=ML.MeasurementStrategy.probs(
                    getattr(ML.ComputationSpace, space)
                ),
                device=device,
                dtype=dtype,
            )
        return self._layers[key]

    def _odd_click_mask(
        self, layer: ML.QuantumLayer, device: torch.device
    ) -> torch.Tensor:
        # click_parity counts detectors that fire; pnr_parity counts photons.
        counts_on_odd = [
            sum(
                (count if self.readout == "pnr_parity" else min(count, 1))
                for mode, count in enumerate(state)
                if mode % 2 == 1
            )
            for state in layer.output_keys
        ]
        return torch.tensor([count % 2 == 1 for count in counts_on_odd], device=device)

    def _inputs(self, x: torch.Tensor, theta: torch.Tensor) -> list[torch.Tensor]:
        first_order = (2.0 * x).repeat(1, self.repetitions)
        tensors = [first_order]
        if self.encoding == "hard":
            second_order = 2.0 * (math.pi - x[:, :-1]) * (math.pi - x[:, 1:])
            tensors.append(second_order.repeat(1, self.repetitions))
        tensors.append(theta)
        return tensors

    def probabilities(
        self, x: torch.Tensor, theta: torch.Tensor, space: str | None = None
    ) -> torch.Tensor:
        """Return ``[p(class 0), p(class 1)]`` per sample, shape ``(B, 2)``.

        Parameters
        ----------
        x : torch.Tensor
            Inputs ``(B, s_in)``.
        theta : torch.Tensor
            Trainable phases ``(B, d)``, one row per sample.
        space : str or None
            Override the computation space (``"FOCK"`` or ``"UNBUNCHED"``). Default value
            is None (the configured detection).
        """
        if x.shape[1] != self.input_size or theta.shape[1] != self.d:
            raise ValueError(
                f"Expected x (B, {self.input_size}) and theta (B, {self.d}); got {tuple(x.shape)}, {tuple(theta.shape)}"
            )
        layer = self.layer(x.device, x.dtype, space)
        distribution = layer(*self._inputs(x, theta))
        mask = self._odd_click_mask(layer, x.device)
        odd = distribution[:, mask].sum(dim=1)
        even = distribution[:, ~mask].sum(dim=1)
        return torch.stack((even, odd), dim=1)

    def collision_free_probability(
        self, x: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Probability mass on collision-free outcomes (the UNBUNCHED postselection success)."""
        layer = self.layer(x.device, x.dtype, "FOCK")
        distribution = layer(*self._inputs(x, theta))
        free = torch.tensor(
            [max(state) <= 1 for state in layer.output_keys], device=x.device
        )
        return distribution[:, free].sum(dim=1)

    def describe(self) -> dict:
        """Hardware-aware description of the configuration (Phase 4 reporting contract)."""
        return {
            "computation_space": "FOCK"
            if self.detection == "threshold"
            else "UNBUNCHED",
            "detector_model": (
                "ideal photon-number-resolving"
                if self.readout == "pnr_parity"
                else "ideal threshold (clicks; bunched events kept)"
            )
            if self.detection == "threshold"
            else "MerLin UNBUNCHED: postselected collision-free outcomes, renormalized",
            "photons": self.n_photons,
            "modes": self.n_modes,
            "input_state": self.input_state
            if self.input_kind == "fock"
            else f"{self.input_kind} state of {self.n_photons} photons from one spin emitter (EmitterState.caterpillar, photon k on modes (2k, 2k+1); emitter measured in X, '+' kept, '-' correctable by a fixed phase: pure entangled state)",
            "encoding": f"{self.encoding}: {self.repetitions} repetition(s); 2x_i phases on modes 0..{self.input_size - 1}"
            + (
                "; 2(pi-x_i)(pi-x_(i+1)) phases after a second mixer"
                if self.encoding == "hard"
                else ""
            ),
            "trainable": f"{self.d // 2} MZIs ({'BS theta then PS phi' if self.mesh_order == 'bs_ps' else 'PS phi then BS theta'}), rectangular brick order, layout={self.layout} (blocks {self._mesh_blocks()})",
            "measurement_strategy": "MeasurementStrategy.probs -> parity of "
            + ("photon counts" if self.readout == "pnr_parity" else "clicks")
            + " on odd modes",
            "postselection": "none"
            if self.detection == "threshold"
            else "collision-free outcomes only",
            "simulator": "MerLin analytic simulation (shots = 0)",
        }
