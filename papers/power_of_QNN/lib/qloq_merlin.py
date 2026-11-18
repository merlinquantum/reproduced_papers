"""QLOQ [n] encoding of the paper's qubit QNN on one photon (Phase 04, Amendment 3, T2b).

Qubit logic on qudits (Lysaght et al., arXiv:2411.03878): all ``n`` qubits form one group,
encoded in a single photon over ``2**n`` modes. Qubit basis state ``b`` (bit ``q`` = qubit
``q``) is the photon in mode ``b``. Every gate of the paper's circuit becomes linear optics,
with no entangling gate and no postselection:

- H, RY(theta) on qubit q: the same 2x2 beam splitter (``BS.H``, ``BS.Ry``) on every mode
  pair ``(k, k + 2**q)`` with bit q of k equal to 0, made adjacent by a fixed ``PERM``.
- P(2 x_q): a phase shifter on every mode whose bit q is 1.
- ZZ term exp(i 2 (pi - x_i)(pi - x_j)) on bit_i XOR bit_j = 1 (Qiskit's CX-P-CX): a phase
  shifter on those modes.
- CX(c, t): the mode permutation k <-> k XOR 2**t for modes with bit c set.

This is a single-photon model. It reproduces the qubit statevector exactly, so it is a
compression demonstration and a labelled ablation, not a multi-photon photonic
implementation (photonic-translation policy).
"""

from __future__ import annotations

import itertools
import math

import merlin as ML
import perceval as pcvl
import torch

FEATURE_MAPS = ("zz", "z")


class QloqQubitQNN:
    """Paper QNN (ZZ or Z feature map + RealAmplitudes full + parity) on one photon.

    Parameters
    ----------
    n_qubits : int
        Number of logical qubits ``S`` (modes ``2**S``).
    feature_map : str
        ``"zz"`` (QNN, repetitions 2) or ``"z"`` (easy model, repetitions 1).
    feature_map_reps : int
        Feature-map repetitions.
    ansatz_reps : int
        RealAmplitudes depth ``D``; ``d = (D + 1) S``.
    """

    output_size = 2

    def __init__(
        self, n_qubits: int, feature_map: str, feature_map_reps: int, ansatz_reps: int
    ):
        if feature_map not in FEATURE_MAPS:
            raise ValueError(
                f"Unsupported feature map {feature_map!r}; expected one of {FEATURE_MAPS}"
            )
        self.n_qubits = n_qubits
        self.input_size = n_qubits
        self.n_modes = 2**n_qubits
        self.n_photons = 1
        self.feature_map = feature_map
        self.feature_map_reps = feature_map_reps
        self.ansatz_reps = ansatz_reps
        self.d = (ansatz_reps + 1) * n_qubits
        self.pairs = list(itertools.combinations(range(n_qubits), 2))
        # name -> ("x", feature, None) | ("zz", i, j) | ("theta", index, None)
        self._sources: dict[str, tuple] = {}
        self._counter = itertools.count()
        self.circuit = self._build()
        self.input_groups = ["a", "t"] if feature_map == "z" else ["a", "b", "t"]
        self._layers: dict[tuple, ML.QuantumLayer] = {}

    def _name(self, group: str, source: tuple) -> str:
        name = f"{group}{next(self._counter):06d}"
        self._sources[name] = source
        return name

    def _pair_permutation(self, qubit: int) -> list[int]:
        """PERM list sending pair (k, k + 2**qubit) of index j to modes (2j, 2j + 1)."""
        permutation = [0] * self.n_modes
        lows = [k for k in range(self.n_modes) if not (k >> qubit) & 1]
        for j, low in enumerate(lows):
            permutation[low] = 2 * j
            permutation[low + 2**qubit] = 2 * j + 1
        return permutation

    def _single_qubit(self, circuit: pcvl.Circuit, qubit: int, make_component) -> None:
        permutation = self._pair_permutation(qubit)
        inverse = [0] * self.n_modes
        for source, target in enumerate(permutation):
            inverse[target] = source
        circuit.add(0, pcvl.PERM(permutation))
        for j in range(self.n_modes // 2):
            circuit.add(2 * j, make_component())
        circuit.add(0, pcvl.PERM(inverse))

    def _phase_on(
        self, circuit: pcvl.Circuit, modes: list[int], group: str, source: tuple
    ) -> None:
        for mode in modes:
            circuit.add(mode, pcvl.PS(pcvl.P(self._name(group, source))))

    def _cx(self, circuit: pcvl.Circuit, control: int, target: int) -> None:
        permutation = list(range(self.n_modes))
        for k in range(self.n_modes):
            if (k >> control) & 1:
                permutation[k] = k ^ (1 << target)
        circuit.add(0, pcvl.PERM(permutation))

    def _build(self) -> pcvl.Circuit:
        circuit = pcvl.Circuit(self.n_modes)
        for _ in range(self.feature_map_reps):
            for qubit in range(self.n_qubits):
                self._single_qubit(circuit, qubit, pcvl.BS.H)
            for qubit in range(self.n_qubits):
                self._phase_on(
                    circuit,
                    [k for k in range(self.n_modes) if (k >> qubit) & 1],
                    "a",
                    ("x", qubit, None),
                )
            if self.feature_map == "zz":
                for i, j in self.pairs:
                    modes = [
                        k for k in range(self.n_modes) if ((k >> i) ^ (k >> j)) & 1
                    ]
                    self._phase_on(circuit, modes, "b", ("zz", i, j))
        theta_index = 0
        for layer in range(self.ansatz_reps + 1):
            if layer > 0:
                for control, target in self.pairs:
                    self._cx(circuit, control, target)
            for qubit in range(self.n_qubits):
                index = theta_index
                self._single_qubit(
                    circuit,
                    qubit,
                    lambda index=index: pcvl.BS.Ry(
                        pcvl.P(self._name("t", ("theta", index, None)))
                    ),
                )
                theta_index += 1
        return circuit

    def layer(self, device: torch.device, dtype: torch.dtype) -> ML.QuantumLayer:
        """Return (and cache) the MerLin layer for a device and dtype (FOCK, 1 photon)."""
        key = (str(device), dtype)
        if key not in self._layers:
            input_state = [1] + [0] * (self.n_modes - 1)
            self._layers[key] = ML.QuantumLayer(
                circuit=self.circuit.copy(),
                input_parameters=self.input_groups,
                input_state=pcvl.BasicState(input_state),
                measurement_strategy=ML.MeasurementStrategy.probs(
                    ML.ComputationSpace.FOCK
                ),
                device=device,
                dtype=dtype,
            )
        return self._layers[key]

    def _inputs(
        self, layer: ML.QuantumLayer, x: torch.Tensor, theta: torch.Tensor
    ) -> list[torch.Tensor]:
        """Assemble one tensor per input group, columns in MerLin's name order."""
        tensors = []
        for group in self.input_groups:
            columns = []
            for name in layer.computation_process.converter.spec_mappings[group]:
                kind, first, second = self._sources[name]
                if kind == "x":
                    columns.append(2.0 * x[:, first])
                elif kind == "zz":
                    columns.append(
                        2.0 * (math.pi - x[:, first]) * (math.pi - x[:, second])
                    )
                else:
                    columns.append(theta[:, first])
            tensors.append(torch.stack(columns, dim=1))
        return tensors

    def probabilities(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Return ``[p(even parity), p(odd parity)]`` of the logical bit string, ``(B, 2)``."""
        if x.shape[1] != self.n_qubits or theta.shape[1] != self.d:
            raise ValueError(
                f"Expected x (B, {self.n_qubits}) and theta (B, {self.d}); got {tuple(x.shape)}, {tuple(theta.shape)}"
            )
        layer = self.layer(x.device, x.dtype)
        distribution = layer(*self._inputs(layer, x, theta))
        odd = torch.tensor(
            [bin(state.index(1)).count("1") % 2 == 1 for state in layer.output_keys],
            device=x.device,
        )
        return torch.stack(
            (distribution[:, ~odd].sum(dim=1), distribution[:, odd].sum(dim=1)), dim=1
        )

    def describe(self) -> dict:
        """Hardware-aware description (Phase 4 reporting contract)."""
        return {
            "computation_space": "FOCK",
            "detector_model": "ideal threshold (one photon: click = photon)",
            "photons": 1,
            "modes": self.n_modes,
            "input_state": "photon in mode 0 (logical |0...0>)",
            "encoding": f"QLOQ [{self.n_qubits}] (arXiv:2411.03878): {'ZZFeatureMap' if self.feature_map == 'zz' else 'ZFeatureMap'} reps {self.feature_map_reps} as beam splitters and phases on mode pairs",
            "trainable": f"RealAmplitudes reps {self.ansatz_reps} (full): BS.Ry on mode pairs, CX as fixed mode permutations; d = {self.d}",
            "measurement_strategy": "MeasurementStrategy.probs -> parity of the logical bit string (mode index)",
            "postselection": "none",
            "simulator": "MerLin analytic simulation (shots = 0); single photon: classically simulable, labelled ablation",
        }
