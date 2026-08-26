"""Qiskit statevector random kitchen sinks for RF-RQKS."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class QiskitQRKS(nn.Module):
    """Qiskit gate-model QRKS sampler.

    Parameters
    ----------
    qubit_count : int
        Number of qubits in the circuit.
    depth : int
        Number of encoding layers.
    episode_count : int
        Number of randomized episodes.
    L_strategy : str
        Encoding strategy, either ``L1`` or ``L2``. Both strategies use the
        Section III single-qubit ``Rx``-then-``Ry`` upload in the gate model.
    V_strategy : str | None
        Entangling strategy. ``None`` omits entangling gates.
    data_size : int
        Flattened input feature count.
    """

    def __init__(
        self,
        qubit_count: int,
        depth: int,
        episode_count: int,
        L_strategy: str,
        V_strategy: str | None,
        data_size: int,
        same_random_layer: bool = False,
    ) -> None:
        super().__init__()
        if qubit_count < 1:
            raise ValueError("qubit_count must be positive")
        if depth < 1 or episode_count < 1:
            raise ValueError("depth and episode_count must be positive")
        if L_strategy not in {"L1", "L2"}:
            raise ValueError("encoding strategy must be L1 or L2")
        if V_strategy not in {None, "V1", "V2", "V3"}:
            raise ValueError("entangling strategy must be V1, V2, V3, or null")
        if L_strategy == "L1" and qubit_count % 2:
            raise ValueError("L1 encoding requires an even qubit_count")

        self.qubit_count = qubit_count
        self.depth = depth
        self.episode_count = episode_count
        self.L_strategy = L_strategy
        self.V_strategy = V_strategy
        self.output_feature_count = episode_count * 2**qubit_count
        # Section III assigns two encoded angles to every qubit in every
        # upload layer: Rx(phi_x) followed by Ry(phi_y).
        input_size = depth * qubit_count * 2
        self.register_buffer(
            "weights", torch.randn(episode_count, input_size, data_size) * 2.0
        )
        self.register_buffer(
            "biases", torch.rand(episode_count, input_size) * 2.0 * torch.pi
        )

    def _build_circuit(self, angles: np.ndarray, episode: int):
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(self.qubit_count)
        phase_index = 0
        for depth_index in range(self.depth):
            for qubit in range(self.qubit_count):
                circuit.rx(float(angles[phase_index]), qubit)
                phase_index += 1
                circuit.ry(float(angles[phase_index]), qubit)
                phase_index += 1

            # Entanglers occur only between successive upload layers. The
            # final upload is therefore never followed by a CZ ring.
            if (
                self.V_strategy is not None
                and depth_index < self.depth - 1
                and self.qubit_count > 1
            ):
                ring_edge_count = 1 if self.qubit_count == 2 else self.qubit_count
                for qubit in range(ring_edge_count):
                    circuit.cz(qubit, (qubit + 1) % self.qubit_count)
        return circuit

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return computational-basis probabilities for every episode.

        Parameters
        ----------
        features : torch.Tensor
            Standardized DCT feature matrix.

        Returns
        -------
        torch.Tensor
            Qiskit statevector probabilities with ``E * 2**q`` columns.
        """
        from qiskit.quantum_info import Statevector

        if features.ndim == 1:
            features = features.unsqueeze(0)
        phases = torch.einsum("bd,eid->bei", features, self.weights) + self.biases
        outputs = []
        for sample_index in range(features.shape[0]):
            sample_outputs = []
            for episode in range(self.episode_count):
                circuit = self._build_circuit(
                    phases[sample_index, episode].detach().cpu().numpy(), episode
                )
                probabilities = Statevector.from_instruction(circuit).probabilities()
                sample_outputs.append(
                    torch.from_numpy(np.asarray(probabilities, dtype=np.float32))
                )
            outputs.append(torch.cat(sample_outputs))
        return torch.stack(outputs).to(features.device)
