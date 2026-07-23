"""Model definitions for the QCNN-ID reproduction.

Three model families are provided:

* ``CNNClassifier`` — the classifier described in the paper text
  (Linear 128 -> Linear 64 -> one binary logit).
* ``QCNNClassifier`` — the provided quantum model in the notebook
* ``PhotonicClassifier`` — MerLin photonic adaptation of the QCNN-ID classifier.
  Reservoir-style classifier:
  Encodes the 8 PCA-scaled features as phase shifts on the 8 modes of a
  photonic circuit, propagates through fixed random interferometers, and
  projects the ``output_size`` probability vector onto one binary logit with a
  trainable Linear head.
  - computation space: UNBUNCHED (threshold detection),
  - modes / photons: 8 / 4,
  - encoding: angle encoding on all modes, scale = pi,
  - measurement: probability distribution over output configurations,
  - postselection: none.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """CNN classifier matching the paper text baseline (Sec. 3.3)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        dropout: float = 0.3,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(64, num_classes, bias=True)

        if device is None:
            device = "cpu"
        self.torch_device = torch.device(device)

        self.encoder.to(self.torch_device)
        self.head.to(self.torch_device)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return E(x), the learned classical feature vector."""
        return self.encoder(x)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def qiskit_like_zz_feature_map(
    qml,
    inputs,
    n_qubits: int,
    reps: int = 1,
    entanglement: str = "full",
    alpha: float = 2.0,
):
    """PennyLane implementation matching the public circuit convention
    of Qiskit's zz_feature_map.

    Equivalent target:
        qiskit.circuit.library.zz_feature_map(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
            alpha=alpha,
        )

    Default Qiskit data map:
        phi(x_i) = x_i
        phi(x_i, x_j) = (pi - x_i) * (pi - x_j)
    """

    if entanglement == "full":
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    elif entanglement == "linear":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]

    elif entanglement == "reverse_linear":
        pairs = [(i, i - 1) for i in range(n_qubits - 1, 0, -1)]

    else:
        raise ValueError(
            "Supported entanglement values are: 'full', 'linear', 'reverse_linear'."
        )

    for _ in range(reps):
        # Single-qubit Z feature terms.
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.PhaseShift(alpha * inputs[..., i], wires=i)

        # Two-qubit ZZ feature terms.
        for i, j in pairs:
            qml.CNOT(wires=[i, j])
            qml.PhaseShift(
                alpha * (math.pi - inputs[..., i]) * (math.pi - inputs[..., j]),
                wires=j,
            )
            qml.CNOT(wires=[i, j])


def qiskit_real_amplitudes_ansatz(
    qml,
    weights,
    n_qubits: int,
    reps: int = 1,
    entanglement: str = "reverse_linear",
    skip_final_rotation_layer: bool = False,
):
    """PennyLane implementation matching the public circuit structure
    of Qiskit's real_amplitudes ansatz."""

    if entanglement == "reverse_linear":
        pairs = [(i, i - 1) for i in range(n_qubits - 1, 0, -1)]

    elif entanglement == "linear":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]

    elif entanglement == "full":
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    else:
        raise ValueError(
            "Supported entanglement values are: 'reverse_linear', 'linear', 'full'."
        )

    expected_layers = reps if skip_final_rotation_layer else reps + 1

    if weights.shape[-2:] != (expected_layers, n_qubits):
        raise ValueError(
            f"Invalid weights shape. Expected (..., {expected_layers}, {n_qubits}), "
            f"got {tuple(weights.shape)}."
        )

    for qubit in range(n_qubits):
        qml.RY(weights[0, qubit], wires=qubit)

    for layer in range(reps):
        for control, target in pairs:
            qml.CNOT(wires=[control, target])

        if not skip_final_rotation_layer or layer < reps - 1:
            for qubit in range(n_qubits):
                qml.RY(weights[layer + 1, qubit], wires=qubit)


class QCNNClassifier(nn.Module):
    """Hybrid quantum-classical classifier with Pauli-Z measurements per qubit."""

    def __init__(
        self,
        n_qubits: int = 8,
        reps: int = 1,
        num_classes: int = 1,
        device: str | torch.device | None = None,
        feature_map_entanglement: str = "full",
        ansatz_entanglement: str = "reverse_linear",
    ):
        super().__init__()
        import pennylane as qml

        self.n_qubits = int(n_qubits)
        self.reps = int(reps)
        self.feature_map_entanglement = feature_map_entanglement
        self.ansatz_entanglement = ansatz_entanglement

        if device is None:
            device = "cpu"
        self.torch_device = torch.device(device)

        if self.torch_device.type == "cpu":
            self.q_device = "lightning.qubit"
        elif self.torch_device.type == "cuda":
            self.q_device = "lightning.gpu"
        else:
            raise ValueError(
                "QCNNClassifier supports only CPU and CUDA devices; "
                f"got {self.torch_device!s}."
            )

        # print("PennyLane version:", qml.__version__)
        # print("Torch device:", self.torch_device)
        # print("PennyLane quantum device:", self.q_device)

        self.dev = qml.device(
            self.q_device,
            wires=self.n_qubits,
        )

        # Qiskit real_amplitudes(..., reps=1) has reps + 1 RY layers
        # by default because skip_final_rotation_layer=False.
        weight_shapes = {
            "weights": (self.reps + 1, self.n_qubits),
        }

        @qml.qnode(
            self.dev,
            interface="torch",
            diff_method="adjoint" if self.q_device == "lightning.gpu" else "best",
        )
        def quantum_circuit(inputs, weights):
            qiskit_like_zz_feature_map(
                qml=qml,
                inputs=inputs,
                n_qubits=self.n_qubits,
                reps=self.reps,
                entanglement=self.feature_map_entanglement,
                alpha=2.0,
            )

            qiskit_real_amplitudes_ansatz(
                qml=qml,
                weights=weights,
                n_qubits=self.n_qubits,
                reps=self.reps,
                entanglement=self.ansatz_entanglement,
                skip_final_rotation_layer=False,
            )

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes,
        )

        self.head = nn.Linear(self.n_qubits, num_classes, bias=True)

        self.to(self.torch_device)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return E(x), the vector of Pauli-Z expectation values."""
        x = x.to(self.torch_device)
        return self.qlayer(x)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class PhotonicClassifier(nn.Module):
    """Photonic classifier"""

    def __init__(
        self,
        n_modes: int = 8,
        n_photons: int = 4,
        num_classes: int = 1,
        seed: int = 42,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        import merlin as ml
        import perceval as pcvl
        from perceval import random_seed

        if n_photons > n_modes:
            raise ValueError(
                "Error with photons_input_mode: Too many photons versus modes "
                f"(got {n_photons} vs {n_modes})."
            )

        self.n_modes = n_modes
        self.n_photons = n_photons
        self.seed = seed

        # Sandwich-style circuit a.k.a. reservoir
        circuit = pcvl.Circuit(n_modes)

        def _generate_generic_interferometer(n_modes, seed):
            random_seed(seed)  # Set Perceval's seed, for reproductibility
            # Generate a Haar random unitary
            unitary = pcvl.Matrix.random_unitary(n_modes)
            interferometer = pcvl.Unitary(unitary)
            return interferometer

        # Pre-circuit
        circuit.add(0, _generate_generic_interferometer(n_modes, seed))

        # Angle encoding: Phase Shifters column
        ps_inputs = pcvl.Circuit(n_modes, "inputs")
        params_prefix = "px"
        for i in range(n_modes):
            ps_inputs.add(i, pcvl.PS(pcvl.P(f"{params_prefix}{i}")))
        circuit.add(0, ps_inputs)

        # Reservoir
        circuit.add(0, _generate_generic_interferometer(n_modes, seed))

        # Input state:
        # Distribute photons across the modes
        step = (n_modes - 1) / (n_photons - 1) if n_photons > 1 else 0
        input_state = [0] * n_modes
        for i in range(n_photons):
            index = int(round(i * step))
            input_state[index] = 1

        if device is None:
            device = "cpu"
        self.torch_device = torch.device(device)

        self.qlayer = ml.QuantumLayer(
            input_size=n_modes,
            circuit=circuit,
            input_parameters=[params_prefix],
            input_state=input_state,
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
            computation_space=ml.ComputationSpace.UNBUNCHED,
            device=self.torch_device,
        )
        out_dim = int(self.qlayer.output_size)
        if out_dim == 0:
            raise RuntimeError(
                "MerLin QuantumLayer reports output_size=0; check "
                "n_modes/n_photons/computation_space combination."
            )
        self.head = nn.Linear(out_dim, num_classes, bias=True).to(self.torch_device)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return E(x), the Fock-probability feature vector."""
        return self.qlayer(x)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @property
    def output_size(self) -> int:
        return int(self.qlayer.output_size)
