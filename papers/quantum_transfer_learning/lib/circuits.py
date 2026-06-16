"""
Variational Quantum Circuits for Transfer Learning
===================================================

Implements quantum circuits from Mari et al. (2020) using two backends:

1. MerLin (photonic) - Primary implementation using QuantumLayer
2. PennyLane (qubit) - Reference implementation for comparison

MerLin uses linear optical circuits with:
- Beam splitter meshes for variational/entangling layers
- Phase shifters for angle encoding
- Fock state measurements

PennyLane uses qubit circuits with:
- RY rotations and CNOT gates
- Pauli-Z measurements
"""

import random
from typing import List, Optional

# MerLin imports
import numpy as np

# PennyLane imports (for comparison)
import pennylane as qml
import perceval as pcvl
import torch
import torch.nn as nn
from merlin import ComputationSpace, QuantumLayer
from merlin.measurement import MeasurementStrategy

# =============================================================================
# MerLin Photonic Implementation (Primary)
# =============================================================================

def create_merlin_vqc_circuit(n_modes: int, n_features: int) -> pcvl.Circuit:
    """Create a variational quantum circuit using Perceval.

    Architecture follows the VQC pattern:
    [Trainable BS Mesh] → [Angle Encoding] → [Trainable BS Mesh]

    Args:
        n_modes: Number of optical modes
        n_features: Number of input features to encode

    Returns:
        Perceval Circuit with trainable and input parameters
    """
    # Left beam splitter mesh (trainable)
    bs_left = pcvl.GenericInterferometer(
        n_modes,
        lambda idx: pcvl.BS(theta=pcvl.P(f"theta_l{idx}"))
                    // (0, pcvl.PS(phi=np.pi * 2 * random.random())),
        shape=pcvl.InterferometerShape.RECTANGLE,
        depth=2 * n_modes,
        phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi * 2 * random.random()),
    )

    # Central encoding layer - phase shifters for angle encoding
    encoding = pcvl.Circuit(n_modes)
    start_mode = max(0, (n_modes - n_features) // 2)
    for i in range(n_features):
        mode = start_mode + i
        if mode < n_modes:
            encoding.add(mode, pcvl.PS(pcvl.P(f"x{i}")))

    # Right beam splitter mesh (trainable)
    bs_right = pcvl.GenericInterferometer(
        n_modes,
        lambda idx: pcvl.BS(theta=pcvl.P(f"theta_r{idx}"))
                    // (0, pcvl.PS(phi=np.pi * 2 * random.random())),
        shape=pcvl.InterferometerShape.RECTANGLE,
        depth=2 * n_modes,
        phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi * 2 * random.random()),
    )

    # Combine into full circuit
    circuit = pcvl.Circuit(n_modes)
    circuit.add(0, bs_left, merge=True)
    circuit.add(0, encoding, merge=True)
    circuit.add(0, bs_right, merge=True)

    return circuit


def create_merlin_deep_circuit(n_modes: int, n_features: int, depth: int) -> pcvl.Circuit:
    """Create a deeper variational circuit with multiple encoding layers.

    Architecture: [BS Mesh] → [Encode] → [BS Mesh] → [Encode] → ... → [BS Mesh]

    Args:
        n_modes: Number of optical modes
        n_features: Number of input features
        depth: Number of variational blocks

    Returns:
        Perceval Circuit
    """
    circuit = pcvl.Circuit(n_modes)

    for layer in range(depth):
        # Trainable beam splitter mesh
        bs_mesh = pcvl.GenericInterferometer(
            n_modes,
            lambda idx, l=layer: pcvl.BS(theta=pcvl.P(f"theta_{l}_{idx}"))
                                 // (0, pcvl.PS(phi=np.pi * 2 * random.random())),
            shape=pcvl.InterferometerShape.RECTANGLE,
            depth=n_modes,
            phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi * 2 * random.random()),
        )
        circuit.add(0, bs_mesh, merge=True)

        # Angle encoding (except for last layer)
        if layer < depth - 1:
            start_mode = max(0, (n_modes - n_features) // 2)
            for i in range(min(n_features, n_modes)):
                mode = start_mode + i
                if mode < n_modes:
                    circuit.add(mode, pcvl.PS(pcvl.P(f"x{layer}_{i}")))

    return circuit


class MerLinQuantumLayer(nn.Module):
    """MerLin-based variational quantum layer.

    Wraps MerLin's QuantumLayer for use in hybrid classical-quantum networks.
    Uses photonic circuits with beam splitter meshes and phase shifter encoding.
    """

    def __init__(
            self,
            n_modes: int = 4,
            n_features: int = 2,
            n_photons: int = 2,
            q_depth: int = 1,
            computation_space: str = "unbunched",
            measurement_strategy: str = "probabilities"
    ):
        """Initialize MerLin quantum layer.

        Args:
            n_modes: Number of optical modes
            n_features: Number of input features to encode
            n_photons: Number of photons in the input state
            q_depth: Circuit depth (number of encoding blocks)
            computation_space: 'fock', 'unbunched', or 'dual_rail'
            measurement_strategy: 'probabilities' or 'mode_expectations'
        """
        super().__init__()

        self.n_modes = n_modes
        self.n_features = n_features
        self.n_photons = n_photons
        self.q_depth = q_depth

        # Parse computation space
        self.computation_space = ComputationSpace.coerce(computation_space)

        # Create Perceval circuit
        if q_depth == 1:
            circuit = create_merlin_vqc_circuit(n_modes, n_features)
        else:
            circuit = create_merlin_deep_circuit(n_modes, n_features, q_depth)

        # Determine trainable vs input parameters
        all_params = [p.name for p in circuit.get_parameters()]
        input_params = [p for p in all_params if p.startswith("x")]
        trainable_params = [p for p in all_params if not p.startswith("x")]

        # Create initial state (dual-rail style)
        input_state = self._create_input_state(n_modes, n_photons)

        # Set measurement strategy
        if measurement_strategy == "probabilities":
            ms = MeasurementStrategy.PROBABILITIES
        elif measurement_strategy == "mode_expectations":
            ms = MeasurementStrategy.MODE_EXPECTATIONS
        else:
            ms = MeasurementStrategy.PROBABILITIES

        # Create MerLin QuantumLayer
        self.quantum_layer = QuantumLayer(
            input_size=n_features,
            circuit=circuit,
            trainable_parameters=trainable_params,
            input_parameters=["x"],  # Prefix for input parameters
            input_state=input_state,
            computation_space=self.computation_space,
            measurement_strategy=ms,
        )

        self._output_size = self.quantum_layer.output_size

    def _create_input_state(self, n_modes: int, n_photons: int) -> List[int]:
        """Create initial Fock state (dual-rail style: [1,0,1,0,...])."""
        state = [0] * n_modes
        for i in range(min(n_photons, n_modes)):
            state[i * 2 % n_modes] = 1 if i * 2 < n_modes else 0

        # Ensure we have the right number of photons
        current = sum(state)
        idx = 0
        while current < n_photons and idx < n_modes:
            if state[idx] == 0:
                state[idx] = 1
                current += 1
            idx += 1

        return state

    @property
    def output_size(self) -> int:
        """Return output dimension of the layer."""
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum layer.

        Args:
            x: Input tensor of shape (batch_size, n_features)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Save input device - MerLin may return CPU tensors
        input_device = x.device

        # MerLin quantum layer execution
        out = self.quantum_layer(x)

        # Ensure output is on same device as input
        return out.to(input_device)


class MerLinDressedCircuit(nn.Module):
    """Dressed quantum circuit using MerLin's photonic backend.

    Implements the dressed circuit architecture from the paper:
    Q̃ = L_{out} ∘ Q ∘ L_{in}

    Where Q is a MerLin QuantumLayer.
    """

    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            n_modes: int = 4,
            n_photons: int = 2,
            q_depth: int = 1,
            computation_space: str = "unbunched",
            scale_type: str = "learned"
    ):
        """Initialize dressed MerLin circuit.

        Args:
            n_inputs: Number of classical input features
            n_outputs: Number of output classes
            n_modes: Number of optical modes
            n_photons: Number of photons
            q_depth: Circuit depth
            computation_space: 'fock', 'unbunched', or 'dual_rail'
            scale_type: Input scaling ('learned', 'pi', '2pi', '1')
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_modes = n_modes

        # Pre-processing: scale and project to n_modes features
        self.scale_layer = ScaleLayer(n_inputs, scale_type=scale_type)
        self.pre_layer = nn.Sequential(
            nn.Linear(n_inputs, n_modes),
            nn.Tanh()
        )

        # Quantum layer
        self.quantum_layer = MerLinQuantumLayer(
            n_modes=n_modes,
            n_features=n_modes,  # After pre-processing
            n_photons=n_photons,
            q_depth=q_depth,
            computation_space=computation_space
        )

        # Post-processing
        self.post_layer = nn.Linear(self.quantum_layer.output_size, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: classical → quantum → classical."""
        # Save input device for ensuring consistent device placement
        input_device = x.device

        x = self.scale_layer(x)
        x = self.pre_layer(x)

        # Quantum layer may return CPU tensor - MerLinQuantumLayer.forward handles this
        x = self.quantum_layer(x)

        # Ensure we're on the right device before post-processing
        #x = x.to(input_device)
        x = x.float().to(input_device)
        x = self.post_layer(x)


        return x


class ScaleLayer(nn.Module):
    """Learnable or fixed scaling layer for inputs.

    From the MerLin VQC example - scales input before encoding.
    """

    def __init__(self, dim: int, scale_type: str = "learned"):
        """Initialize scale layer.

        Args:
            dim: Input dimension
            scale_type: 'learned', 'pi', '2pi', '/pi', or '1'
        """
        super().__init__()

        if scale_type == "learned":
            self.scale = nn.Parameter(torch.rand(dim))
        elif scale_type == "2pi":
            self.register_buffer("scale", torch.full((dim,), 2 * np.pi))
        elif scale_type == "pi":
            self.register_buffer("scale", torch.full((dim,), np.pi))
        elif scale_type == "/pi":
            self.register_buffer("scale", torch.full((dim,), 1 / np.pi))
        else:  # "1"
            self.register_buffer("scale", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input element-wise."""
        return x * self.scale


class MerLinSimpleLayer(nn.Module):
    """Simple MerLin quantum layer using QuantumLayer.simple().

    Convenience wrapper around MerLin's built-in simple layer constructor.
    """

    def __init__(
            self,
            input_size: int,
            n_params: int = 90,
            output_size: Optional[int] = None,
            computation_space: str = "unbunched"
    ):
        """Initialize simple MerLin layer.

        Args:
            input_size: Number of input features
            n_params: Number of trainable parameters
            output_size: Output dimension (None for default)
            computation_space: 'fock', 'unbunched', or 'dual_rail'
        """
        super().__init__()

        self.quantum_layer = QuantumLayer.simple(
            input_size=input_size,
            n_params=n_params,
            output_size=output_size,
        )
        self._output_size = self.quantum_layer.output_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input device - MerLin may return CPU tensors
        input_device = x.device

        out = self.quantum_layer(x)

        # Ensure output is on same device as input
        return out.to(input_device)


# =============================================================================
# PennyLane Qubit Implementation (Reference/Comparison)
# =============================================================================

def create_pennylane_device(n_qubits: int, shots: Optional[int] = None):
    """Create a PennyLane quantum device."""
    return qml.device("default.qubit", wires=n_qubits, shots=shots)


def pennylane_embedding(x: torch.Tensor, wires: List[int]):
    """Angle embedding for PennyLane (H + RY encoding)."""
    for i, wire in enumerate(wires):
        qml.Hadamard(wires=wire)
        qml.RY(x[i] * np.pi / 2, wires=wire)


def pennylane_variational_layer(weights: torch.Tensor, wires: List[int]):
    """Variational layer: RY rotations + CNOT ladder."""
    n_qubits = len(wires)

    for i, wire in enumerate(wires):
        qml.RY(weights[i], wires=wire)

    for i in range(n_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


class PennyLaneQuantumLayer(nn.Module):
    """PennyLane variational quantum circuit for comparison.

    Implements the qubit-based circuit from the original paper.
    """

    def __init__(
            self,
            n_qubits: int = 4,
            q_depth: int = 5,
            shots: Optional[int] = None
    ):
        """Initialize PennyLane circuit.

        Args:
            n_qubits: Number of qubits
            q_depth: Number of variational layers
            shots: Measurement shots (None for analytic)
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.wires = list(range(n_qubits))

        # Trainable weights
        self.weights = nn.Parameter(torch.randn(q_depth, n_qubits) * 0.01)

        # Create device and QNode
        self.dev = create_pennylane_device(n_qubits, shots)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor):
        """Define quantum circuit."""
        pennylane_embedding(inputs, self.wires)

        for layer in range(self.q_depth):
            pennylane_variational_layer(weights[layer], self.wires)

        return [qml.expval(qml.PauliZ(w)) for w in self.wires]

    @property
    def output_size(self) -> int:
        return self.n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Save input device
        input_device = x.device


        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            result = self.qnode(x[i], self.weights)
            outputs.append(torch.stack(result))

        out = torch.stack(outputs)

        # Ensure output is on same device as input
        return out.to(input_device)


class PennyLaneDressedCircuit(nn.Module):
    """Dressed quantum circuit using PennyLane (qubit-based)."""

    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            n_qubits: int = 4,
            q_depth: int = 5
    ):
        """Initialize PennyLane dressed circuit."""
        super().__init__()

        self.pre_layer = nn.Sequential(
            nn.Linear(n_inputs, n_qubits),
            nn.Tanh()
        )

        self.quantum_layer = PennyLaneQuantumLayer(
            n_qubits=n_qubits,
            q_depth=q_depth
        )

        self.post_layer = nn.Linear(n_qubits, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input device
        input_device = x.device

        x = self.pre_layer(x)
        x = self.quantum_layer(x)

        # Ensure we're on the right device before post-processing
        # x = x.to(input_device)
        x = x.float().to(input_device)
        x = self.post_layer(x)

        return x


# =============================================================================
# Unified Interface
# =============================================================================

class DressedQuantumCircuit(nn.Module):
    """Unified dressed quantum circuit supporting both backends.

    Args:
        backend: 'merlin' for photonic or 'pennylane' for qubit-based
    """

    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            n_qubits: int = 4,
            q_depth: int = 5,
            backend: str = "merlin",
            **kwargs
    ):
        """Initialize dressed circuit with specified backend."""
        super().__init__()

        self.backend = backend

        if backend == "merlin":
            self.circuit = MerLinDressedCircuit(
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                n_modes=n_qubits,  # Map qubits to modes
                n_photons=kwargs.get("n_photons", 2),
                q_depth=kwargs.get("merlin_depth", 1),
                computation_space=kwargs.get("computation_space", "unbunched"),
                scale_type=kwargs.get("scale_type", "learned")
            )
        else:  # pennylane
            self.circuit = PennyLaneDressedCircuit(
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                n_qubits=n_qubits,
                q_depth=q_depth
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)


class VariationalCircuit(nn.Module):
    """Unified variational circuit supporting both backends."""

    def __init__(
            self,
            n_qubits: int = 4,
            q_depth: int = 5,
            backend: str = "merlin",
            **kwargs
    ):
        """Initialize variational circuit."""
        super().__init__()

        self.backend = backend
        self.n_qubits = n_qubits
        self.q_depth = q_depth

        if backend == "merlin":
            self.circuit = MerLinQuantumLayer(
                n_modes=n_qubits,
                n_features=n_qubits,
                n_photons=kwargs.get("n_photons", 2),
                q_depth=kwargs.get("merlin_depth", 1),
                computation_space=kwargs.get("computation_space", "unbunched")
            )
        else:
            self.circuit = PennyLaneQuantumLayer(
                n_qubits=n_qubits,
                q_depth=q_depth
            )

        # For compatibility
        if hasattr(self.circuit, 'weights'):
            self.weights = self.circuit.weights

    @property
    def output_size(self) -> int:
        return self.circuit.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)
