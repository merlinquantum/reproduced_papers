"""
Variational Quantum Circuits for Adversarial Learning
======================================================

Implements the quantum circuit classifier from Lu et al. (2020) using MerLin.

The paper uses a hardware-efficient variational quantum circuit with:
- Rotation layers (single-qubit Euler rotations)
- Entangling layers (CNOT ladders)

For MerLin (photonic), we implement an analogous architecture:
- Beam splitter meshes for variational/entangling layers
- Phase shifters for angle encoding
- Fock state measurements for classification
"""

import logging
import random

import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from merlin import ComputationSpace, QuantumLayer
from merlin.measurement import MeasurementStrategy

logger = logging.getLogger(__name__)

# Try to import PennyLane for gate-based circuits
try:
    import pennylane as qml

    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


# =============================================================================
# PennyLane Gate-Based Circuits (Original Paper Architecture)
# =============================================================================


class PennyLaneQuantumClassifier(nn.Module):
    """Gate-based variational quantum classifier using PennyLane.

    Implements the exact architecture from Lu et al. (2020):
    - Rotation layers: Single-qubit Euler rotations (Z-X-Z)
    - Entangling layers: CNOT ladder

    This matches the paper's circuit structure shown in Figure 2.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_inputs: int = 8,
        n_outputs: int = 2,
        scale_type: str = "learned",
    ):
        """Initialize PennyLane quantum classifier.

        Args:
            n_qubits: Number of qubits (n in the paper)
            n_layers: Number of variational layers (p in the paper)
            n_inputs: Number of input features
            n_outputs: Number of output classes
            scale_type: Input scaling type
        """
        super().__init__()

        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required for gate-based circuits")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Input scaling
        self.scale_layer = ScaleLayer(n_inputs, scale_type=scale_type)

        # Input projection if dimensions don't match
        if n_inputs != n_qubits:
            self.input_projection = nn.Sequential(
                nn.Linear(n_inputs, n_qubits), nn.Tanh()
            )
        else:
            self.input_projection = nn.Tanh()

        # Number of parameters per layer: 3 per qubit (Euler angles)
        n_params_per_layer = 3 * n_qubits
        total_params = n_params_per_layer * n_layers

        # Trainable parameters
        self.theta = nn.Parameter(torch.randn(total_params) * 0.1)

        # Create quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Build quantum circuit
        self._build_circuit()

        # Output layer
        # Output qubits for classification (paper uses m qubits where 2^(m-1) < K <= 2^m)
        n_output_qubits = max(1, int(np.ceil(np.log2(n_outputs))))
        self.output_layer = nn.Linear(n_output_qubits, n_outputs)

    def _build_circuit(self):
        """Build the variational quantum circuit."""
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            """Variational quantum circuit.

            Architecture from Figure 2 of the paper:
            [Amplitude Encoding] → [Rot + Ent layers] × p → [Measurement]
            """
            # Amplitude encoding of input
            # Normalize inputs for valid quantum state
            norm = torch.sqrt(torch.sum(inputs**2) + 1e-8)
            normalized = inputs / norm

            # Pad to 2^n if needed
            n_amplitudes = 2**n_qubits
            if len(normalized) < n_amplitudes:
                padded = torch.zeros(
                    n_amplitudes, device=inputs.device, dtype=inputs.dtype
                )
                padded[: len(normalized)] = normalized
                normalized = padded / torch.sqrt(torch.sum(padded**2) + 1e-8)

            # Use AmplitudeEmbedding
            qml.AmplitudeEmbedding(
                normalized[:n_amplitudes], wires=range(n_qubits), normalize=True
            )

            # Variational layers
            param_idx = 0
            for _ in range(n_layers):
                # Rotation layer: Euler rotations (Z-X-Z) on each qubit
                for qubit in range(n_qubits):
                    qml.RZ(weights[param_idx], wires=qubit)
                    qml.RX(weights[param_idx + 1], wires=qubit)
                    qml.RZ(weights[param_idx + 2], wires=qubit)
                    param_idx += 3

                # Entangling layer: CNOT ladder
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

            # Measure output qubits (last few qubits)
            n_output_qubits = max(1, int(np.ceil(np.log2(self.n_outputs))))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_output_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum classifier.

        Args:
            x: Input tensor of shape (batch_size, n_inputs)

        Returns:
            Class logits of shape (batch_size, n_outputs)
        """
        batch_size = x.shape[0]

        # Scale and project input
        x = self.scale_layer(x)
        x = self.input_projection(x)

        # Process each sample through quantum circuit
        outputs = []
        for i in range(batch_size):
            out = self.circuit(x[i], self.theta)
            if isinstance(out, list):
                out = torch.stack(out)
            outputs.append(out)

        quantum_out = torch.stack(outputs).float()  # Cast to float32 for linear layer

        # Classification
        logits = self.output_layer(quantum_out)

        return logits


class PennyLaneHybridClassifier(nn.Module):
    """Hybrid classical-quantum classifier with PennyLane backend.

    Uses classical neural network for dimensionality reduction,
    then PennyLane quantum circuit for classification.
    """

    def __init__(
        self,
        input_dim: int,
        n_outputs: int,
        hidden_dims: list[int] | None = None,
        n_qubits: int = 8,
        n_layers: int = 2,
    ):
        """Initialize hybrid classifier.

        Args:
            input_dim: Input dimension
            n_outputs: Number of output classes
            hidden_dims: Hidden layer dimensions for classical encoder
            n_qubits: Number of qubits
            n_layers: Quantum circuit depth
        """
        super().__init__()

        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required for gate-based circuits")

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.n_outputs = n_outputs

        # Classical encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.Tanh(),
                ]
            )
            prev_dim = hidden_dim

        # Project to match quantum input
        encoder_layers.extend([nn.Linear(prev_dim, n_qubits), nn.Tanh()])

        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum classifier
        self.quantum_classifier = PennyLaneQuantumClassifier(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_inputs=n_qubits,
            n_outputs=n_outputs,
            scale_type="pi",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        features = self.encoder(x)
        return self.quantum_classifier(features)


class PennyLaneSimpleClassifier(nn.Module):
    """Simpler PennyLane classifier using angle encoding.

    Uses RY rotations for encoding (qubit encoding from the paper)
    instead of amplitude encoding, which is more efficient for
    small input dimensions.
    """

    def __init__(
        self, n_inputs: int, n_outputs: int, n_layers: int = 2, scale_type: str = "pi"
    ):
        """Initialize simple PennyLane classifier.

        Args:
            n_inputs: Number of input features (= number of qubits)
            n_outputs: Number of output classes
            n_layers: Number of variational layers
            scale_type: Input scaling type
        """
        super().__init__()

        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required for gate-based circuits")

        self.n_qubits = n_inputs
        self.n_layers = n_layers
        self.n_outputs = n_outputs

        # Input scaling
        self.scale_layer = ScaleLayer(n_inputs, scale_type=scale_type)

        # Trainable parameters: 3 angles per qubit per layer
        n_params = 3 * self.n_qubits * n_layers
        self.theta = nn.Parameter(torch.randn(n_params) * 0.1)

        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Build circuit
        self._build_circuit()

        # Output layer
        self.output_layer = nn.Linear(self.n_qubits, n_outputs)

    def _build_circuit(self):
        """Build quantum circuit with angle encoding."""
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Angle encoding: RY rotations
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            param_idx = 0
            for _ in range(n_layers):
                # Rotation layer
                for qubit in range(n_qubits):
                    qml.RZ(weights[param_idx], wires=qubit)
                    qml.RX(weights[param_idx + 1], wires=qubit)
                    qml.RZ(weights[param_idx + 2], wires=qubit)
                    param_idx += 3

                # Entangling layer
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]

        x = self.scale_layer(x)

        outputs = []
        for i in range(batch_size):
            out = self.circuit(x[i], self.theta)
            if isinstance(out, list):
                out = torch.stack(out)
            outputs.append(out)

        quantum_out = torch.stack(outputs).float()  # Cast to float32 for linear layer
        logits = self.output_layer(quantum_out)

        return logits


# =============================================================================
# MerLin Photonic Circuits
# =============================================================================


def create_angle_encoding_circuit(
    n_modes: int, n_features: int, n_layers: int = 2
) -> pcvl.Circuit:
    """Create angle encoding circuit for VQC classifier.

    Architecture (sandwich pattern from Mari et al.):
    [Trainable BS Mesh] → [Angle Encoding via PS] → [Trainable BS Mesh(es)]

    Data is encoded ONCE into phase shifter rotation angles (not at every layer).
    Additional trainable layers add expressivity without re-encoding.

    Args:
        n_modes: Number of optical modes
        n_features: Number of input features to encode
        n_layers: Number of trainable blocks (minimum 2 for sandwich)

    Returns:
        Perceval Circuit with trainable theta parameters and input x parameters
    """
    circuit = pcvl.Circuit(n_modes)

    # Ensure at least 2 layers for sandwich architecture
    n_layers = max(2, n_layers)

    # Counter for unique trainable parameter names
    trainable_counter = [0]

    for layer_idx in range(n_layers):
        # Trainable beam splitter mesh
        def make_bs_component(idx, layer=layer_idx, counter=trainable_counter):
            param_name = f"theta_{layer}_{counter[0]}"
            counter[0] += 1
            return pcvl.BS(theta=pcvl.P(param_name)) // (
                0,
                pcvl.PS(phi=np.pi * 2 * random.random()),
            )

        bs_mesh = pcvl.GenericInterferometer(
            n_modes,
            make_bs_component,
            shape=pcvl.InterferometerShape.RECTANGLE,
            depth=n_modes,  # Reduced depth for efficiency
            phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi * 2 * random.random()),
        )
        circuit.add(0, bs_mesh, merge=True)

        # Angle encoding layer - ONLY ONCE after the first trainable layer
        if layer_idx == 0:
            # Center the encoding on the modes
            start_mode = max(0, (n_modes - n_features) // 2)
            for i in range(min(n_features, n_modes)):
                mode = start_mode + i
                if mode < n_modes:
                    # Input parameters named x0, x1, x2, ...
                    circuit.add(mode, pcvl.PS(pcvl.P(f"x{i}")))

    return circuit


def create_classifier_circuit(
    n_modes: int, n_features: int, n_layers: int = 1
) -> pcvl.Circuit:
    """Create a variational quantum classifier circuit using Perceval.

    Architecture follows the VQC pattern from the paper:
    [Trainable Layer] → [Encoding] → [Trainable Layer] → ... (repeated n_layers times)

    For photonic implementation:
    - Trainable layers: Beam splitter meshes
    - Encoding: Phase shifters for angle encoding

    Args:
        n_modes: Number of optical modes
        n_features: Number of input features to encode
        n_layers: Number of variational layers (depth p in the paper)

    Returns:
        Perceval Circuit with trainable and input parameters
    """
    circuit = pcvl.Circuit(n_modes)

    # Use a counter to ensure unique parameter names across all layers
    bs_counter = [0]  # Use list to allow modification in lambda

    for layer_idx in range(n_layers):
        # Trainable beam splitter mesh (rotation + entangling layer)
        # Capture layer_idx in closure properly
        def make_bs_component(idx, layer=layer_idx, counter=bs_counter):
            param_name = f"theta_{layer}_{counter[0]}"
            counter[0] += 1
            return pcvl.BS(theta=pcvl.P(param_name)) // (
                0,
                pcvl.PS(phi=np.pi * 2 * random.random()),
            )

        bs_mesh = pcvl.GenericInterferometer(
            n_modes,
            make_bs_component,
            shape=pcvl.InterferometerShape.RECTANGLE,
            depth=2 * n_modes,
            phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi * 2 * random.random()),
        )
        circuit.add(0, bs_mesh, merge=True)

        # Angle encoding layer (except for the last layer)
        if layer_idx < n_layers - 1:
            start_mode = max(0, (n_modes - n_features) // 2)
            for i in range(min(n_features, n_modes)):
                mode = start_mode + i
                if mode < n_modes:
                    # Use layer-indexed names to avoid duplicate parameter names
                    circuit.add(mode, pcvl.PS(pcvl.P(f"x{layer_idx}_{i}")))

    return circuit


def create_deep_classifier_circuit(
    n_modes: int, n_features: int, n_layers: int
) -> pcvl.Circuit:
    """Create a deeper classifier circuit with full encoding at each layer.

    This matches the paper's description more closely where encoding
    happens at every layer.

    Args:
        n_modes: Number of optical modes
        n_features: Number of input features
        n_layers: Number of layers (p in the paper)

    Returns:
        Perceval Circuit
    """
    circuit = pcvl.Circuit(n_modes)

    # Use a counter to ensure unique parameter names
    bs_counter = [0]

    for layer_idx in range(n_layers):
        # Initial encoding (at each layer for re-encoding scheme)
        start_mode = max(0, (n_modes - n_features) // 2)
        for i in range(min(n_features, n_modes)):
            mode = start_mode + i
            if mode < n_modes:
                circuit.add(mode, pcvl.PS(pcvl.P(f"x{layer_idx}_{i}")))

        # Trainable beam splitter mesh
        def make_bs_component(idx, layer=layer_idx, counter=bs_counter):
            param_name = f"theta_{layer}_{counter[0]}"
            counter[0] += 1
            return pcvl.BS(theta=pcvl.P(param_name)) // (
                0,
                pcvl.PS(phi=np.pi * 2 * random.random()),
            )

        bs_mesh = pcvl.GenericInterferometer(
            n_modes,
            make_bs_component,
            shape=pcvl.InterferometerShape.RECTANGLE,
            depth=n_modes,
            phase_shifter_fun_gen=lambda idx: pcvl.PS(phi=np.pi * 2 * random.random()),
        )
        circuit.add(0, bs_mesh, merge=True)

    return circuit


class ScaleLayer(nn.Module):
    """Learnable or fixed scaling layer for inputs.

    Scales input features before encoding into quantum circuit.
    """

    def __init__(self, dim: int, scale_type: str = "learned"):
        """Initialize scale layer.

        Args:
            dim: Input dimension
            scale_type: 'learned', 'pi', '2pi', or '1'
        """
        super().__init__()

        if scale_type == "learned":
            self.scale = nn.Parameter(torch.rand(dim))
        elif scale_type == "2pi":
            self.register_buffer("scale", torch.full((dim,), 2 * np.pi))
        elif scale_type == "pi":
            self.register_buffer("scale", torch.full((dim,), np.pi))
        else:  # "1"
            self.register_buffer("scale", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input element-wise."""
        return x * self.scale


class MerLinQuantumClassifier(nn.Module):
    """MerLin-based variational quantum classifier with ANGLE ENCODING.

    Uses phase shifter encoding where classical data is encoded into
    rotation angles of phase shifters in the photonic circuit.

    Architecture (sandwich pattern):
        [Trainable BS Mesh] → [Angle Encoding via PS] → [Trainable BS Mesh(es)]

    Data is encoded ONCE after the first trainable layer. Additional trainable
    layers (n_layers > 2) add expressivity without re-encoding.

    Encoding: x_i → PS(φ = x_i * scale)

    This is different from amplitude encoding where data is encoded
    into the amplitudes of the quantum state directly.

    For amplitude encoding, use MerLinAmplitudeClassifier instead.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_modes: int = 8,
        n_photons: int = 2,
        n_layers: int = 2,
        computation_space: str = "unbunched",
        scale_type: str = "learned",
    ):
        """Initialize MerLin quantum classifier with angle encoding.

        Args:
            n_inputs: Number of input features (after preprocessing)
            n_outputs: Number of output classes
            n_modes: Number of optical modes
            n_photons: Number of photons in the input state
            n_layers: Number of trainable blocks (2 = sandwich architecture)
            computation_space: 'fock', 'unbunched', or 'dual_rail'
            scale_type: Input scaling type ('learned', 'pi', '2pi', '1')
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.n_layers = n_layers

        # Input scaling - important for angle encoding
        self.scale_layer = ScaleLayer(n_inputs, scale_type=scale_type)

        # Input projection to match n_modes if needed
        if n_inputs != n_modes:
            self.input_projection = nn.Sequential(
                nn.Linear(n_inputs, n_modes), nn.Tanh()
            )
        else:
            self.input_projection = nn.Tanh()

        # Parse computation space
        self.computation_space = ComputationSpace.coerce(computation_space)

        # Create angle encoding circuit
        # n_features = n_modes after projection
        circuit = create_angle_encoding_circuit(n_modes, n_modes, n_layers)

        # Determine trainable parameters
        all_params = [p.name for p in circuit.get_parameters()]
        trainable_params = [p for p in all_params if not p.startswith("x")]

        # Create initial state (dual-rail style)
        input_state = self._create_input_state(n_modes, n_photons)

        # Create MerLin QuantumLayer
        # input_size must match number of unique x parameters (x0, x1, ..., x_{n_modes-1})
        self.quantum_layer = QuantumLayer(
            input_size=n_modes,  # After projection, we encode n_modes features
            circuit=circuit,
            trainable_parameters=trainable_params,
            input_parameters=["x"],  # Prefix for input parameters
            input_state=input_state,
            computation_space=self.computation_space,
        )

        self._quantum_output_size = self.quantum_layer.output_size

        # Output classification layer
        self.output_layer = nn.Linear(self._quantum_output_size, n_outputs)

    def _create_input_state(self, n_modes: int, n_photons: int) -> list[int]:
        """Create initial Fock state."""
        state = [0] * n_modes
        for i in range(min(n_photons, n_modes)):
            state[i * 2 % n_modes] = 1 if i * 2 < n_modes else 0

        # Ensure correct photon number
        current = sum(state)
        idx = 0
        while current < n_photons and idx < n_modes:
            if state[idx] == 0:
                state[idx] = 1
                current += 1
            idx += 1

        return state

    @property
    def quantum_output_size(self) -> int:
        """Return output dimension of the quantum layer."""
        return self._quantum_output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum classifier.

        Args:
            x: Input tensor of shape (batch_size, n_inputs)

        Returns:
            Class logits of shape (batch_size, n_outputs)
        """
        input_device = x.device

        # Scale and project input
        x = self.scale_layer(x)
        x = self.input_projection(x)

        # Quantum layer
        x = self.quantum_layer(x)

        # Ensure output is on correct device
        x = x.float().to(input_device)

        # Classification
        logits = self.output_layer(x)

        return logits

    def get_quantum_state_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get the quantum layer output (for fidelity calculations).

        Args:
            x: Input tensor

        Returns:
            Quantum layer output (measurement probabilities)
        """
        input_device = x.device
        x = self.scale_layer(x)
        x = self.input_projection(x)
        x = self.quantum_layer(x)
        return x.float().to(input_device)


class MerLinAmplitudeClassifier(nn.Module):
    """Quantum classifier with proper amplitude encoding.

    This faithfully implements the paper's approach where classical data
    is encoded directly into quantum state amplitudes:

    |ψ_x⟩ = Σ_i x_i |i⟩  (normalized)

    The paper states: "The input classical data x is encoded into the
    amplitudes of an n-qubit quantum state |ψ_x⟩"

    For photonic implementation:
    - Fock states |n_1, n_2, ..., n_m⟩ with n_1 + ... + n_m = n_photons
    - Dimension (unbunched) = C(n_modes, n_photons)

    IMPORTANT: For faithful paper reproduction, choose n_modes and n_photons
    such that C(n_modes, n_photons) >= input_dim. If input_dim == fock_dim,
    no classical compression is used (matching the paper exactly).

    Recommended configurations:
    - MNIST 16x16 (256 dims): n_modes=13, n_photons=3 → C(13,3)=286
    - Topological (400 dims): n_modes=15, n_photons=3 → C(15,3)=455
    - Ising 8 spins (256 dims): n_modes=13, n_photons=3 → C(13,3)=286
    """

    def __init__(
        self,
        input_dim: int,
        n_outputs: int,
        n_modes: int = 13,
        n_photons: int = 3,
        n_layers: int = 2,
        hidden_dims: list[int] | None = None,
        computation_space: str = "unbunched",
    ):
        """Initialize amplitude-encoding quantum classifier.

        Args:
            input_dim: Dimension of input data (e.g., 256 for 16x16 images)
            n_outputs: Number of output classes
            n_modes: Number of optical modes (choose so C(n_modes,n_photons) >= input_dim)
            n_photons: Number of photons
            n_layers: Circuit depth
            hidden_dims: Hidden dimensions for classical encoder (only used if needed)
            computation_space: Computation space type ("unbunched" recommended)
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_outputs = n_outputs
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.n_layers = n_layers
        self.computation_space = computation_space

        # Compute state space dimension based on computation space
        from math import comb

        if computation_space == "fock":
            # Full Fock space: any number of photons per mode
            # Dimension = C(n_modes + n_photons - 1, n_photons)
            self.fock_dim = comb(n_modes + n_photons - 1, n_photons)
        else:
            # Unbunched/dual_rail: at most 1 photon per mode
            # Dimension = C(n_modes, n_photons)
            self.fock_dim = comb(n_modes, n_photons)

        # Determine if classical encoder is needed
        self.use_classical_encoder = input_dim != self.fock_dim

        if self.use_classical_encoder:
            if input_dim > self.fock_dim:
                logger.warning(
                    f"Input dim ({input_dim}) > Fock dim ({self.fock_dim}). "
                    f"Using classical compression. For faithful paper reproduction, "
                    f"increase n_modes or n_photons so C(n_modes,n_photons) >= {input_dim}."
                )
                # Classical dimensionality reduction
                if hidden_dims is None:
                    hidden_dims = [128, 64]
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, hidden_dim), nn.Tanh()])
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, self.fock_dim))
                self.classical_encoder = nn.Sequential(*layers)
            else:
                # input_dim < fock_dim: pad with zeros (handled in forward)
                logger.info(
                    f"Input dim ({input_dim}) < Fock dim ({self.fock_dim}). "
                    f"Will zero-pad inputs to match Fock space dimension."
                )
                self.classical_encoder = None
        else:
            # Exact match - no encoder needed (faithful to paper)
            logger.info(
                f"Input dim ({input_dim}) == Fock dim ({self.fock_dim}). "
                f"Direct amplitude encoding (faithful to paper)."
            )
            self.classical_encoder = None

        # Create trainable Perceval circuit
        circuit = self._create_trainable_circuit()

        # Get all trainable parameter names (all parameters are trainable, no input parameters)
        all_params = [p.name for p in circuit.get_parameters()]

        # Parse computation space
        comp_space = ComputationSpace.coerce(computation_space)

        # Create MerLin QuantumLayer with amplitude encoding
        # For amplitude encoding, we pass the state directly as amplitudes
        # All circuit parameters are trainable
        self.quantum_layer = QuantumLayer(
            circuit=circuit,
            n_photons=n_photons,
            trainable_parameters=all_params,
            amplitude_encoding=True,
            computation_space=comp_space,
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
        )

        # Get output size from the quantum layer
        self._quantum_output_size = self.quantum_layer.output_size

        # Output classification layer
        self.output_layer = nn.Linear(self._quantum_output_size, n_outputs)

    def _create_trainable_circuit(self) -> pcvl.Circuit:
        """Create trainable unitary circuit using beam splitter meshes."""
        circuit = pcvl.Circuit(self.n_modes)

        bs_counter = [0]

        for layer_idx in range(self.n_layers):
            # Trainable beam splitter mesh (universal unitary)
            def make_bs_component(idx, layer=layer_idx, counter=bs_counter):
                param_name = f"theta_{layer}_{counter[0]}"
                counter[0] += 1
                return pcvl.BS(theta=pcvl.P(param_name)) // (
                    0,
                    pcvl.PS(phi=np.pi * 2 * random.random()),
                )

            bs_mesh = pcvl.GenericInterferometer(
                self.n_modes,
                make_bs_component,
                shape=pcvl.InterferometerShape.RECTANGLE,
                depth=self.n_modes,
                phase_shifter_fun_gen=lambda idx: pcvl.PS(
                    phi=np.pi * 2 * random.random()
                ),
            )
            circuit.add(0, bs_mesh, merge=True)

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with amplitude encoding.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Class logits of shape (batch_size, n_outputs)
        """
        # Flatten if needed (for images)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Apply classical encoder if needed, otherwise use input directly
        if self.classical_encoder is not None:
            x = self.classical_encoder(x)
        elif self.input_dim < self.fock_dim:
            # Zero-pad to match Fock dimension
            padding = torch.zeros(
                x.size(0), self.fock_dim - self.input_dim, device=x.device
            )
            x = torch.cat([x, padding], dim=-1)
        # else: input_dim == fock_dim, use x directly

        # Normalize to unit L2 norm (required for amplitude encoding)
        # Convert to complex for quantum layer
        x_complex = x.to(torch.complex64)
        norms = torch.norm(x_complex, dim=-1, keepdim=True)
        x_normalized = x_complex / (norms + 1e-8)

        # Quantum layer with amplitude encoding
        quantum_out = self.quantum_layer(x_normalized)

        # Classification
        logits = self.output_layer(quantum_out)

        return logits


class MerLinAmplitudeClassifierDirect(nn.Module):
    """Direct amplitude encoding without classical reduction.

    Use when input dimension already matches Fock space dimension,
    e.g., for Ising model ground states or prepared quantum states.
    """

    def __init__(
        self,
        state_dim: int,
        n_outputs: int,
        n_modes: int = 8,
        n_photons: int = 2,
        n_layers: int = 2,
    ):
        """Initialize direct amplitude encoding classifier.

        Args:
            state_dim: Dimension of quantum state (must match Fock space)
            n_outputs: Number of output classes
            n_modes: Number of optical modes
            n_photons: Number of photons
            n_layers: Circuit depth
        """
        super().__init__()

        self.state_dim = state_dim
        self.n_outputs = n_outputs
        self.n_modes = n_modes
        self.n_photons = n_photons

        # Verify dimension compatibility
        from math import comb

        self.fock_dim = comb(n_modes + n_photons - 1, n_photons)

        if state_dim != self.fock_dim:
            # Add projection layer if dimensions don't match
            self.projection = nn.Linear(state_dim, self.fock_dim)
        else:
            self.projection = None

        # Create circuit
        circuit = self._create_circuit(n_layers)

        # Quantum layer with amplitude encoding
        self.quantum_layer = QuantumLayer(
            circuit=circuit,
            n_photons=n_photons,
            amplitude_encoding=True,
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
        )

        self._quantum_output_size = self.quantum_layer.output_size
        self.output_layer = nn.Linear(self._quantum_output_size, n_outputs)

    def _create_circuit(self, n_layers: int) -> pcvl.Circuit:
        """Create trainable circuit."""
        circuit = pcvl.Circuit(self.n_modes)

        bs_counter = [0]

        for layer_idx in range(n_layers):

            def make_bs_component(idx, layer=layer_idx, counter=bs_counter):
                param_name = f"theta_{layer}_{counter[0]}"
                counter[0] += 1
                return pcvl.BS(theta=pcvl.P(param_name)) // (
                    0,
                    pcvl.PS(phi=np.pi * 2 * random.random()),
                )

            bs_mesh = pcvl.GenericInterferometer(
                self.n_modes,
                make_bs_component,
                shape=pcvl.InterferometerShape.RECTANGLE,
                depth=self.n_modes,
                phase_shifter_fun_gen=lambda idx: pcvl.PS(
                    phi=np.pi * 2 * random.random()
                ),
            )
            circuit.add(0, bs_mesh, merge=True)

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Quantum state amplitudes of shape (batch_size, state_dim)

        Returns:
            Class logits
        """
        if self.projection is not None:
            x = self.projection(x)

        # Normalize and convert to complex
        x_complex = x.to(torch.complex64)
        x_normalized = x_complex / (torch.norm(x_complex, dim=-1, keepdim=True) + 1e-8)

        # Quantum evolution
        quantum_out = self.quantum_layer(x_normalized)

        return self.output_layer(quantum_out)
