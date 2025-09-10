"""
TorchQuantum model training utilities for QLLM experiments using MerLin.
"""

# to install torchquantum: clone the original repo https://github.com/mit-han-lab/torchquantum/tree/main
# pip install --editable . (I think pip install torchquantum is not up to date with latest qiskit versions)
# some guidance can be found in the original GitHub repo: https://github.com/mit-han-lab/torchquantum/tree/main

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

#####################
### Data Encoding ###
#####################

### Amplitude encoding ###
"""
Quote: The classical data values are embedded into the amplitudes of the quantum state.
Specifically, a classical vector x is encoded as:  |ψ(x)⟩ =  n  X  i=1  xi |i⟩
Since the process is simulated, the state can be directly “loaded” without explicitly
constructing a quantum circuit to prepare |ψ(x)⟩.
"""


class AmplitudeEncodingModule(tq.QuantumModule):
    """
    Amplitude encoding module for embedding classical feature vectors
    Implements |ψ(x)⟩ = Σᵢ xᵢ |i⟩ encoding from classical data
    """

    def __init__(self, n_qubits: int, normalize: bool = True):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_amplitudes = 2**n_qubits
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode classical vector into quantum amplitudes
        Args:
            x: Classical feature vector (batch_size, feature_dim)
        Returns:
            Processed feature vector for quantum device
        """
        batch_size, feature_dim = x.shape

        # Ensure feature dimension matches number of amplitudes
        if feature_dim > self.n_amplitudes:
            amplitudes = x[:, : self.n_amplitudes]
        elif feature_dim < self.n_amplitudes:
            padding = torch.zeros(
                batch_size,
                self.n_amplitudes - feature_dim,
                device=x.device,
                dtype=x.dtype,
            )
            amplitudes = torch.cat([x, padding], dim=1)
        else:
            amplitudes = x

        # Normalize to ensure valid quantum state (|ψ|² = 1)
        if self.normalize:
            amplitudes = torch.nn.functional.normalize(amplitudes, p=2, dim=1)

        return amplitudes


### Angle encoding ###
"""
Quote: In this scheme, classical data determines the phases of qubits.
The qubits are rotated around the Bloch sphere by an angle proportional to the respective data values  x = x1 x2 . . . xn  ⊤.
An example of angular encoding, using the y-axis is:  |ψ(x)⟩ = RY (x1) ⊗ RY (x2) ⊗ · · · ⊗ RY (xn) |0⟩⊗n
"""


class AngleEncodingModule(tq.QuantumModule):
    """
    Angle encoding module for loading data into QPU
    Implements |ψ(x)⟩ = RY(x₁) ⊗ RY(x₂) ⊗ ... ⊗ RY(xₙ) |0⟩⊗n
    """

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        # RY rotation gates for angle encoding
        self.ry_gates = tq.QuantumModuleList()
        for _ in range(n_qubits):
            self.ry_gates.append(tq.RY(has_params=False, trainable=False))

    def forward(
        self,
        qdev: tq.QuantumDevice,
        data_vector: torch.Tensor,
        reset_state: bool = True,
    ) -> tq.QuantumDevice:
        """Encode data vector using angle encoding"""
        batch_size = data_vector.shape[0]

        # Only reset quantum device if explicitly requested (for initial encoding)
        if reset_state:
            qdev.reset_states(batch_size)

        # Apply RY rotations with angles from data vector
        for i, ry_gate in enumerate(self.ry_gates):
            if i < data_vector.shape[1]:
                # Use data vector values as rotation angles, scaled properly
                angles = data_vector[:, i] * np.pi
                ry_gate(qdev, wires=i, params=angles)

        return qdev


#####################################
### Parametrized Quantum Circuits ###
#####################################

# simulated PQC whose expectation values of the observables (Pauli-Z) are computed exactly and used as a latent representation.
# the paper states that this is equivalent to an autoencoder
"""
Quote: the ansatz used as a fundamental building block in the quantum circuit,
which consists of layers with varying connectivity spanning the width of the circuit.
This design facilitates operation on hardware platforms that can operate beyond nearest neighbor pairs of qubits
Fig2.png presents this ansatz : The ansatz (left) consists of layers with
- increasing connectivity C across the width of the circuit.
- Each block Ui is defined as a combination of a controlled NOT and a single-qubit rotation RY through angle θi around the Y-axis
Fig3.png presents the complete set-up
This simulated PQC contains a variable number of E parallel encoders which encode
the embedding vectors from SetFit of dimension equal to 768 into output vectors of dimension Qc
"""

""" We build the QLLM as follows:
- a SimpleParametrizedQuantumCircuit implements one branch of the simulated PQC (which can contain up to E = 2 SimpleParametrizedQuantumCircuit)
        INPUTS: n_qubits (Q), n_layers (M), connectivity (C)
- a SimulatedQuantumEncoder implements the sQE with E = 1 as per the paper, combining the amplitude encoding and the SimpleParametrizedQuantumCircuit
        INPUTS: n_qubits, n_layers, connectivity
- a MultiSimulatedQuantumEncoder which consists of the multiSQE of the paper and parallelised E SimulatedQuantumEncoder
        INPUTS: configs of the encoders, fusion method
        OUTPUTS: measurements based on the chosen fusion method (not precised in the paper)
- a Quantum Processing Unit: the actual quantum processing unit with data re-uploading and angle encoding
        INPUTS: n_qubits (Q), n_main_layers (R), n_reuploading (N), connectivity (C)
"""


class SimpleParameterizedQuantumCircuit(tq.QuantumModule):
    """
    Simple parameterized quantum circuit following the working single encoder pattern.
    Each layer contains CNOT + RY blocks with proper gradient flow.
    """

    def __init__(self, n_qubits: int, n_layers: int, connectivity: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.connectivity = connectivity

        # Build parameterized layers
        self.quantum_layers = nn.ModuleList()

        for _layer_idx in range(n_layers):
            layer_gates = nn.ModuleList()

            # Create CNOT + RY blocks for this layer
            for offset in range(1, connectivity + 1):
                for qubit_idx in range(n_qubits):
                    target_qubit = (qubit_idx + offset) % n_qubits
                    if target_qubit != qubit_idx:
                        # CNOT gate
                        layer_gates.append(tq.CNOT())
                        # RY rotation on target
                        layer_gates.append(tq.RY(has_params=True, trainable=True))

            self.quantum_layers.append(layer_gates)

        # Store wire patterns for each layer
        self.wire_patterns = []
        for _layer_idx in range(n_layers):
            layer_wires = []
            for offset in range(1, connectivity + 1):
                for qubit_idx in range(n_qubits):
                    target_qubit = (qubit_idx + offset) % n_qubits
                    if target_qubit != qubit_idx:
                        # CNOT wires
                        layer_wires.append([qubit_idx, target_qubit])
                        # RY wires
                        layer_wires.append([target_qubit])
            self.wire_patterns.append(layer_wires)

    def forward(self, qdev: tq.QuantumDevice) -> tq.QuantumDevice:
        """Apply parameterized quantum circuit"""

        # Apply each layer
        for layer_gates, layer_wires in zip(self.quantum_layers, self.wire_patterns):
            for gate, wires in zip(layer_gates, layer_wires):
                gate(qdev, wires=wires)

        return qdev


class SimulatedQuantumEncoder(tq.QuantumModule):
    """
    First module: Simulated quantum encoder with amplitude encoding
    """

    def __init__(self, n_qubits: int, n_layers: int, connectivity: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.connectivity = connectivity

        # Amplitude encoding
        self.amplitude_encoder = AmplitudeEncodingModule(n_qubits)

        # Parameterized quantum circuit
        self.pqc = SimpleParameterizedQuantumCircuit(n_qubits, n_layers, connectivity)

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through simulated quantum encoder
        Args:
            x: Classical feature vector from LLM layer
        Returns:
            Exact expectation values (latent representation)
        """
        batch_size = x.shape[0]

        # Step 1: Amplitude encoding (preprocess features)
        amplitudes = self.amplitude_encoder(x)

        # Step 2: Create quantum device and initialize state
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch_size, device=x.device)
        qdev.reset_states(bsz=batch_size)

        # Step 3: Initialize quantum state with amplitudes
        try:
            # Try using StatePreparation if available
            state_prep = tq.StatePreparation()
            state_prep(qdev, amplitudes)
        except (ValueError, RuntimeError, AttributeError):
            # Fallback: use RY rotations to approximate amplitude encoding
            for i in range(self.n_qubits):
                if i < amplitudes.shape[1]:
                    angle = torch.arccos(torch.clamp(amplitudes[:, i], -1, 1))
                    tq.RY(has_params=False, trainable=False)(
                        qdev, wires=i, params=angle
                    )

        # Step 4: Apply parameterized quantum circuit
        qdev = self.pqc(qdev)

        # Step 5: Exact expectation value computation (noiseless)
        expectation_values = self.measure(qdev)

        return expectation_values


class MultiSimulatedQuantumEncoder(nn.Module):
    """
    Multi-encoder design for the first module with different fusion strategies
    """

    def __init__(self, encoder_configs: list[dict], fusion_method: str = "concatenate"):
        super().__init__()
        self.encoder_configs = encoder_configs
        self.fusion_method = fusion_method
        self.n_encoders = len(encoder_configs)

        # Create multiple simulated quantum encoders
        self.encoders = nn.ModuleList()
        for config in encoder_configs:
            encoder = SimulatedQuantumEncoder(
                n_qubits=config["n_qubits"],
                n_layers=config["n_layers"],
                connectivity=config.get("connectivity", 1),
            )
            self.encoders.append(encoder)

        # Fusion weights for weighted combination
        if fusion_method == "weighted":
            self.fusion_weights = nn.Parameter(torch.ones(self.n_encoders))

        # Calculate output dimension
        if fusion_method == "concatenate":
            self.output_dim = sum(config["n_qubits"] for config in encoder_configs)
        else:
            # For average/weighted, output dim is max of encoder outputs
            self.output_dim = max(config["n_qubits"] for config in encoder_configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-encoder system"""
        encoder_outputs = []

        # Each encoder processes full input
        for encoder in self.encoders:
            encoder_output = encoder(x)
            encoder_outputs.append(encoder_output)

        # Fuse encoder outputs
        if self.fusion_method == "concatenate":
            fused_output = torch.cat(encoder_outputs, dim=1)

        elif self.fusion_method == "average":
            # Average encoder outputs (pad shorter ones with zeros)
            max_dim = max(output.shape[1] for output in encoder_outputs)
            padded_outputs = []

            for output in encoder_outputs:
                if output.shape[1] < max_dim:
                    padding = torch.zeros(
                        output.shape[0], max_dim - output.shape[1], device=output.device
                    )
                    padded_output = torch.cat([output, padding], dim=1)
                else:
                    padded_output = output[:, :max_dim]
                padded_outputs.append(padded_output)

            fused_output = torch.stack(padded_outputs, dim=0).mean(dim=0)

        elif self.fusion_method == "weighted":
            # Weighted combination of encoder outputs
            max_dim = max(output.shape[1] for output in encoder_outputs)
            padded_outputs = []

            for output in encoder_outputs:
                if output.shape[1] < max_dim:
                    padding = torch.zeros(
                        output.shape[0], max_dim - output.shape[1], device=output.device
                    )
                    padded_output = torch.cat([output, padding], dim=1)
                else:
                    padded_output = output[:, :max_dim]
                padded_outputs.append(padded_output)

            # Apply learned weights
            weights = torch.softmax(self.fusion_weights, dim=0)
            weighted_outputs = []
            for i, output in enumerate(padded_outputs):
                weighted_outputs.append(weights[i] * output)

            fused_output = torch.stack(weighted_outputs, dim=0).sum(dim=0)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused_output


class QuantumProcessingUnit(tq.QuantumModule):
    """
    Second module: Actual quantum processing unit with data re-uploading
    This implementation is noiseless
    """

    def __init__(
        self,
        n_qubits: int,
        n_main_layers: int,
        n_reuploading: int,
        connectivity: int = 1,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_main_layers = n_main_layers
        self.n_reuploading = n_reuploading
        self.connectivity = connectivity

        # Angle encoding for data loading/re-loading
        self.angle_encoder = AngleEncodingModule(n_qubits)

        # Main parameterized quantum circuit
        self.main_pqc = SimpleParameterizedQuantumCircuit(
            n_qubits, n_main_layers, connectivity
        )

        # Re-uploading circuits (one for each re-upload)
        # REMARK : from the paper, we could understand that the same pqc is used multiple times (drop in performance)
        self.reuploading_pqcs = nn.ModuleList()
        # pqc = SimpleParameterizedQuantumCircuit(n_qubits, 1, connectivity)
        for _ in range(n_reuploading):
            pqc = SimpleParameterizedQuantumCircuit(
                n_qubits, 1, connectivity
            )  # Single layer per re-upload
            self.reuploading_pqcs.append(pqc)

        # Measurement (single qubit as per paper: Qm = 1)
        self.measured_qubit = 0
        self.observable = tq.PauliZ()

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice = None) -> torch.Tensor:
        """
        Forward pass through quantum processing unit with data re-uploading
        """
        batch_size = x.shape[0]

        if qdev is None:
            qdev = tq.QuantumDevice(
                n_wires=self.n_qubits, bsz=batch_size, device=x.device
            )

        # Initial angle encoding
        qdev = self.angle_encoder(qdev, x)

        # Data re-uploading blocks
        for reuploading_pqc in self.reuploading_pqcs:
            # Re-encode data (don't reset state, just apply additional rotations)
            qdev = self.angle_encoder(qdev, x, reset_state=False)
            # Apply parameterized circuit
            qdev = reuploading_pqc(qdev)

        # Main parameterized circuit - ensure it's applied
        qdev = self.main_pqc(qdev)
        # Measure single qubit (Qm = 1 as per paper)
        output = tq.expval(
            qdev, wires=[self.measured_qubit], observables=[self.observable]
        )

        return output


class QLLM(nn.Module):
    """
    Complete QLLM model with modular design
    """

    def __init__(
        self,
        llm_output_dim: int = 768,
        encoder_configs: list[dict] | None = None,
        qpu_config: dict | None = None,
        fusion_method: str = "concatenate",
        n_classes: int = 2,
    ):
        super().__init__()

        if encoder_configs is None:
            encoder_configs = [{"n_qubits": 10, "n_layers": 3, "connectivity": 2}]

        if qpu_config is None:
            qpu_config = {
                "n_qubits": 10,
                "n_main_layers": 3,
                "n_reuploading": 2,
                "connectivity": 2,
            }

        self.llm_output_dim = llm_output_dim
        self.n_classes = n_classes

        # First module: Multi-encoder simulated quantum block
        self.first_module = MultiSimulatedQuantumEncoder(
            encoder_configs=encoder_configs, fusion_method=fusion_method
        )

        # Second module: Quantum processing unit
        self.second_module = QuantumProcessingUnit(
            n_qubits=qpu_config["n_qubits"],
            n_main_layers=qpu_config["n_main_layers"],
            n_reuploading=qpu_config["n_reuploading"],
            connectivity=qpu_config.get("connectivity", 2),
        )

        # Calculate combined input dimension for final linear layer
        first_module_dim = self.first_module.output_dim  # Qc
        second_module_dim = 1  # Qm = 1 (single qubit measurement)
        combined_dim = first_module_dim + second_module_dim

        # Output layer: (Qc + 1) × k parameters as per paper
        self.output_layer = nn.Linear(combined_dim, n_classes)

        # Store configurations
        self.encoder_configs = encoder_configs
        self.qpu_config = qpu_config

    def forward(self, llm_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete QLLM model
        """
        batch_size = llm_features.shape[0]

        # First module: Multi-encoder simulated quantum processing
        first_output = self.first_module(llm_features)

        # Prepare input for second module (QPU)
        n_qpu_qubits = self.qpu_config["n_qubits"]
        if first_output.shape[1] > n_qpu_qubits:
            # Use first n_qpu_qubits components
            qpu_input = first_output[:, :n_qpu_qubits]
        elif first_output.shape[1] < n_qpu_qubits:
            # Pad with zeros
            padding = torch.zeros(
                batch_size,
                n_qpu_qubits - first_output.shape[1],
                device=first_output.device,
            )
            qpu_input = torch.cat([first_output, padding], dim=1)
        else:
            qpu_input = first_output

        # Second module: Quantum processing unit
        second_output = self.second_module(qpu_input)

        # Combine outputs from both modules
        if second_output.dim() == 1:
            second_output = second_output.unsqueeze(1)

        # REMARK: there is an inconcistency in the paper with regards to the input of the Linear Layer
        # only 1 qubits is measured but this layer should have an input of size Q_c + 1
        combined_features = torch.cat([first_output, second_output], dim=1)

        # Final classification
        logits = self.output_layer(combined_features)

        return logits


def test_gradient_propagation():
    """
    Test gradient propagation through the implementation
    """
    print("=== Testing Implementation Gradient Flow ===\n")

    # Test parameters
    batch_size = 4
    llm_output_dim = 16
    n_classes = 2

    # Create model
    encoder_configs = [
        {"n_qubits": 4, "n_layers": 2, "connectivity": 1},
        {"n_qubits": 3, "n_layers": 2, "connectivity": 1},
    ]
    qpu_config = {
        "n_qubits": 4,
        "n_main_layers": 2,
        "n_reuploading": 1,
        "connectivity": 1,
    }

    model = QLLM(
        llm_output_dim=llm_output_dim,
        encoder_configs=encoder_configs,
        qpu_config=qpu_config,
        fusion_method="concatenate",
        n_classes=n_classes,
    )

    # Create dummy data
    dummy_features = torch.randn(batch_size, llm_output_dim, requires_grad=True)
    _dummy_targets = torch.randint(0, n_classes, (batch_size,))

    # Forward pass
    outputs = model(dummy_features)

    # Create loss
    loss = outputs.sum()  # Simple loss for gradient test

    # Backward pass
    loss.backward()

    # Check gradients
    print("Checking gradient flow...")
    gradient_stats = {"has_gradient": 0, "no_gradient": 0, "total_params": 0}

    for name, param in model.named_parameters():
        if param.requires_grad:
            gradient_stats["total_params"] += 1
            has_grad = param.grad is not None and param.grad.abs().sum() > 0

            if has_grad:
                gradient_stats["has_gradient"] += 1
                grad_norm = param.grad.norm().item()
                print(f"  ✓ {name}: grad_norm = {grad_norm:.6f}")
            else:
                gradient_stats["no_gradient"] += 1
                print(f"  ✗ {name}: NO GRADIENT")

    print("\nGradient Summary:")
    print(f"  Parameters with gradients: {gradient_stats['has_gradient']}")
    print(f"  Parameters without gradients: {gradient_stats['no_gradient']}")
    print(f"  Total trainable parameters: {gradient_stats['total_params']}")

    success = gradient_stats["no_gradient"] == 0
    print(f"  Gradient test: {'PASS' if success else 'FAIL'}")

    return success


if __name__ == "__main__":
    # Test the simplified implementation
    test_gradient_propagation()
