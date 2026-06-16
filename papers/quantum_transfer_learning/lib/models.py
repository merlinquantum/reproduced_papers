"""
Hybrid Classical-Quantum Models
================================

Implements the full models from Mari et al. (2020):
- HybridModel: Base hybrid model with MerLin or PennyLane backend
- CQTransferModel: Classical-to-Quantum transfer learning model
- ClassicalBaseline: Classical comparison model

Supports both:
- MerLin (photonic): Primary implementation using QuantumLayer
- PennyLane (qubit): Reference implementation for comparison
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision.models as models

from .circuits import (
    DressedQuantumCircuit,
    MerLinSimpleLayer,
    ScaleLayer,
)


class ClassicalBaseline(nn.Module):
    """Classical neural network baseline for comparison.
    
    A simple MLP with configurable hidden layers, matching
    the parameter count of the quantum model approximately.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden_sizes: list = [4],
        activation: str = "tanh"
    ):
        """Initialize classical baseline.
        
        Args:
            n_inputs: Number of input features
            n_outputs: Number of output classes
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('tanh', 'relu')
        """
        super().__init__()

        activation_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()

        layers = []
        prev_size = n_inputs

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation_fn
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, n_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class HybridModel(nn.Module):
    """Base hybrid classical-quantum model.
    
    Wraps a dressed quantum circuit and handles the interface
    between classical data and quantum processing.
    
    Supports both MerLin (photonic) and PennyLane (qubit) backends.
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
        """Initialize hybrid model.
        
        Args:
            n_inputs: Number of input features
            n_outputs: Number of output classes
            n_qubits: Number of qubits/modes
            q_depth: Quantum circuit depth
            backend: 'merlin' or 'pennylane'
            **kwargs: Additional backend-specific arguments
                - n_photons: Number of photons (MerLin)
                - computation_space: 'fock', 'unbunched', 'dual_rail' (MerLin)
                - merlin_depth: Circuit depth for MerLin
                - scale_type: Input scaling type
        """
        super().__init__()

        self.backend = backend

        self.dressed_circuit = DressedQuantumCircuit(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_qubits=n_qubits,
            q_depth=q_depth,
            backend=backend,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid model."""
        return self.dressed_circuit(x)


class FeatureExtractor(nn.Module):
    """Pre-trained CNN feature extractor.
    
    Uses a pre-trained model (e.g., ResNet18) with the final
    classification layer removed to extract features.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = True
    ):
        """Initialize feature extractor.
        
        Args:
            model_name: Name of the pretrained model
            pretrained: Use pretrained weights
            freeze: Freeze the feature extractor weights
        """
        super().__init__()

        # Load pretrained model
        if model_name == "resnet18":
            self.model = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.n_features = 512
        elif model_name == "resnet34":
            self.model = models.resnet34(
                weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.n_features = 512
        elif model_name == "resnet50":
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.n_features = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Remove final classification layer
        self.model.fc = nn.Identity()

        # Optionally freeze weights
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        return self.model(x)


class CQTransferModel(nn.Module):
    """Classical-to-Quantum Transfer Learning Model.
    
    Implements the CQ transfer learning scheme from the paper:
    1. Pre-trained CNN extracts features (512-dim for ResNet18)
    2. Dressed quantum circuit processes features
    3. Outputs class logits
    
    Architecture:
        [Image] → [ResNet18] → [512 features] → [L_512→n] → [VQC] → [L_n→2] → [2 classes]
    
    Supports both MerLin (photonic) and PennyLane (qubit) backends.
    """

    def __init__(
        self,
        n_outputs: int = 2,
        n_qubits: int = 4,
        q_depth: int = 6,
        feature_extractor: str = "resnet18",
        pretrained: bool = True,
        freeze_extractor: bool = True,
        backend: str = "merlin",
        **kwargs
    ):
        """Initialize CQ transfer learning model.
        
        Args:
            n_outputs: Number of output classes
            n_qubits: Number of qubits/modes
            q_depth: Quantum circuit depth
            feature_extractor: Pre-trained model name
            pretrained: Use pretrained weights
            freeze_extractor: Freeze feature extractor
            backend: 'merlin' or 'pennylane'
            **kwargs: Additional backend-specific arguments
        """
        super().__init__()

        self.backend = backend

        # Feature extractor (classical pre-trained)
        self.feature_extractor = FeatureExtractor(
            model_name=feature_extractor,
            pretrained=pretrained,
            freeze=freeze_extractor
        )

        n_features = self.feature_extractor.n_features

        # Dressed quantum circuit
        self.quantum_classifier = DressedQuantumCircuit(
            n_inputs=n_features,
            n_outputs=n_outputs,
            n_qubits=n_qubits,
            q_depth=q_depth,
            backend=backend,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image → features → quantum → logits.
        
        Args:
            x: Input images of shape (batch_size, 3, H, W)
            
        Returns:
            Class logits of shape (batch_size, n_outputs)
        """
        # Extract features (frozen CNN)
        features = self.feature_extractor(x)

        # Quantum classification
        logits = self.quantum_classifier(features)

        return logits

    def get_trainable_params(self):
        """Get only the trainable parameters (quantum circuit)."""
        return [p for p in self.parameters() if p.requires_grad]


class MerLinCQTransferModel(nn.Module):
    """MerLin-specific CQ Transfer Learning Model.
    
    Uses MerLin's QuantumLayer.simple() for a quick-start photonic circuit.
    """

    def __init__(
        self,
        n_outputs: int = 2,
        n_params: int = 90,
        feature_extractor: str = "resnet18",
        computation_space: str = "unbunched"
    ):
        """Initialize MerLin-specific CQ model.
        
        Args:
            n_outputs: Number of output classes
            n_params: Number of trainable quantum parameters
            feature_extractor: Pre-trained model name
            computation_space: 'fock', 'unbunched', or 'dual_rail'
        """
        super().__init__()

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            model_name=feature_extractor,
            pretrained=True,
            freeze=True
        )

        n_features = self.feature_extractor.n_features

        # Dimensionality reduction to match quantum layer input
        self.pre_quantum = nn.Sequential(
            nn.Linear(n_features, 6),
            nn.Tanh()
        )

        # MerLin quantum layer using simple() factory
        self.quantum_layer = MerLinSimpleLayer(
            input_size=6,
            n_params=n_params,
            computation_space=computation_space
        )

        # Post-processing
        self.post_quantum = nn.Linear(self.quantum_layer.output_size, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.feature_extractor(x)
        x = self.pre_quantum(features)
        x = self.quantum_layer(x)
        x = self.post_quantum(x)
        return x


class MerLinVQCModel(nn.Module):
    """MerLin VQC model following the paper reproduction pattern.
    
    Architecture:
        [ScaleLayer] → [MerLin QuantumLayer] → [Linear] → [output]
    
    Similar to the VQC example in MerLin documentation.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_modes: int = 4,
        n_photons: int = 2,
        computation_space: str = "unbunched",
        scale_type: str = "learned",
        activation: str = "none"
    ):
        """Initialize MerLin VQC model.
        
        Args:
            n_inputs: Number of input features
            n_outputs: Number of output classes
            n_modes: Number of optical modes
            n_photons: Number of photons
            computation_space: 'fock', 'unbunched', or 'dual_rail'
            scale_type: Input scaling type
            activation: Output activation ('none', 'sigmoid', 'softmax')
        """
        super().__init__()

        from .circuits import MerLinQuantumLayer

        self.scale_layer = ScaleLayer(n_inputs, scale_type=scale_type)

        # If n_inputs != n_modes, add projection layer
        if n_inputs != n_modes:
            self.projection = nn.Linear(n_inputs, n_modes)
        else:
            self.projection = None

        self.quantum_layer = MerLinQuantumLayer(
            n_modes=n_modes,
            n_features=n_modes,
            n_photons=n_photons,
            computation_space=computation_space
        )

        # Output layer
        if activation == "none":
            self.output_layer = nn.Linear(self.quantum_layer.output_size, n_outputs)
        elif activation == "sigmoid":
            self.output_layer = nn.Sequential(
                nn.Linear(self.quantum_layer.output_size, n_outputs),
                nn.Sigmoid()
            )
        elif activation == "softmax":
            self.output_layer = nn.Sequential(
                nn.Linear(self.quantum_layer.output_size, n_outputs),
                nn.Softmax(dim=1)
            )
        else:
            self.output_layer = nn.Linear(self.quantum_layer.output_size, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.scale_layer(x)
        if self.projection is not None:
            x = self.projection(x)
        x = self.quantum_layer(x)
        x = self.output_layer(x)
        return x


def create_model(config: Dict[str, Any], backend: str = "merlin") -> nn.Module:
    """Factory function to create models from config.
    
    Args:
        config: Model configuration dictionary
        backend: 'merlin' or 'pennylane'
        
    Returns:
        Initialized model
    """
    model_type = config.get("type", "dressed_quantum")

    if model_type == "dressed_quantum":
        return HybridModel(
            n_inputs=config.get("n_inputs", 2),
            n_outputs=config.get("n_outputs", 2),
            n_qubits=config.get("n_qubits", 4),
            q_depth=config.get("q_depth", 5),
            backend=backend,
            n_photons=config.get("n_photons", 2),
            computation_space=config.get("computation_space", "unbunched")
        )

    elif model_type == "cq_transfer":
        return CQTransferModel(
            n_outputs=config.get("n_outputs", 2),
            n_qubits=config.get("n_qubits", 4),
            q_depth=config.get("q_depth", 6),
            feature_extractor=config.get("feature_extractor", "resnet18"),
            backend=backend,
            n_photons=config.get("n_photons", 2),
            computation_space=config.get("computation_space", "unbunched")
        )

    elif model_type == "merlin_vqc":
        return MerLinVQCModel(
            n_inputs=config.get("n_inputs", 2),
            n_outputs=config.get("n_outputs", 2),
            n_modes=config.get("n_modes", 4),
            n_photons=config.get("n_photons", 2),
            computation_space=config.get("computation_space", "unbunched"),
            scale_type=config.get("scale_type", "learned"),
            activation=config.get("activation", "none")
        )

    elif model_type == "merlin_simple_cq":
        return MerLinCQTransferModel(
            n_outputs=config.get("n_outputs", 2),
            n_params=config.get("n_params", 90),
            feature_extractor=config.get("feature_extractor", "resnet18"),
            computation_space=config.get("computation_space", "unbunched")
        )

    elif model_type == "classical":
        return ClassicalBaseline(
            n_inputs=config.get("n_inputs", 2),
            n_outputs=config.get("n_outputs", 2),
            hidden_sizes=config.get("hidden_sizes", [4]),
            activation=config.get("activation", "tanh")
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
