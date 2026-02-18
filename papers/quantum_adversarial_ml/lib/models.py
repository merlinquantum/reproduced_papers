"""
Quantum Classifier Models
=========================

Model architectures for the quantum adversarial learning experiments:
- QuantumClassifier: Direct quantum circuit classifier
- HybridQuantumClassifier: Classical preprocessing + quantum circuit
- ClassicalBaseline: Classical MLP for comparison
"""

from typing import Any

import torch
import torch.nn as nn

from .circuits import MerLinAmplitudeClassifier, MerLinQuantumClassifier


class QuantumClassifier(nn.Module):
    """Quantum circuit classifier for low-dimensional inputs.

    Direct quantum classification without heavy preprocessing.
    Suitable for spiral dataset or pre-processed features.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_modes: int = 4,
        n_photons: int = 2,
        n_layers: int = 2,
        computation_space: str = "unbunched",
        scale_type: str = "learned",
    ):
        """Initialize quantum classifier.

        Args:
            n_inputs: Number of input features
            n_outputs: Number of output classes
            n_modes: Number of optical modes
            n_photons: Number of photons
            n_layers: Circuit depth (p in the paper)
            computation_space: Computation space type
            scale_type: Input scaling type
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.classifier = MerLinQuantumClassifier(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_modes=n_modes,
            n_photons=n_photons,
            n_layers=n_layers,
            computation_space=computation_space,
            scale_type=scale_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class HybridQuantumClassifier(nn.Module):
    """Hybrid classical-quantum classifier for high-dimensional inputs.

    Uses classical neural network for dimensionality reduction,
    then quantum circuit for classification.
    Suitable for MNIST images.
    """

    def __init__(
        self,
        input_dim: int,
        n_outputs: int,
        hidden_dims: list[int] | None = None,
        n_modes: int = 8,
        n_photons: int = 2,
        n_layers: int = 2,
        computation_space: str = "unbunched",
    ):
        """Initialize hybrid classifier.

        Args:
            input_dim: Dimension of input (e.g., 256 for 16x16 images)
            n_outputs: Number of output classes
            hidden_dims: Hidden layer dimensions for classical encoder
            n_modes: Number of optical modes
            n_photons: Number of photons
            n_layers: Quantum circuit depth
            computation_space: Computation space type
        """
        super().__init__()

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

        # Final projection to match quantum input
        encoder_layers.extend([nn.Linear(prev_dim, n_modes), nn.Tanh()])

        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum classifier
        self.quantum_classifier = MerLinQuantumClassifier(
            n_inputs=n_modes,
            n_outputs=n_outputs,
            n_modes=n_modes,
            n_photons=n_photons,
            n_layers=n_layers,
            computation_space=computation_space,
            scale_type="pi",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim) or (batch_size, H, W)

        Returns:
            Class logits
        """
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Encode
        features = self.encoder(x)

        # Classify
        return self.quantum_classifier(features)


class ClassicalBaseline(nn.Module):
    """Classical neural network baseline for comparison.

    Multi-layer perceptron with similar capacity to quantum model.
    """

    def __init__(
        self,
        input_dim: int,
        n_outputs: int,
        hidden_dims: list[int] | None = None,
        activation: str = "tanh",
    ):
        """Initialize classical baseline.

        Args:
            input_dim: Input dimension
            n_outputs: Number of output classes
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('tanh', 'relu')
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        activation_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), activation_fn])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class ClassicalCNN(nn.Module):
    """Convolutional Neural Network for image classification.

    Used as surrogate model for black-box transfer attacks.
    Architecture similar to LeNet for MNIST.
    """

    def __init__(
        self,
        input_channels: int = 1,
        image_size: int = 16,
        n_outputs: int = 2,
        n_filters: list[int] | None = None,
    ):
        """Initialize CNN.

        Args:
            input_channels: Number of input channels
            image_size: Input image size (assumes square)
            n_outputs: Number of output classes
            n_filters: Number of filters in each conv layer
        """
        super().__init__()

        if n_filters is None:
            n_filters = [16, 32]

        self.image_size = image_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, n_filters[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters[0], n_filters[1], kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Calculate flattened size after convolutions
        # After 2 pooling layers: size = image_size / 4
        flat_size = n_filters[1] * (image_size // 4) * (image_size // 4)

        # Fully connected layers
        self.fc1 = nn.Linear(flat_size, 64)
        self.fc2 = nn.Linear(64, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape if flattened
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 1, self.image_size, self.image_size)

        # Conv layers
        x = self.pool(self.relu(self.conv1(x)))  # -> (batch, 16, 8, 8)
        x = self.pool(self.relu(self.conv2(x)))  # -> (batch, 32, 4, 4)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ClassicalFNN(nn.Module):
    """Feedforward Neural Network for classification.

    Used as surrogate model for black-box transfer attacks.
    """

    def __init__(
        self, input_dim: int, n_outputs: int, hidden_dims: list[int] | None = None
    ):
        """Initialize FNN.

        Args:
            input_dim: Input dimension
            n_outputs: Number of output classes
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class IsingQuantumClassifier(nn.Module):
    """Quantum classifier for Ising model ground states.

    Takes quantum state probability distributions as input
    (already quantum data, no encoding needed).
    """

    def __init__(
        self,
        state_dim: int,
        n_outputs: int = 2,
        n_modes: int = 8,
        n_photons: int = 2,
        n_layers: int = 2,
        computation_space: str = "unbunched",
    ):
        """Initialize Ising classifier.

        Args:
            state_dim: Dimension of quantum state (2^n_spins)
            n_outputs: Number of output classes
            n_modes: Number of optical modes
            n_photons: Number of photons
            n_layers: Circuit depth
            computation_space: Computation space type
        """
        super().__init__()

        self.state_dim = state_dim
        self.n_outputs = n_outputs

        # Dimensionality reduction (state_dim can be large: 2^8 = 256)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, n_modes), nn.Tanh()
        )

        # Quantum classifier
        self.quantum_classifier = MerLinQuantumClassifier(
            n_inputs=n_modes,
            n_outputs=n_outputs,
            n_modes=n_modes,
            n_photons=n_photons,
            n_layers=n_layers,
            computation_space=computation_space,
            scale_type="pi",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.quantum_classifier(features)


def create_model(config: dict[str, Any]) -> nn.Module:
    """Factory function to create models from config.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model
    """
    model_type = config.get("type", "hybrid_quantum")

    # =========================================================================
    # Photonic (MerLin) Models
    # =========================================================================

    # ANGLE ENCODING: Data encoded into phase shifter rotation angles
    if model_type in ("quantum", "angle_quantum", "angle_photonic"):
        return QuantumClassifier(
            n_inputs=config.get("n_inputs", 2),
            n_outputs=config.get("n_outputs", 2),
            n_modes=config.get("n_modes", 4),
            n_photons=config.get("n_photons", 2),
            n_layers=config.get("n_layers", 2),
            computation_space=config.get("computation_space", "unbunched"),
            scale_type=config.get("scale_type", "learned"),
        )

    elif model_type in ("hybrid_quantum", "hybrid_photonic", "hybrid_angle"):
        # Hybrid with angle encoding: Classical NN + Angle-encoded quantum
        return HybridQuantumClassifier(
            input_dim=config.get("input_dim", 256),
            n_outputs=config.get("n_outputs", 2),
            hidden_dims=config.get("hidden_dims"),
            n_modes=config.get("n_modes", 8),
            n_photons=config.get("n_photons", 2),
            n_layers=config.get("n_layers", 2),
            computation_space=config.get("computation_space", "unbunched"),
        )

    # AMPLITUDE ENCODING: Data encoded into Fock state amplitudes
    elif model_type in ("amplitude_quantum", "amplitude_photonic"):
        return MerLinAmplitudeClassifier(
            input_dim=config.get("input_dim", 256),
            n_outputs=config.get("n_outputs", 2),
            n_modes=config.get("n_modes", 4),
            n_photons=config.get("n_photons", 3),
            n_layers=config.get("n_layers", 2),
            hidden_dims=config.get("hidden_dims"),
            computation_space=config.get("computation_space", "unbunched"),
        )

    elif model_type == "ising_quantum":
        return IsingQuantumClassifier(
            state_dim=config.get("state_dim", 256),
            n_outputs=config.get("n_outputs", 2),
            n_modes=config.get("n_modes", 8),
            n_photons=config.get("n_photons", 2),
            n_layers=config.get("n_layers", 2),
            computation_space=config.get("computation_space", "unbunched"),
        )

    # =========================================================================
    # Classical Baselines
    # =========================================================================

    elif model_type == "classical":
        return ClassicalBaseline(
            input_dim=config.get("input_dim", 256),
            n_outputs=config.get("n_outputs", 2),
            hidden_dims=config.get("hidden_dims"),
            activation=config.get("activation", "tanh"),
        )

    elif model_type == "cnn":
        return ClassicalCNN(
            input_channels=config.get("input_channels", 1),
            image_size=config.get("image_size", 16),
            n_outputs=config.get("n_outputs", 2),
            n_filters=config.get("n_filters"),
        )

    elif model_type == "fnn":
        return ClassicalFNN(
            input_dim=config.get("input_dim", 256),
            n_outputs=config.get("n_outputs", 2),
            hidden_dims=config.get("hidden_dims"),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
