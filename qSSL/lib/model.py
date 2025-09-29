# Model definitions for Quantum Self-Supervised Learning (qSSL)
# Supports both quantum (MerLin/Qiskit) and classical representation networks
# Based on "Quantum Self-Supervised Learning" by Jaderberg et al. (2022)

import math

# Lazy-import quantum libs to avoid hard dependency during classical tests
import torch
import torch.nn as nn
import torchvision

# Import QNet from relocated package within lib
from .qnn.qnet import QNet
from .training_utils import InfoNCELoss


def create_quantum_circuit(modes=10, feature_size=10):
    """
    Create a photonic quantum circuit for feature encoding and processing.
    """
    # Local import to avoid requiring perceval unless MerLin backend is used
    import perceval as pcvl  # type: ignore

    # First trainable interferometer - processes input photons
    pre_circuit = pcvl.GenericInterferometer(
        modes,
        lambda i: (
            pcvl.BS()
            .add(0, pcvl.PS(pcvl.P(f"phase_train_1_{i}")))
            .add(0, pcvl.BS())
            .add(0, pcvl.PS(pcvl.P(f"phase_train_2_{i}")))
        ),
    )
    # Data encoding layer - embed classical features into quantum states
    var = pcvl.Circuit(modes)
    for k in range(0, feature_size):
        var.add(k % modes, pcvl.PS(pcvl.P(f"feature-{k}")))

    # Second trainable interferometer - processes encoded features
    post_circuit = pcvl.GenericInterferometer(
        modes,
        lambda i: (
            pcvl.BS()
            .add(0, pcvl.PS(pcvl.P(f"phase_train_3_{i}")))
            .add(0, pcvl.BS())
            .add(0, pcvl.PS(pcvl.P(f"phase_train_4_{i}")))
        ),
    )

    circuit = pcvl.Circuit(modes)
    circuit.add(0, pre_circuit, merge=True)
    circuit.add(0, var, merge=True)
    circuit.add(0, post_circuit, merge=True)

    return circuit


def initialize_resnet_kaiming(model):
    """
    Apply Kaiming (He) initialization to ResNet model components.
    This initialization is specifically designed for ReLU activations and helps
    prevent vanishing/exploding gradients during training.

    Args:
        model: PyTorch model to initialize
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Kaiming normal initialization for convolutional layers
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            # Initialize normalization layer weights to 1, biases to 0
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Kaiming normal initialization for linear layers
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class QSSL(nn.Module):
    """
    Quantum Self-Supervised Learning model that supports both quantum and classical
    representation networks for contrastive learning.

    Architecture:
    1. ResNet18 backbone for feature extraction
    2. Compression layer to reduce feature dimension
    3. Representation network (quantum or classical)
    4. Projection head for contrastive loss computation
    """

    def __init__(
        self,
        args,
    ):
        super().__init__()
        # Determine which framework is being used for logging
        framework_used = "Quantum (MerLin)" if args.merlin else "Classical ResNet18"
        framework_used = "Quantum (Qiskit)" if args.qiskit else framework_used
        print(f"\n Defining the SSL model with \n -{framework_used} \n - ")
        # ========== Backbone Network ==========
        self.width = args.width  # Feature dimension after compression

        # ResNet18 backbone for feature extraction from images
        # zero_init_residual=True improves training stability
        self.backbone = torchvision.models.resnet18(
            pretrained=False, zero_init_residual=True
        )
        # Apply Kaiming initialization as specified in Jaderberg et al.
        initialize_resnet_kaiming(self.backbone)

        # Remove the final classification layer and replace with identity
        backbone_features = (
            self.backbone.fc.in_features
        )  # Get ResNet18 output dimension (512)
        self.backbone.fc = nn.Identity()

        # Compression layer: map ResNet features to quantum-compatible dimension
        self.comp = nn.Linear(backbone_features, self.width)

        # ========== Representation Network Configuration ==========
        self.merlin = args.merlin  # Use MerLin quantum framework
        self.qiskit = args.qiskit  # Use Qiskit quantum framework
        self.batch_norm = args.batch_norm  # Apply batch norm after compression
        self.bn = nn.BatchNorm2d(self.width)  # Batch normalization layer

        # ========== Quantum Representation Network (MerLin) ==========
        if self.merlin:
            print("\n -> Building the quantum representation network with MerLin")
            # Local import to avoid hard dependency unless used
            from merlin import OutputMappingStrategy, QuantumLayer  # type: ignore

            self.modes = args.modes  # Number of photonic modes
            self.no_bunching = args.no_bunching  # Photon bunching configuration

            # Create photonic quantum circuit for feature processing
            self.circuit = create_quantum_circuit(
                modes=self.modes, feature_size=self.width
            )

            # Define initial photon state: alternating pattern of 0s and 1s
            input_state = [(i + 1) % 2 for i in range(args.modes)]
            print(
                f"Initial photon state: {input_state}, no_bunching = {self.no_bunching}"
            )

            # Create quantum layer using MerLin framework
            self.representation_network = QuantumLayer(
                input_size=self.width,
                output_size=None,
                circuit=self.circuit,
                trainable_parameters=[
                    p.name
                    for p in self.circuit.get_parameters()
                    if not p.name.startswith("feature")
                ],
                input_parameters=["feature"],
                input_state=input_state,
                no_bunching=self.no_bunching,
                output_mapping_strategy=OutputMappingStrategy.NONE,
            )

            self.rep_net_output_size = self.representation_network.output_size

        # ========== Quantum Representation Network (Qiskit) ==========
        elif self.qiskit:
            # Implementation based on https://github.com/bjader/QSSL
            print("\n -> Building the quantum representation network with Qiskit")
            self.representation_network = QNet(
                n_qubits=self.width,  # Number of qubits = feature dimension
                encoding=args.encoding,  # Data encoding method (e.g., 'vector')
                ansatz_type=args.q_ansatz,  # Variational ansatz architecture
                layers=args.layers,  # Number of circuit layers
                sweeps_per_layer=args.q_sweeps,  # Ansatz repetitions per layer
                activation_function_type=args.activation,  # Quantum activation function
                shots=args.shots,  # Number of measurement shots
                backend_type=args.q_backend,  # Quantum simulator type
                save_statevectors=args.save_dhs,  # Save quantum states for analysis
            )
            self.rep_net_output_size = self.width  # Output size equals input for Qiskit
        # ========== Classical Representation Network ==========
        else:
            mapping_paper = True  # Use paper's classical baseline architecture
            print("\n -> Building the classical representation network ")

            # Option 1: Parameter-matched classical network (for fair comparison)
            if not mapping_paper:
                # Create classical MLP with similar parameter count to MerLin network
                # This ensures fair comparison between quantum and classical approaches
                # Calculate quantum network parameters for reference
                circuit = create_quantum_circuit(modes=10, feature_size=8)
                nb_trainable_params = len(
                    [
                        p.name
                        for p in circuit.get_parameters()
                        if not p.name.startswith(
                            "feature"
                        )  # Only trainable circuit parameters
                    ]
                )
                # Compute quantum output dimension based on photon statistics
                output_size = (
                    math.comb(10, 5) if args.no_bunching else math.comb(10 + 5 - 1, 5)
                )
                # Total quantum parameters: circuit + projection layer
                total_parameters = nb_trainable_params + output_size * self.width
                # Classical MLP parameters for comparison
                classical_params = (
                    self.width * self.width + self.width
                ) * 3  # Three layers with biases
                # Parameter difference to match
                diff = total_parameters - classical_params
                print(
                    f"--> Parameter difference: {diff} (total quantum params: {total_parameters})"
                )

                # Build classical MLP to match quantum parameter count
                layers = []
                # First two hidden layers with self-connections
                for _i in range(2):
                    layers.append(nn.Linear(self.width, self.width, bias=True))
                    layers.append(nn.LeakyReLU())

                # Additional layer to consume remaining parameters
                catching_output_size = int(diff / self.width - 1)
                layers.append(nn.Linear(self.width, catching_output_size, bias=True))
                layers.append(nn.LeakyReLU())

                self.representation_network = nn.Sequential(*layers)
                print(
                    f" Now in repnet = {sum(p.numel() for p in self.representation_network.parameters())}"
                )
                self.rep_net_output_size = catching_output_size

            # Option 2: Paper's classical baseline architecture
            else:
                # Simple MLP matching the paper's classical baseline
                layers = []
                for _i in range(args.layers):
                    layers.append(nn.Linear(args.width, args.width, bias=True))
                    layers.append(nn.LeakyReLU())
                self.representation_network = nn.Sequential(*layers)
                self.rep_net_output_size = args.width  # Output dimension matches input
        # ========== Contrastive Learning Components ==========
        self.loss_dim = args.loss_dim  # Dimension of contrastive loss space

        # Projection head: maps representations to contrastive loss space
        # This is a key component in contrastive learning (SimCLR, MoCo, etc.)
        self.proj = nn.Sequential(
            nn.Linear(
                self.rep_net_output_size, self.width
            ),  # Project to intermediate dim
            nn.BatchNorm1d(self.width),  # Normalize features
            nn.ReLU(),  # Non-linear activation
            nn.Linear(self.width, self.loss_dim),  # Final projection to loss space
        )

        self.normalize = nn.Sigmoid()  # Normalization function (if needed)
        self.temperature = args.temperature  # Temperature parameter for InfoNCE loss

        # InfoNCE contrastive loss function
        self.criterion = InfoNCELoss(temperature=self.temperature)

    def forward(self, y1, y2):
        """
        Forward pass for contrastive learning with two augmented views.

        Args:
            y1, y2: Two augmented views of the same batch of images

        Returns:
            loss: InfoNCE contrastive loss between the two views
        """
        # ========== Feature Extraction ==========
        # Extract features using ResNet backbone + compression
        x1 = self.comp(self.backbone(y1))
        x2 = self.comp(self.backbone(y2))

        # Optional batch normalization after compression
        if self.batch_norm:
            x1 = self.bn(x1)
            x2 = self.bn(x2)

        # ========== Feature Preprocessing ==========
        # Apply sigmoid activation before quantum/classical representation network
        # This ensures features are in [0,1] range, suitable for quantum encoding
        factor = torch.pi if not self.merlin else 1 / torch.pi
        x1 = torch.sigmoid(x1) * factor
        x2 = torch.sigmoid(x2) * factor
        # Note: Original implementation scales by π for phase encoding
        # print(f"\n ---x1 = {torch.min(x1)} to {torch.max(x1)} with factor = {factor}")
        # ========== Representation Learning ==========
        # Process through quantum or classical representation network
        z1 = self.representation_network(x1)
        z2 = self.representation_network(x2)

        # ========== Contrastive Loss Computation ==========
        # Project representations to contrastive loss space
        z1 = self.proj(z1)
        z2 = self.proj(z2)

        # Compute InfoNCE contrastive loss between the two views
        # The loss encourages similar representations for augmented versions
        # of the same image while pushing apart different images
        loss = self.criterion(z1, z2)

        return loss
