"""
Dataset Utilities
=================

Data loading and preprocessing for the quantum adversarial learning experiments:
- MNIST handwritten digits (binary and multi-class)
- Transverse field Ising model ground states
- Amplitude encoding utilities
"""

import logging
from typing import Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


def amplitude_encode(data: torch.Tensor) -> torch.Tensor:
    """Normalize data for amplitude encoding.

    Converts classical data vector to normalized amplitudes suitable
    for encoding into quantum state amplitudes.

    Args:
        data: Input tensor of shape (batch_size, features)

    Returns:
        Normalized tensor where each sample has unit L2 norm
    """
    # Ensure positive values (shift if needed)
    data = data - data.min(dim=-1, keepdim=True)[0]

    # Normalize to unit L2 norm
    norms = torch.norm(data, p=2, dim=-1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)  # Avoid division by zero

    return data / norms


class MNISTBinaryDataset(Dataset):
    """Binary MNIST dataset for two-class classification.

    Filters MNIST to only include two specified digit classes,
    as done in the paper (e.g., digits 1 and 9).
    """

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        digits: tuple[int, int] = (1, 9),
        image_size: int = 16,
        normalize: bool = True,
    ):
        """Initialize binary MNIST dataset.

        Args:
            root: Root directory for data
            train: If True, use training split
            download: Download if not present
            digits: Tuple of two digit classes to include
            image_size: Size to resize images (paper uses 16x16)
            normalize: Whether to normalize for amplitude encoding
        """
        self.digits = digits
        self.image_size = image_size
        self.normalize = normalize

        # Define transforms
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(transform_list)

        # Load full MNIST
        full_dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=download, transform=transform
        )

        # Filter to binary classes
        targets = full_dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.clone().detach()
        else:
            targets = torch.tensor(targets)
        mask = (targets == digits[0]) | (targets == digits[1])
        indices = torch.where(mask)[0].tolist()

        self.dataset = Subset(full_dataset, indices)

        # Map original labels to binary (0, 1)
        self.label_map = {digits[0]: 0, digits[1]: 1}

        logger.info(
            f"MNIST binary: digit {digits[0]} vs {digits[1]}, "
            f"{len(self.dataset)} images"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, orig_label = self.dataset[idx]

        # Flatten image to 1D vector
        image = image.view(-1)

        # Normalize for amplitude encoding
        if self.normalize:
            image = amplitude_encode(image.unsqueeze(0)).squeeze(0)

        # Map label
        label = self.label_map[orig_label]

        return image, label


class MNISTMulticlassDataset(Dataset):
    """Multi-class MNIST dataset.

    Filters MNIST to include specified digit classes,
    as done in the paper (e.g., digits 1, 3, 7, 9 for 4-class).
    """

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        digits: list[int] | None = None,
        image_size: int = 16,
        normalize: bool = True,
    ):
        """Initialize multiclass MNIST dataset.

        Args:
            root: Root directory for data
            train: If True, use training split
            download: Download if not present
            digits: List of digit classes to include
            image_size: Size to resize images (paper uses 16x16)
            normalize: Whether to normalize for amplitude encoding
        """
        if digits is None:
            digits = [1, 3, 7, 9]

        self.digits = digits
        self.image_size = image_size
        self.normalize = normalize
        self.n_classes = len(digits)

        # Define transforms
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(transform_list)

        # Load full MNIST
        full_dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=download, transform=transform
        )

        # Filter to selected classes
        targets = full_dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.clone().detach()
        else:
            targets = torch.tensor(targets)

        mask = torch.zeros_like(targets, dtype=torch.bool)
        for digit in digits:
            mask |= targets == digit
        indices = torch.where(mask)[0].tolist()

        self.dataset = Subset(full_dataset, indices)

        # Map original labels to sequential (0, 1, 2, ...)
        self.label_map = {digit: i for i, digit in enumerate(digits)}

        logger.info(
            f"MNIST multiclass: digits {digits}, "
            f"{len(self.dataset)} images, {self.n_classes} classes"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, orig_label = self.dataset[idx]

        # Flatten image to 1D vector
        image = image.view(-1)

        # Normalize for amplitude encoding
        if self.normalize:
            image = amplitude_encode(image.unsqueeze(0)).squeeze(0)

        # Map label
        label = self.label_map[orig_label]

        return image, label


class IsingDataset(Dataset):
    """Transverse field Ising model dataset.

    Generates ground states of the 1D transverse field Ising model:
        H = -Σ σ_i^z σ_{i+1}^z - J_x Σ σ_i^x

    The phase transition occurs at J_x = 1:
    - J_x < 1: Ordered phase (ferromagnetic)
    - J_x > 1: Disordered phase (paramagnetic)

    From: Section III.C "Quantum adversarial learning quantum phase transitions"
    """

    def __init__(
        self,
        n_spins: int = 8,
        n_samples: int = 1000,
        jx_range: tuple[float, float] = (0.0, 2.0),
        seed: Optional[int] = None,
    ):
        """Initialize Ising dataset.

        Args:
            n_spins: Number of spins (L in the paper)
            n_samples: Number of samples to generate
            jx_range: Range of transverse field strength
            seed: Random seed for reproducibility
        """
        self.n_spins = n_spins
        self.n_samples = n_samples
        self.jx_range = jx_range
        self.critical_point = 1.0  # Phase transition at J_x = 1

        if seed is not None:
            np.random.seed(seed)

        # Generate samples
        self.states = []
        self.labels = []
        self.jx_values = []

        for _ in range(n_samples):
            # Random J_x value
            jx = np.random.uniform(jx_range[0], jx_range[1])
            self.jx_values.append(jx)

            # Compute ground state
            state = self._compute_ground_state(jx)
            self.states.append(state)

            # Label: 0 = ordered (J_x < 1), 1 = disordered (J_x > 1)
            label = 0 if jx < self.critical_point else 1
            self.labels.append(label)

        self.states = np.array(self.states, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        logger.info(
            f"Ising dataset: L={n_spins}, {n_samples} samples, "
            f"J_x in [{jx_range[0]}, {jx_range[1]}]"
        )

    def _compute_ground_state(self, jx: float) -> np.ndarray:
        """Compute ground state of the Ising Hamiltonian.

        Args:
            jx: Transverse field strength

        Returns:
            Ground state wavefunction as numpy array
        """
        # Build Hamiltonian
        H = self._build_hamiltonian(jx)

        # Diagonalize
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # Ground state is the eigenvector with smallest eigenvalue
        ground_state = eigenvectors[:, 0]

        # Take absolute values of amplitudes (for real representation)
        # and normalize
        amplitudes = np.abs(ground_state) ** 2
        amplitudes = amplitudes / np.sum(amplitudes)

        return amplitudes.astype(np.float32)

    def _build_hamiltonian(self, jx: float) -> np.ndarray:
        """Build the Ising Hamiltonian matrix.

        Args:
            jx: Transverse field strength

        Returns:
            Hamiltonian matrix
        """
        L = self.n_spins
        dim = 2**L

        # Pauli matrices
        sx = np.array([[0, 1], [1, 0]], dtype=np.float64)
        sz = np.array([[1, 0], [0, -1]], dtype=np.float64)
        identity_2x2 = np.eye(2, dtype=np.float64)

        H = np.zeros((dim, dim), dtype=np.float64)

        # ZZ interactions: -Σ σ_i^z σ_{i+1}^z
        for i in range(L - 1):
            # Build σ_i^z ⊗ σ_{i+1}^z
            op = np.eye(1, dtype=np.float64)
            for j in range(L):
                if j == i or j == i + 1:
                    op = np.kron(op, sz)
                else:
                    op = np.kron(op, identity_2x2)
            H -= op

        # Transverse field: -J_x Σ σ_i^x
        for i in range(L):
            op = np.eye(1, dtype=np.float64)
            for j in range(L):
                if j == i:
                    op = np.kron(op, sx)
                else:
                    op = np.kron(op, identity_2x2)
            H -= jx * op

        return H

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        state = torch.tensor(self.states[idx])
        label = self.labels[idx]
        return state, label


class SpiralDataset(Dataset):
    """2D Spiral dataset for simple classification tests."""

    def __init__(
        self, n_samples: int = 1000, noise: float = 0.0, seed: Optional[int] = None
    ):
        """Initialize spiral dataset.

        Args:
            n_samples: Number of samples
            noise: Gaussian noise std
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        n_per_class = n_samples // 2

        # Generate spiral parameters
        theta = np.sqrt(np.random.rand(n_per_class)) * 2 * np.pi

        # Class 0: Spiral 1
        r0 = 2 * theta + np.pi
        x0 = r0 * np.cos(theta)
        y0 = r0 * np.sin(theta)

        # Class 1: Spiral 2 (rotated)
        r1 = -2 * theta - np.pi
        x1 = r1 * np.cos(theta)
        y1 = r1 * np.sin(theta)

        # Combine
        X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

        # Add noise
        if noise > 0:
            X += np.random.randn(*X.shape) * noise

        # Normalize to [-1, 1]
        X = X / np.max(np.abs(X))

        # Shuffle
        idx = np.random.permutation(len(y))
        self.X = X[idx].astype(np.float32)
        self.y = y[idx].astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class QAHDataset(Dataset):
    """Quantum Anomalous Hall (QAH) effect dataset.

    Generates time-of-flight images for the 2D square-lattice model
    exhibiting the quantum anomalous Hall effect:

    H = J_SO^x Σ [(c†_r↑ c_{r+x̂}↓ - c†_r↑ c_{r-x̂}↓) + h.c.]
      + i J_SO^y Σ [(c†_r↑ c_{r+ŷ}↓ - c†_r↑ c_{r-ŷ}↓) + h.c.]
      - t Σ (c†_r↑ c_s↑ - c†_r↓ c_s↓)
      + μ Σ (c†_r↑ c_r↑ - c†_r↓ c_r↓)

    Topological invariant: First Chern number C₁
    - C₁ = -sign(μ) when 0 < |μ| < 4t
    - C₁ = 0 otherwise

    From: Section III.C "Quantum adversarial learning topological phases"
    """

    def __init__(
        self,
        lattice_size: int = 10,
        n_samples: int = 1000,
        t: float = 1.0,
        j_so_range: tuple[float, float] = (0.5, 2.0),
        mu_range: tuple[float, float] = (-6.0, 6.0),
        momentum_resolution: int = 20,
        seed: Optional[int] = None,
    ):
        """Initialize QAH dataset.

        Args:
            lattice_size: Size of the square lattice (L x L)
            n_samples: Number of samples to generate
            t: Hopping amplitude
            j_so_range: Range for spin-orbit coupling strength
            mu_range: Range for Zeeman field (controls topology)
            momentum_resolution: Resolution of momentum-space grid
            seed: Random seed for reproducibility
        """
        self.lattice_size = lattice_size
        self.n_samples = n_samples
        self.t = t
        self.momentum_resolution = momentum_resolution

        if seed is not None:
            np.random.seed(seed)

        # Generate samples
        self.tof_images = []
        self.labels = []
        self.chern_numbers = []
        self.parameters = []

        for _ in range(n_samples):
            # Random parameters
            j_so = np.random.uniform(j_so_range[0], j_so_range[1])
            mu = np.random.uniform(mu_range[0], mu_range[1])

            # Compute Chern number (analytical result)
            if 0 < np.abs(mu) < 4 * t:
                chern = -np.sign(mu)
            else:
                chern = 0

            # Label: 0 for trivial (C₁=0), 1 for topological (C₁≠0)
            label = 0 if chern == 0 else 1

            # Generate time-of-flight image
            tof_image = self._compute_tof_image(j_so, mu, t)

            self.tof_images.append(tof_image)
            self.labels.append(label)
            self.chern_numbers.append(chern)
            self.parameters.append((j_so, mu))

        self.tof_images = np.array(self.tof_images, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        # Log class distribution
        n_trivial = (np.array(self.labels) == 0).sum()
        n_topological = (np.array(self.labels) == 1).sum()
        logger.info(
            f"QAH dataset: {n_samples} samples, "
            f"trivial: {n_trivial}, topological: {n_topological}"
        )

    def _compute_tof_image(self, j_so: float, mu: float, t: float) -> np.ndarray:
        """Compute time-of-flight image (momentum distribution).

        The ToF image shows the momentum-space density distribution
        of the lower band, which is measured in cold atom experiments.

        Args:
            j_so: Spin-orbit coupling strength
            mu: Zeeman field strength
            t: Hopping amplitude

        Returns:
            2D momentum distribution image
        """
        n_k = self.momentum_resolution
        kx_vals = np.linspace(-np.pi, np.pi, n_k)
        ky_vals = np.linspace(-np.pi, np.pi, n_k)

        density = np.zeros((n_k, n_k))

        for i, kx in enumerate(kx_vals):
            for j, ky in enumerate(ky_vals):
                # Build 2x2 Bloch Hamiltonian H(k)
                # H(k) = d(k) · σ where σ are Pauli matrices
                dx = 2 * j_so * np.sin(kx)  # Spin-orbit x
                dy = 2 * j_so * np.sin(ky)  # Spin-orbit y
                dz = mu - 2 * t * (np.cos(kx) + np.cos(ky))  # Zeeman + hopping

                # Energy spectrum
                E = np.sqrt(dx**2 + dy**2 + dz**2)

                if E > 1e-10:
                    # Lower band eigenvector |u_-(k)⟩
                    # For a 2x2 Hamiltonian d·σ, the lower eigenstate is:
                    # |u_-⟩ = (sin(θ/2), -cos(θ/2)e^{iφ})
                    # where θ = arccos(dz/|d|), φ = arctan2(dy, dx)
                    theta = np.arccos(dz / E)
                    # phi = np.arctan2(dy, dx)  # Not used in density calculation

                    # Momentum distribution (|⟨↑|u_-⟩|² + |⟨↓|u_-⟩|²)
                    # which is 1 for normalized state, but we weight by
                    # the spin-up probability |⟨↑|u_-⟩|² = sin²(θ/2)
                    density[j, i] = np.sin(theta / 2) ** 2
                else:
                    density[j, i] = 0.5

        # Normalize to [0, 1]
        if density.max() > density.min():
            density = (density - density.min()) / (density.max() - density.min())

        return density

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Flatten image for input to classifier
        image = torch.tensor(self.tof_images[idx]).flatten()
        label = self.labels[idx]
        return image, label

    def get_image_2d(self, idx: int) -> np.ndarray:
        """Get the 2D ToF image for visualization."""
        return self.tof_images[idx]


class TopologicalPhaseDataset(Dataset):
    """Simplified topological phase dataset.

    Alternative implementation using a simpler 1D SSH-like model
    for faster computation while maintaining the topological classification task.
    """

    def __init__(
        self,
        n_sites: int = 20,
        n_samples: int = 1000,
        parameter_range: tuple[float, float] = (0.1, 2.0),
        seed: Optional[int] = None,
    ):
        """Initialize topological phase dataset.

        Args:
            n_sites: Number of sites in the 1D chain
            n_samples: Number of samples
            parameter_range: Range for hopping ratio parameter
            seed: Random seed
        """
        self.n_sites = n_sites
        self.n_samples = n_samples

        if seed is not None:
            np.random.seed(seed)

        self.features = []
        self.labels = []

        for _ in range(n_samples):
            # Random hopping ratio (controls topology)
            v = np.random.uniform(parameter_range[0], parameter_range[1])
            w = 1.0  # Fixed

            # Topological: v < w, Trivial: v > w
            label = 1 if v < w else 0

            # Compute features (e.g., correlation functions or band structure)
            features = self._compute_features(v, w)

            self.features.append(features)
            self.labels.append(label)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def _compute_features(self, v: float, w: float) -> np.ndarray:
        """Compute observable features from SSH model.

        Args:
            v: Intra-cell hopping
            w: Inter-cell hopping

        Returns:
            Feature vector
        """
        n_k = 50
        k_vals = np.linspace(-np.pi, np.pi, n_k)

        # Band structure features
        energies = []
        for k in k_vals:
            # SSH Hamiltonian eigenvalues: E = ±|v + w*e^{ik}|
            h_k = v + w * np.exp(1j * k)
            E = np.abs(h_k)
            energies.append(E)

        energies = np.array(energies)

        # Features: energy at different k points
        return energies.astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.tensor(self.features[idx]), self.labels[idx]


def create_dataloaders(
    dataset_name: str, config: dict, seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders from config.

    Args:
        dataset_name: Name of dataset ('mnist_binary', 'mnist_multi', 'ising', 'spiral')
        config: Dataset configuration
        seed: Random seed

    Returns:
        train_loader, test_loader
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset_name == "mnist_binary":
        train_dataset = MNISTBinaryDataset(
            root=config.get("root", "data"),
            train=True,
            download=config.get("download", True),
            digits=tuple(config.get("digits", [1, 9])),
            image_size=config.get("image_size", 16),
            normalize=config.get("normalize", True),
        )
        test_dataset = MNISTBinaryDataset(
            root=config.get("root", "data"),
            train=False,
            download=config.get("download", True),
            digits=tuple(config.get("digits", [1, 9])),
            image_size=config.get("image_size", 16),
            normalize=config.get("normalize", True),
        )
        batch_size = config.get("batch_size", 256)

    elif dataset_name == "mnist_multi":
        train_dataset = MNISTMulticlassDataset(
            root=config.get("root", "data"),
            train=True,
            download=config.get("download", True),
            digits=config.get("digits", [1, 3, 7, 9]),
            image_size=config.get("image_size", 16),
            normalize=config.get("normalize", True),
        )
        test_dataset = MNISTMulticlassDataset(
            root=config.get("root", "data"),
            train=False,
            download=config.get("download", True),
            digits=config.get("digits", [1, 3, 7, 9]),
            image_size=config.get("image_size", 16),
            normalize=config.get("normalize", True),
        )
        batch_size = config.get("batch_size", 512)

    elif dataset_name == "ising":
        # For Ising, we generate train/test from the same distribution
        n_total = config.get("n_samples", 2000)
        n_train = int(n_total * 0.8)

        full_dataset = IsingDataset(
            n_spins=config.get("n_spins", 8),
            n_samples=n_total,
            jx_range=tuple(config.get("jx_range", [0.0, 2.0])),
            seed=seed,
        )

        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_total - n_train]
        )
        batch_size = config.get("batch_size", 64)

    elif dataset_name == "spiral":
        n_total = config.get("n_samples", 2200)
        n_train = config.get("n_train", 2000)

        full_dataset = SpiralDataset(
            n_samples=n_total, noise=config.get("noise", 0.0), seed=seed
        )

        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_total - n_train]
        )
        batch_size = config.get("batch_size", 10)

    elif dataset_name == "qah" or dataset_name == "topological":
        n_total = config.get("n_samples", 2000)
        n_train = int(n_total * 0.8)

        full_dataset = QAHDataset(
            lattice_size=config.get("lattice_size", 10),
            n_samples=n_total,
            t=config.get("t", 1.0),
            j_so_range=tuple(config.get("j_so_range", [0.5, 2.0])),
            mu_range=tuple(config.get("mu_range", [-6.0, 6.0])),
            momentum_resolution=config.get("momentum_resolution", 20),
            seed=seed,
        )

        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_total - n_train]
        )
        batch_size = config.get("batch_size", 64)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Subsample for faster smoke tests if requested
    train_samples = config.get("train_samples")
    test_samples = config.get("test_samples")

    if train_samples is not None and train_samples < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:train_samples].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    if test_samples is not None and test_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:test_samples].tolist()
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader
