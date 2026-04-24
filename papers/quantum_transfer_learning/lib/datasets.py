"""
Dataset Utilities
=================

Data loading and preprocessing for the transfer learning experiments:
- SpiralDataset: 2D concentric spirals (Example 1)
- HymenopteraDataset: Ants vs Bees (Example 2)
- CIFAR10Binary: Binary subsets of CIFAR-10 (Example 3)
"""

import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


def generate_spiral_data(
    n_samples: int,
    noise: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2D spiral dataset.
    
    Creates two interleaved spirals for binary classification,
    matching the dataset in Figure 2 of the paper.
    
    Args:
        n_samples: Total number of samples (split evenly between classes)
        noise: Standard deviation of Gaussian noise
        seed: Random seed
        
    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
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

    # Class 1: Spiral 2 (rotated by pi)
    r1 = -2 * theta - np.pi
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)

    # Combine
    X = np.vstack([
        np.column_stack([x0, y0]),
        np.column_stack([x1, y1])
    ])
    y = np.hstack([
        np.zeros(n_per_class),
        np.ones(n_per_class)
    ])

    # Add noise
    if noise > 0:
        X += np.random.randn(*X.shape) * noise

    # Normalize to [-1, 1]
    X = X / np.max(np.abs(X))

    # Shuffle
    idx = np.random.permutation(len(y))
    X = X[idx]
    y = y[idx]

    return X.astype(np.float32), y.astype(np.int64)


class SpiralDataset(Dataset):
    """PyTorch Dataset for 2D spiral data."""

    def __init__(
        self,
        n_samples: int = 2200,
        noise: float = 0.0,
        seed: Optional[int] = None
    ):
        """Initialize spiral dataset.
        
        Args:
            n_samples: Number of samples
            noise: Gaussian noise std
            seed: Random seed
        """
        self.X, self.y = generate_spiral_data(n_samples, noise, seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class HymenopteraDataset(Dataset):
    """Hymenoptera dataset (ants vs bees) for transfer learning.
    
    Downloads and prepares the dataset used in Example 2 of the paper.
    """

    DOWNLOAD_URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        image_size: int = 224
    ):
        """Initialize Hymenoptera dataset.
        
        Args:
            root: Root directory for data
            train: If True, use training split; else test split
            download: Download data if not present
            image_size: Size to resize images
        """
        self.root = Path(root)
        self.train = train
        self.image_size = image_size

        # Download if needed
        data_dir = self.root / "hymenoptera_data"
        if download and not data_dir.exists():
            self._download()

        # Set up paths
        split = "train" if train else "val"
        self.data_dir = data_dir / split

        # Define transforms (matching ResNet preprocessing)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Load image paths and labels
        self.classes = ["ants", "bees"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

        logger.info(f"Loaded {len(self.samples)} images for {'train' if train else 'test'}")

    def _download(self):
        """Download and extract the dataset."""
        self.root.mkdir(parents=True, exist_ok=True)

        zip_path = self.root / "hymenoptera_data.zip"

        logger.info("Downloading Hymenoptera dataset...")
        urllib.request.urlretrieve(self.DOWNLOAD_URL, zip_path)

        logger.info("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)

        zip_path.unlink()
        logger.info("Download complete!")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


class CIFAR10Binary(Dataset):
    """Binary subset of CIFAR-10 for transfer learning experiments.
    
    Filters CIFAR-10 to only include two specified classes,
    as done in Example 3 of the paper.
    """

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        download: bool = True,
        classes: List[int] = [3, 5],  # Default: cat, dog
        image_size: int = 224
    ):
        """Initialize CIFAR-10 binary dataset.
        
        Args:
            root: Root directory for data
            train: If True, use training split
            download: Download if not present
            classes: Two class indices to include
            image_size: Size to resize images
        """
        self.classes = classes
        self.image_size = image_size

        # CIFAR class names for reference
        self.cifar_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        # Define transforms
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Load full CIFAR-10
        full_dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=self.transform
        )

        # Filter to binary classes
        targets = np.array(full_dataset.targets)
        mask = np.isin(targets, classes)
        indices = np.where(mask)[0]

        self.dataset = Subset(full_dataset, indices)

        # Map original class labels to binary (0, 1)
        self.label_map = {classes[0]: 0, classes[1]: 1}

        logger.info(
            f"CIFAR-10 binary: {self.cifar_classes[classes[0]]} vs {self.cifar_classes[classes[1]]}, "
            f"{len(self.dataset)} images"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, orig_label = self.dataset[idx]
        binary_label = self.label_map[orig_label]
        return image, binary_label


def create_dataloaders(
    dataset_name: str,
    config: dict,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders from config.
    
    Args:
        dataset_name: Name of dataset ('spiral', 'hymenoptera', 'cifar10')
        config: Dataset configuration
        seed: Random seed
        
    Returns:
        train_loader, test_loader
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset_name == "spiral":
        dataset = SpiralDataset(
            n_samples=config.get("n_samples", 2200),
            noise=config.get("noise", 0.0),
            seed=seed
        )

        n_train = config.get("n_train", 2000)
        n_test = len(dataset) - n_train

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_test]
        )

        batch_size = config.get("batch_size", 10)

    elif dataset_name == "hymenoptera":
        train_dataset = HymenopteraDataset(
            root=config.get("root", "data"),
            train=True,
            download=config.get("download", True),
            image_size=config.get("image_size", 224)
        )
        test_dataset = HymenopteraDataset(
            root=config.get("root", "data"),
            train=False,
            download=config.get("download", True),
            image_size=config.get("image_size", 224)
        )
        batch_size = config.get("batch_size", 4)

    elif dataset_name == "cifar10":
        train_dataset = CIFAR10Binary(
            root=config.get("root", "data"),
            train=True,
            download=config.get("download", True),
            classes=config.get("classes", [3, 5]),
            image_size=config.get("image_size", 224)
        )
        test_dataset = CIFAR10Binary(
            root=config.get("root", "data"),
            train=False,
            download=config.get("download", True),
            classes=config.get("classes", [3, 5]),
            image_size=config.get("image_size", 224)
        )
        batch_size = config.get("batch_size", 8)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues with quantum
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader
