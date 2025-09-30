"""
paper_datasets.py - Unified dataset module for paper reproduction
___________________________________________________________________________________________

This module implements all datasets used in "Experimental data re-uploading
with provable enhanced learning capabilities" (Pérez-Salinas et al., 2025).

All datasets follow the same clean interface and exact paper specifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# Configure matplotlib for publication-quality plots
import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.datasets import make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

plt.rcParams.update(
    {
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "figure.figsize": (10, 6),
    }
)
plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "stix",  # or "cm"
        "font.family": "STIXGeneral",
    }
)


@dataclass
class PaperDataset:
    """Base class for paper reproduction datasets."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    name: str

    def plot_samples(
        self, split: str = "train", figsize: tuple[int, int] = (8, 6)
    ) -> None:
        """Plot dataset samples."""
        X = self.X_train if split == "train" else self.X_test
        y = self.y_train if split == "train" else self.y_test

        plt.figure(figsize=figsize)
        plt.scatter(
            X[y == 0, 0],
            X[y == 0, 1],
            c="red",
            alpha=0.6,
            s=20,
            label="class 0",
            edgecolors="black",
            linewidth=0.5,
        )
        plt.scatter(
            X[y == 1, 0],
            X[y == 1, 1],
            c="blue",
            alpha=0.6,
            s=20,
            label="class 1",
            edgecolors="black",
            linewidth=0.5,
        )
        plt.title(f"{self.name} Dataset ({split} split)")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close()
        return

    @property
    def train(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X_train, self.y_train

    @property
    def test(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X_test, self.y_test


class CirclesDataset(PaperDataset):
    """
    Circles dataset as specified in the paper.
    Paper quote: "The scaling factor between the inner and outer circles is 0.6 with 0.05 Gaussian noise."
    """

    def __init__(
        self,
        n_train: int = 400,
        n_test: int = 100,
        factor: float = 0.6,
        noise: float = 0.05,
    ):
        # Generate training and test data
        X_train, y_train = make_circles(n_samples=n_train, factor=factor, noise=noise)
        X_test, y_test = make_circles(n_samples=n_test, factor=factor, noise=noise)

        super().__init__(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            name="Circles",
        )
        self.factor = factor
        self.noise = noise


class MoonsDataset(PaperDataset):
    """
    Moons dataset as specified in the paper.
    Paper quote: "Two interleaving half circles... with 0.1 Gaussian noise."
    """

    def __init__(self, n_train: int = 400, n_test: int = 100, noise: float = 0.1):
        # Generate training and test data
        X_train, y_train = make_moons(n_samples=n_train, noise=noise)
        X_test, y_test = make_moons(n_samples=n_test, noise=noise)

        super().__init__(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, name="Moons"
        )
        self.noise = noise


class TetrominoDataset(PaperDataset):
    """
    Tetromino dataset as specified in the paper.
    Paper quote: "3×3 pixel figures, representing 'T' and 'L' letters...
    uniform background noise bounded between −0.1 and 0.1"
    """

    # Tetromino definitions
    @dataclass(frozen=True, slots=True)
    class Tetromino:
        """A canonical tetromino piece and all its rotations."""

        name: str
        base: np.ndarray  # (3, 3) binary array
        label: str  # "T" or "L"

        @property
        def rotations(self) -> tuple[np.ndarray, ...]:
            """Return **unique** 90° rotations of *base* (incl. 0°)."""
            rots = tuple(np.rot90(self.base, k) for k in range(4))
            uniq: list[np.ndarray] = []
            for r in rots:  # de‑duplicate
                if not any(np.array_equal(r, u) for u in uniq):
                    uniq.append(r)
            return tuple(uniq)

    # Canonical pieces (3×3 binary grids)
    _T_UP = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=float)
    _T_DOWN = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=float)
    _L_LEFT = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    _L_RIGHT = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=float)

    _SHAPES = (
        Tetromino("T_up", _T_UP, "T"),
        Tetromino("T_down", _T_DOWN, "T"),
        Tetromino("L_left", _L_LEFT, "L"),
        Tetromino("L_right", _L_RIGHT, "L"),
    )

    _LABELS = np.array([t.label for t in _SHAPES])
    _ROTATIONS = [t.rotations for t in _SHAPES]

    def __init__(
        self,
        n_train: int = 400,
        n_test: int = 100,
        noise: float = 0.05,
        invert_prob: float = 0.5,
    ):
        # Generate training and test data
        X_train, y_train_str = self._make_tetromino_dataset(
            n_train, noise=noise, invert_prob=invert_prob
        )
        X_test, y_test_str = self._make_tetromino_dataset(
            n_test, noise=noise, invert_prob=invert_prob
        )

        # Convert string labels to integers
        y_train = np.array([0 if label == "T" else 1 for label in y_train_str])
        y_test = np.array([0 if label == "T" else 1 for label in y_test_str])

        # Flatten 3x3 images to 9D vectors
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        super().__init__(
            X_train=X_train_flat,
            y_train=y_train,
            X_test=X_test_flat,
            y_test=y_test,
            name="Tetromino",
        )

        self.noise = noise
        self.dimension = 9  # 3x3 flattened
        self.invert_prob = invert_prob

    def _make_tetromino_dataset(
        self, n_samples: int, noise: float = 0.0, invert_prob: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a batch of noisy tetromino images."""
        rng = np.random.default_rng()

        # Choose shapes (balanced)
        reps = -(-n_samples // 2)  # ceil division
        idx_pool = np.tile(np.arange(len(self._SHAPES)), reps)[:n_samples]
        rng.shuffle(idx_pool)

        X = np.empty((n_samples, 3, 3), dtype=np.float32)
        y = self._LABELS[idx_pool]

        # Generate samples
        for i, idx in enumerate(idx_pool):
            arr = self._ROTATIONS[idx][rng.integers(len(self._ROTATIONS[idx]))].copy()

            # Polarity inversion (dark ↔ light)
            if rng.random() < invert_prob:
                arr = 1.0 - arr

            # Asymmetric greyscale noise
            if noise:
                eps = rng.normal(0.0, noise, size=arr.shape)
                direction = 1.0 - 2.0 * arr  # +1 for 0, −1 for 1
                arr += direction * np.abs(eps)
                np.clip(arr, 0.0, 1.0, out=arr)

            X[i] = arr

        return X, y

    def plot_samples(
        self,
        split: str = "train",
        n_samples: int = 9,
        figsize: tuple[int, int] = (8, 8),
    ) -> None:
        """Plot tetromino samples as 3x3 grids."""
        X = self.X_train if split == "train" else self.X_test
        y = self.y_train if split == "train" else self.y_test

        # Reshape back to 3x3 for visualization
        X_reshaped = X.reshape(-1, 3, 3)

        # Select samples to show
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)

        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.ravel()

        for i, idx in enumerate(indices):
            if i < len(axes):
                axes[i].imshow(X_reshaped[idx], cmap="gray", vmin=0, vmax=1)
                label_name = "T" if y[idx] == 0 else "L"
                axes[i].set_title(f"{label_name}", fontsize=12)
                axes[i].axis("off")

        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis("off")

        plt.suptitle(f"Tetromino Dataset ({split} split)", fontsize=14)
        plt.tight_layout()
        plt.show()

    def array_to_image(self, array: np.ndarray, cell_size: int = 32) -> Image.Image:
        """Convert a tetromino array to a Pillow image."""
        if array.ndim != 2:
            raise ValueError("Expected a 2-D array.")

        # Normalize to uint8 0‥255
        if np.issubdtype(array.dtype, np.floating):
            arr255 = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
        elif array.dtype == np.uint8:
            maxv = array.max()
            if maxv == 0:
                arr255 = array.copy()
            elif maxv == 1:
                arr255 = array * 255
            else:
                arr255 = array.copy()
        else:
            raise TypeError("Array must be float or uint8.")

        img = Image.fromarray(arr255, mode="L")

        if cell_size != 1:
            img = img.resize(
                (arr255.shape[1] * cell_size, arr255.shape[0] * cell_size),
                resample=Image.NEAREST,
            )
        return img


class OverheadMNISTDataset(PaperDataset):
    """
    Overhead MNIST dataset as specified in the paper.

    Paper quote: "Images were pre-processed through principal component
    analysis featuring 20 parameters."
    """

    # Hard-coded class map (index → name)
    _LABEL_NAMES = {0: "car", 7: "ship"}
    _CLASSES = tuple(_LABEL_NAMES.keys())

    def __init__(
        self,
        n_train: int = 400,
        n_test: int = 100,
        root: str = "data/overhead",
        n_components: int = 20,
        balanced: bool = True,
        random_state: int = 42,
    ):
        self.root = Path(root)
        self.n_components = n_components
        self.balanced = balanced
        self.n_train = n_train
        self.n_test = n_test
        self.rng = np.random.default_rng(random_state)

        # Scan raw JPEGs
        self._X_train_raw, self._y_train_raw = self._scan_split(
            "training", n_samples=n_train
        )
        self._X_test_raw, self._y_test_raw = self._scan_split(
            "testing", n_samples=n_test
        )

        # Fit PCA on training, transform both splits
        self._pca = PCA(n_components=self.n_components, random_state=random_state)
        X_train_flat = self._X_train_raw.reshape(len(self._X_train_raw), -1) / 255.0
        X_test_flat = self._X_test_raw.reshape(len(self._X_test_raw), -1) / 255.0
        X_train_pca = self._pca.fit_transform(X_train_flat)
        X_test_pca = self._pca.transform(X_test_flat)

        scaler = MinMaxScaler()
        X_train_pca = scaler.fit_transform(X_train_pca)
        X_test_pca = scaler.transform(X_test_pca)

        # Convert labels: car=0, ship=1 (original has car=0, ship=7)
        y_train = (self._y_train_raw == 7).astype(int)
        y_test = (self._y_test_raw == 7).astype(int)

        super().__init__(
            X_train=X_train_pca,
            y_train=y_train,
            X_test=X_test_pca,
            y_test=y_test,
            name="Overhead MNIST",
        )

        self.dimension = n_components

    def _scan_split(
        self, folder: str, n_samples: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load all *.jpg in **folder/{car,ship}** returning (X,y), optionally limited to n_samples."""

        imgs, labels = [], []
        for label, cls in self._LABEL_NAMES.items():
            folder_path = self.root / folder / cls
            if not folder_path.exists():
                continue

            for p in folder_path.glob("*.jpg"):
                img = iio.imread(str(p))
                if img.ndim == 3:  # some JPEGs are RGB
                    img = img[..., 0]  # take first channel
                imgs.append(img.astype(np.uint8))
                labels.append(label)

        if not imgs:
            raise FileNotFoundError(f"No images found in {self.root}/{folder}")

        imgs = np.stack(imgs, axis=0)
        labels = np.array(labels, dtype=np.uint8)

        # Optional balancing (before sampling)
        if self.balanced:
            idx_car = np.where(labels == 0)[0]
            idx_ship = np.where(labels == 7)[0]
            n_per_class = min(len(idx_car), len(idx_ship))

            # If n_samples is specified, calculate samples per class
            if n_samples is not None:
                n_per_class = min(n_per_class, n_samples // 2)

            idx = np.concatenate(
                [
                    self.rng.choice(idx_car, n_per_class, replace=False),
                    self.rng.choice(idx_ship, n_per_class, replace=False),
                ]
            )
            self.rng.shuffle(idx)
            imgs, labels = imgs[idx], labels[idx]

        # Apply sample limit if specified and not already handled by balancing
        elif n_samples is not None and len(imgs) > n_samples:
            # Random sampling without balancing
            idx = self.rng.choice(len(imgs), n_samples, replace=False)
            imgs, labels = imgs[idx], labels[idx]

        return imgs, labels

    def plot_samples(
        self, split: str = "train", n_samples: int = 25, figsize: tuple[int, int] = None
    ) -> None:
        """Show *n* random raw images in a square grid."""
        X_raw = self._X_train_raw if split == "train" else self._X_test_raw
        y_raw = self._y_train_raw if split == "train" else self._y_test_raw

        n_samples = min(max(n_samples, 1), len(y_raw))
        idx = self.rng.choice(len(y_raw), n_samples, replace=False)

        side = math.ceil(math.sqrt(n_samples))
        if figsize is None:
            figsize = (side * 1.625, side * 1.625)

        fig, axes = plt.subplots(side, side, figsize=figsize)
        axes = axes.ravel()

        for ax, i in zip(axes, idx):
            ax.imshow(X_raw[i], cmap="gray", vmin=0, vmax=255)
            ax.set_title(self._LABEL_NAMES[int(y_raw[i])], fontsize=8)
            ax.axis("off")

        for j in range(n_samples, len(axes)):
            axes[j].axis("off")

        plt.suptitle(f"Overhead MNIST ({split} split)", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_pca_pairs(self, split: str = "train", max_points: int = 2500) -> None:
        """Plot PCA scatter matrix."""
        X = self.X_train if split == "train" else self.X_test
        y = self.y_train if split == "train" else self.y_test

        # Optional sub-sample
        rng = np.random.default_rng(0)
        if len(X) > max_points:
            idx = rng.choice(len(X), max_points, replace=False)
            X, y = X[idx], y[idx]

        # DataFrame for seaborn
        cols = [f"PC{i + 1}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df["label"] = np.where(y == 0, "car", "ship")

        sns.pairplot(
            df,
            hue="label",
            plot_kws={"s": 12, "alpha": 0.6, "linewidth": 0},
            diag_kws={"fill": False},
            corner=True,
            height=1.6,
        )
        plt.suptitle(f"PCA scatter-matrix ({split} split)", y=1.02)
        plt.show()


# Public API
__all__ = [
    "PaperDataset",
    "CirclesDataset",
    "MoonsDataset",
    "TetrominoDataset",
    "OverheadMNISTDataset",
]
