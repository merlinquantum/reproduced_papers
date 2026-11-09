"""Fashion-MNIST loading, PCA preprocessing and subsampling (paper App. J).

Pipeline (arXiv:2011.01938 App. J; AMB-10 for the PCA fit set):

1. Read all 70,000 Fashion-MNIST images (60k train + 10k test) from the official
   gzipped IDX files under ``<data_root>/power_of_data_qml/fashion_mnist/``.
2. Fit scikit-learn PCA once on all images, keeping the largest requested number
   of components. The first ``n`` components of that fit equal an ``n``-component
   fit (full SVD), so every system size slices the same projection.
3. Standardize each component to mean 0 and standard deviation 1 over all images.
4. Per seed, subsample points without replacement: from all classes for the
   quantum and engineered datasets, and from classes 3 (dress) and 6 (shirt)
   for Dataset (C).

There is no synthetic fallback: the real dataset is required. Missing files are
downloaded once from the official release and verified against the MD5 sums that
Zalando publishes; a failed download or a checksum mismatch raises.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)

PAPER_DATA_DIRNAME = "power_of_data_qml"
DATASET_DIRNAME = "fashion_mnist"
IDX_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
# Official release (https://github.com/zalandoresearch/fashion-mnist, MIT License),
# MD5 sums as published with the files. The mirrors serve identical files; the
# first reachable one is used and the checksum decides whether it is accepted.
IDX_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
DOWNLOAD_MIRRORS = (
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
)
DOWNLOAD_TIMEOUT_SECONDS = 60
REACQUIRE_HINT = (
    "Fashion-MNIST is missing and could not be downloaded; see the Data section "
    "of papers/power_of_data_qml/README.md for the manual acquisition commands."
)


def dataset_dir(data_root: str | Path) -> Path:
    """Return ``<data_root>/power_of_data_qml/fashion_mnist``."""
    return Path(data_root) / PAPER_DATA_DIRNAME / DATASET_DIRNAME


def _md5(payload: bytes) -> str:
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def ensure_fashion_mnist(data_root: str | Path) -> Path:
    """Download any missing IDX file, verifying its MD5 sum, and return the directory.

    Parameters
    ----------
    data_root : str or pathlib.Path
        Shared data root (``DATA_DIR`` or ``--data-root``).

    Returns
    -------
    pathlib.Path
        ``<data_root>/power_of_data_qml/fashion_mnist``.

    Raises
    ------
    RuntimeError
        If a file cannot be downloaded from any mirror with the expected checksum.
    """
    directory = dataset_dir(data_root)
    for name, checksum in IDX_MD5.items():
        target = directory / name
        if target.is_file():
            continue
        failures = []
        for mirror in DOWNLOAD_MIRRORS:
            try:
                with urllib.request.urlopen(
                    mirror + name, timeout=DOWNLOAD_TIMEOUT_SECONDS
                ) as response:
                    payload = response.read()
            except OSError as error:
                failures.append(f"{mirror}: {error}")
                continue
            if _md5(payload) != checksum:
                failures.append(f"{mirror}: MD5 mismatch")
                continue
            directory.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            partial.write_bytes(payload)
            partial.replace(target)
            LOGGER.info("Downloaded %s from %s", name, mirror)
            break
        else:
            raise RuntimeError(f"Could not download {name}: {'; '.join(failures)}")
    return directory


def _read_idx(path: Path, expected_magic: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. {REACQUIRE_HINT}")
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    magic = int.from_bytes(raw[0:4], "big")
    if magic != expected_magic:
        raise ValueError(f"{path}: IDX magic {magic:#x}, expected {expected_magic:#x}")
    n_dims = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(n_dims)]
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * n_dims)
    return data.reshape(dims)


def load_fashion_mnist(data_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all 70,000 images and labels.

    Parameters
    ----------
    data_root : str or pathlib.Path
        Shared data root (``DATA_DIR`` or ``--data-root``).

    Returns
    -------
    tuple of numpy.ndarray
        Images of shape ``(70000, 784)`` as float64 in [0, 1] and labels of shape
        ``(70000,)`` (training images first, then test images).

    Raises
    ------
    RuntimeError
        If a missing file cannot be downloaded (see ``ensure_fashion_mnist``).
    """
    directory = ensure_fashion_mnist(data_root)
    train_images = _read_idx(directory / IDX_FILES["train_images"], 0x803)
    train_labels = _read_idx(directory / IDX_FILES["train_labels"], 0x801)
    test_images = _read_idx(directory / IDX_FILES["test_images"], 0x803)
    test_labels = _read_idx(directory / IDX_FILES["test_labels"], 0x801)
    images = np.concatenate([train_images, test_images]).reshape(-1, 28 * 28)
    labels = np.concatenate([train_labels, test_labels]).astype(np.int64)
    if images.shape != (70000, 784):
        raise ValueError(f"Unexpected Fashion-MNIST shape {images.shape}")
    return images.astype(np.float64) / 255.0, labels


@dataclass
class PCAFeatures:
    """Standardized PCA features of every Fashion-MNIST image.

    Attributes
    ----------
    components : numpy.ndarray
        Shape ``(70000, max_components)``; column k is the k-th principal
        component, standardized to mean 0 and std 1.
    labels : numpy.ndarray
        Class labels, shape ``(70000,)``.
    explained_variance_ratio : numpy.ndarray
        PCA explained variance ratio per component.
    """

    components: np.ndarray
    labels: np.ndarray
    explained_variance_ratio: np.ndarray

    def features(self, n_components: int, indices: np.ndarray) -> np.ndarray:
        """Return the first ``n_components`` standardized components of ``indices``."""
        if n_components > self.components.shape[1]:
            raise ValueError(
                f"Requested {n_components} components; PCA was fit with "
                f"{self.components.shape[1]}"
            )
        return self.components[indices, :n_components]


def build_pca_features(data_root: str | Path, max_components: int) -> PCAFeatures:
    """Fit PCA on all images and standardize the components (App. J, AMB-10).

    Parameters
    ----------
    data_root : str or pathlib.Path
        Shared data root.
    max_components : int
        Largest system size ``n`` used by the run.
    """
    images, labels = load_fashion_mnist(data_root)
    pca = PCA(n_components=max_components, svd_solver="full")
    projected = pca.fit_transform(images)
    standardized = (projected - projected.mean(axis=0)) / projected.std(axis=0)
    LOGGER.info(
        "PCA fit on %d images | components=%d | explained_variance=%.4f",
        images.shape[0],
        max_components,
        float(pca.explained_variance_ratio_.sum()),
    )
    return PCAFeatures(standardized, labels, pca.explained_variance_ratio_)


def subsample_indices(
    labels: np.ndarray,
    size: int,
    rng: np.random.Generator,
    classes: list[int] | None = None,
) -> np.ndarray:
    """Draw ``size`` image indices without replacement.

    Parameters
    ----------
    labels : numpy.ndarray
        Labels of all images.
    size : int
        Subsample size (paper: 800).
    rng : numpy.random.Generator
        Seeded generator for this run.
    classes : list of int or None
        Restrict the pool to these classes (Dataset (C): [3, 6]). Default value
        is None (all classes).
    """
    pool = (
        np.arange(labels.shape[0])
        if classes is None
        else np.flatnonzero(np.isin(labels, classes))
    )
    if size > pool.shape[0]:
        raise ValueError(f"Subsample size {size} exceeds the pool of {pool.shape[0]}")
    return rng.choice(pool, size=size, replace=False)


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    """Return a loader over one seed's subsample, for inspection and tests.

    The experiments consume whole subsamples as kernel matrices rather than
    mini-batches; this loader exposes the same data through the standard
    interface. ``split="train"`` gives the first ``train_size`` points of the
    subsample and ``split="test"`` the remainder (paper: test = 800 - N).

    Parameters
    ----------
    cfg : dict
        Resolved config; uses ``data_root``, ``dataset`` and ``study``.
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    torch.utils.data.DataLoader
        Batches of ``(features, class_labels)`` with features of shape
        ``(batch, n)`` for ``n = study.system_sizes[0]``.
    """
    dataset_cfg = cfg["dataset"]
    study_cfg = cfg["study"]
    n_components = int(study_cfg["system_sizes"][0])
    pca_features = build_pca_features(cfg["data_root"], n_components)
    rng = np.random.default_rng(int(study_cfg["seeds"][0]))
    indices = subsample_indices(
        pca_features.labels, int(study_cfg["subsample_size"]), rng
    )
    train_size = int(study_cfg["quantum_dataset_train_size"])
    if split == "train":
        chosen = indices[:train_size]
    elif split == "test":
        chosen = indices[train_size:]
    else:
        raise ValueError(f"Unknown split: {split}")
    features = torch.as_tensor(pca_features.features(n_components, chosen))
    labels = torch.as_tensor(pca_features.labels[chosen])
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=int(dataset_cfg["batch_size"]),
        shuffle=False,
    )


__all__ = [
    "PCAFeatures",
    "build_dataloader",
    "build_pca_features",
    "dataset_dir",
    "load_fashion_mnist",
    "subsample_indices",
]
