"""Datasets, fixed splits and loaders for the PQKD reproduction.

The paper (Methods, "Datasets and preprocessing") uses MNIST and Fashion-MNIST
rescaled to [0, 1] without normalisation or augmentation, CIFAR-10 normalised
per channel, and fixed ``train/val = 45,000/5,000`` splits drawn with a fixed
seed. The exact split indices were not released, so the split is regenerated
here from the canonical training set with ``dataset.split_seed``; the canonical
test set is kept as the held-out test split.

The paper's synthetic Gaussian-blob dataset (Methods, "Synthetic dataset") is
available as ``dataset.name = "synthetic"``. It is the fallback dataset behind
the same interface and needs no download.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from torch.utils.data import TensorDataset

from runtime_lib.data_paths import paper_data_dir

logger = logging.getLogger(__name__)

PAPER_NAME = "photonic_QKD"
NUM_CLASSES = 10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DATASET_INFO: dict[str, dict[str, Any]] = {
    "mnist": {"channels": 1, "image_size": 28, "preprocessing": "ToTensor in [0,1]"},
    "fashion_mnist": {
        "channels": 1,
        "image_size": 28,
        "preprocessing": "ToTensor in [0,1]",
    },
    "cifar10": {
        "channels": 3,
        "image_size": 32,
        "preprocessing": f"ToTensor then per-channel mean {CIFAR10_MEAN} std {CIFAR10_STD}",
    },
    "synthetic": {
        "channels": 1,
        "image_size": 28,
        "preprocessing": "generated in [0,1] (paper Methods synthetic blobs)",
    },
}


@dataclass(frozen=True)
class DatasetSplits:
    """In-memory train/validation/test tensors for one dataset configuration.

    Attributes
    ----------
    name : str
        Dataset key from :data:`DATASET_INFO`.
    train, val, test : torch.utils.data.TensorDataset
        ``(images, labels)`` tensors; images are ``float32`` with shape
        ``(N, C, H, W)``.
    description : dict
        Dataset identity recorded in ``run_status.json``.
    """

    name: str
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    description: dict[str, Any]

    def split(self, split: str) -> TensorDataset:
        """Return one split by name (``train``, ``val`` or ``test``)."""
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}; expected train, val or test")
        return getattr(self, split)


def dataset_info(name: str) -> dict[str, Any]:
    """Return channel count, image size and preprocessing for a dataset key."""
    if name not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset {name!r}; expected one of {sorted(DATASET_INFO)}"
        )
    return DATASET_INFO[name]


def _required(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise KeyError(f"Missing required config key {where}.{key}")
    return section[key]


def _canonical_tensors(
    name: str, root: str, train: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    import torchvision

    if name in ("mnist", "fashion_mnist"):
        cls = (
            torchvision.datasets.MNIST
            if name == "mnist"
            else torchvision.datasets.FashionMNIST
        )
        dataset = cls(root, train=train, download=True)
        images = dataset.data.to(torch.float32).div_(255.0).unsqueeze(1)
        labels = dataset.targets.to(torch.long)
        return images, labels
    if name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root, train=train, download=True)
        images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).to(torch.float32)
        images.div_(255.0)
        mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        images = (images - mean) / std
        labels = torch.tensor(dataset.targets, dtype=torch.long)
        return images, labels
    raise ValueError(f"No canonical download for dataset {name!r}")


def generate_synthetic_blobs(
    n_samples: int,
    *,
    seed: int,
    anchor_seed: int,
    blob_sigma: float,
    jitter_sigma: float,
    pixel_sigma: float,
    image_size: int = 28,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the paper's class-balanced synthetic Gaussian-blob images.

    Each class owns two or three fixed anchors (drawn once with
    ``anchor_seed``); every image adds jittered Gaussian blobs at its class
    anchors with amplitudes in ``U(0.8, 1.2)``, then pixel noise, then clips to
    [0, 1] (paper Methods, steps 1-3).

    Parameters
    ----------
    n_samples : int
        Number of images.
    seed : int
        Seed for labels, jitter, amplitudes and pixel noise.
    anchor_seed : int
        Seed for the class anchor layout, shared by every split.
    blob_sigma, jitter_sigma, pixel_sigma : float
        Paper values are 1.6, 0.8 and 0.05.
    image_size : int
        Image side length. Default value is 28.

    Returns
    -------
    tuple of torch.Tensor
        Images ``(n_samples, 1, image_size, image_size)`` and labels.
    """
    anchor_rng = torch.Generator().manual_seed(anchor_seed)
    anchors_per_class = torch.randint(2, 4, (NUM_CLASSES,), generator=anchor_rng)
    max_anchors = 3
    margin = 4.0
    anchors = margin + torch.rand(NUM_CLASSES, max_anchors, 2, generator=anchor_rng) * (
        image_size - 1 - 2 * margin
    )
    anchor_mask = torch.arange(max_anchors).view(1, -1) < anchors_per_class.view(-1, 1)

    rng = torch.Generator().manual_seed(seed)
    labels = torch.arange(n_samples) % NUM_CLASSES
    labels = labels[torch.randperm(n_samples, generator=rng)]
    grid = torch.arange(image_size, dtype=torch.float32)
    images = torch.empty(n_samples, 1, image_size, image_size)
    chunk = 4096
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        chunk_labels = labels[start:stop]
        centres = anchors[chunk_labels] + jitter_sigma * torch.randn(
            stop - start, max_anchors, 2, generator=rng
        )
        amplitudes = 0.8 + 0.4 * torch.rand(stop - start, max_anchors, generator=rng)
        amplitudes = amplitudes * anchor_mask[chunk_labels]
        du = (grid.view(1, 1, -1) - centres[..., 0:1]) ** 2  # (n, m, H)
        dv = (grid.view(1, 1, -1) - centres[..., 1:2]) ** 2  # (n, m, W)
        blobs = torch.exp(-(du.unsqueeze(-1) + dv.unsqueeze(-2)) / (2 * blob_sigma**2))
        image = (amplitudes.view(stop - start, max_anchors, 1, 1) * blobs).sum(dim=1)
        image = image + pixel_sigma * torch.randn(image.shape, generator=rng)
        images[start:stop, 0] = image.clamp_(0.0, 1.0)
    return images, labels.to(torch.long)


@lru_cache(maxsize=4)
def _load_splits_cached(
    name: str,
    data_root: str | None,
    train_size: int,
    val_size: int,
    split_seed: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    synthetic_key: tuple | None,
) -> DatasetSplits:
    if name == "synthetic":
        assert synthetic_key is not None
        test_size, blob_sigma, jitter_sigma, pixel_sigma = synthetic_key
        common = {
            "anchor_seed": split_seed,
            "blob_sigma": blob_sigma,
            "jitter_sigma": jitter_sigma,
            "pixel_sigma": pixel_sigma,
        }
        pool_images, pool_labels = generate_synthetic_blobs(
            train_size + val_size, seed=split_seed + 1, **common
        )
        test_images, test_labels = generate_synthetic_blobs(
            test_size, seed=split_seed + 2, **common
        )
        source = "generated (paper Methods synthetic dataset)"
    else:
        root = str(paper_data_dir(PAPER_NAME, data_root))
        pool_images, pool_labels = _canonical_tensors(name, root, train=True)
        test_images, test_labels = _canonical_tensors(name, root, train=False)
        source = f"torchvision {name} canonical train/test under {root}"

    pool_size = pool_images.shape[0]
    if train_size + val_size > pool_size:
        raise ValueError(
            f"train_size + val_size = {train_size + val_size} exceeds the "
            f"{pool_size} canonical training images of {name}"
        )
    permutation = torch.randperm(
        pool_size, generator=torch.Generator().manual_seed(split_seed)
    )
    train_index = permutation[:train_size]
    val_index = permutation[train_size : train_size + val_size]
    test_index = torch.arange(test_images.shape[0])
    if max_train_samples is not None:
        train_index = train_index[:max_train_samples]
    if max_eval_samples is not None:
        val_index = val_index[:max_eval_samples]
        test_index = test_index[:max_eval_samples]

    subset = None
    if max_train_samples is not None or max_eval_samples is not None:
        subset = {
            "max_train_samples": max_train_samples,
            "max_eval_samples": max_eval_samples,
        }
    description = {
        "name": name,
        "source": source,
        "splits": {
            "train": int(train_index.numel()),
            "val": int(val_index.numel()),
            "test": int(test_index.numel()),
            "split_seed": split_seed,
            "rule": "seeded permutation of canonical train: first train_size -> train, next val_size -> val; canonical test kept",
        },
        "preprocessing": dataset_info(name)["preprocessing"],
        "subset": subset,
    }
    return DatasetSplits(
        name=name,
        train=TensorDataset(pool_images[train_index], pool_labels[train_index]),
        val=TensorDataset(pool_images[val_index], pool_labels[val_index]),
        test=TensorDataset(test_images[test_index], test_labels[test_index]),
        description=description,
    )


def load_splits(cfg: dict[str, Any]) -> DatasetSplits:
    """Load (and cache) the train/validation/test splits described by ``cfg``.

    Parameters
    ----------
    cfg : dict
        Resolved run configuration. Uses ``cfg["dataset"]`` and the optional
        ``cfg["data_root"]`` set by the shared runtime (``--data-root`` or the
        ``DATA_DIR`` environment variable).

    Returns
    -------
    DatasetSplits
        In-memory splits.

    Raises
    ------
    KeyError
        If a required ``dataset`` key is missing.
    """
    dataset_cfg = _required(cfg, "dataset", "cfg")
    name = _required(dataset_cfg, "name", "dataset")
    dataset_info(name)
    synthetic_key = None
    if name == "synthetic":
        synthetic_cfg = _required(dataset_cfg, "synthetic", "dataset")
        synthetic_key = (
            int(_required(synthetic_cfg, "test_size", "dataset.synthetic")),
            float(_required(synthetic_cfg, "blob_sigma", "dataset.synthetic")),
            float(_required(synthetic_cfg, "jitter_sigma", "dataset.synthetic")),
            float(_required(synthetic_cfg, "pixel_sigma", "dataset.synthetic")),
        )
    return _load_splits_cached(
        name,
        cfg.get("data_root"),
        int(_required(dataset_cfg, "train_size", "dataset")),
        int(_required(dataset_cfg, "val_size", "dataset")),
        int(_required(dataset_cfg, "split_seed", "dataset")),
        _required(dataset_cfg, "max_train_samples", "dataset"),
        _required(dataset_cfg, "max_eval_samples", "dataset"),
        synthetic_key,
    )


class TensorBatchLoader:
    """Mini-batch iterator over in-memory tensors, indexed one batch at a time.

    Same semantics as ``DataLoader(TensorDataset(...), batch_size, shuffle)``
    with ``drop_last=False``: a fresh permutation per epoch when shuffling, the
    last partial batch kept, fixed order otherwise. Indexing whole batches
    avoids ``DataLoader``'s per-sample fetch and collate, which cost about 30%
    of a teacher epoch on CPU.

    Parameters
    ----------
    dataset : torch.utils.data.TensorDataset
        Tensors sharing their first dimension.
    batch_size : int
        Samples per batch.
    shuffle : bool
        Draw a new permutation at the start of every epoch.
    seed : int, optional
        Seed of the permutation generator; required when ``shuffle`` is True.
        Default value is None.
    """

    def __init__(
        self,
        dataset: TensorDataset,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int | None = None,
    ) -> None:
        if shuffle and seed is None:
            raise ValueError("A shuffled loader needs a seed")
        self.tensors = dataset.tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed) if shuffle else None
        self.n_samples = self.tensors[0].shape[0]

    def __len__(self) -> int:
        return math.ceil(self.n_samples / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n_samples, generator=self.generator)
            for start in range(0, self.n_samples, self.batch_size):
                index = order[start : start + self.batch_size]
                yield tuple(tensor[index] for tensor in self.tensors)
        else:
            for start in range(0, self.n_samples, self.batch_size):
                yield tuple(
                    tensor[start : start + self.batch_size] for tensor in self.tensors
                )


def make_loader(
    dataset: TensorDataset, batch_size: int, *, shuffle: bool, seed: int | None = None
) -> TensorBatchLoader:
    """Wrap an in-memory dataset in a :class:`TensorBatchLoader`.

    Shuffling uses its own generator so the batch order depends only on
    ``seed`` (paper "Randomness control").
    """
    return TensorBatchLoader(dataset, batch_size, shuffle=shuffle, seed=seed)


def build_dataloader(cfg: dict[str, Any], split: str) -> TensorBatchLoader:
    """Build the loader for one split.

    Training batches use ``dataset.batch_size`` (paper: 64) and are shuffled
    with ``cfg["seed"]``; validation and test use ``dataset.eval_batch_size``
    in a fixed order.

    Parameters
    ----------
    cfg : dict
        Resolved run configuration.
    split : str
        ``train``, ``val`` or ``test``.

    Returns
    -------
    TensorBatchLoader
        Loader yielding ``(images, labels)``.
    """
    splits = load_splits(cfg)
    dataset_cfg = cfg["dataset"]
    if split == "train":
        return make_loader(
            splits.train,
            int(_required(dataset_cfg, "batch_size", "dataset")),
            shuffle=True,
            seed=int(_required(cfg, "seed", "cfg")),
        )
    return make_loader(
        splits.split(split),
        int(_required(dataset_cfg, "eval_batch_size", "dataset")),
        shuffle=False,
    )
