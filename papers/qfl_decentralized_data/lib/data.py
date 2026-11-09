"""MNIST pipeline for the QFL reproduction (paper Sec. III-A).

Steps: read the original IDX files, reduce 28x28 images to 4x4 by area
interpolation (exact 7x7 block means), keep the classes of one task, optionally hold
out a validation split, and split the training set across clients.

The paper does not release its client split (it cites LEAF and shows label
histograms only), so the partition here is an assumption: for every class, the
samples are divided among clients with proportions drawn from a Dirichlet
distribution. Small ``dirichlet_alpha`` gives both skewed label proportions and
unbalanced client sizes, which are the two properties the paper asks for.
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from runtime_lib.data_paths import resolve_data_root

logger = logging.getLogger(__name__)

IMAGE_SIDE = 28
DOWNSAMPLED_SIDE = 4
BLOCK = IMAGE_SIDE // DOWNSAMPLED_SIDE  # 7
MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000),
}
MAX_PARTITION_ATTEMPTS = 200


def mnist_directory(cfg: dict[str, Any]) -> Path:
    """Locate the raw MNIST files under the shared data root (``--data-root`` / ``DATA_DIR``)."""
    root = resolve_data_root(cfg.get("data_root"))
    directory = root / cfg["dataset"]["subdir"]
    if not directory.is_dir():
        raise FileNotFoundError(
            f"MNIST directory not found at {directory}. See the README's Data "
            "section for the download command, or pass --data-root."
        )
    return directory


def read_idx(path: Path, expected_count: int, is_image: bool) -> np.ndarray:
    """Read one gzip IDX file; images come back as ``(n, 28, 28)`` uint8."""
    with gzip.open(path, "rb") as handle:
        buffer = handle.read()
    if is_image:
        array = np.frombuffer(buffer, dtype=np.uint8, offset=16).reshape(
            -1, IMAGE_SIDE, IMAGE_SIDE
        )
    else:
        array = np.frombuffer(buffer, dtype=np.uint8, offset=8)
    if len(array) != expected_count:
        raise ValueError(
            f"{path.name}: expected {expected_count} entries, found {len(array)}"
        )
    return array


def downsample_area(images: np.ndarray) -> np.ndarray:
    """Area interpolation 28x28 -> 4x4 scaled to [0, 1], flattened to 16 values."""
    count = len(images)
    blocks = images.reshape(count, DOWNSAMPLED_SIDE, BLOCK, DOWNSAMPLED_SIDE, BLOCK)
    return (blocks.mean(axis=(2, 4)) / 255.0).reshape(count, -1)


def _stratified_holdout(
    labels: torch.Tensor, fraction: float, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    kept, held = [], []
    for label in labels.unique():
        indices = (labels == label).nonzero().flatten()
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        cut = int(round(fraction * len(indices)))
        held.append(shuffled[:cut])
        kept.append(shuffled[cut:])
    return torch.cat(kept).sort().values, torch.cat(held).sort().values


def dirichlet_partition(
    labels: torch.Tensor,
    n_clients: int,
    alpha: float,
    generator: torch.Generator,
    min_client_size: int,
) -> list[torch.Tensor]:
    """Split sample indices over clients with Dirichlet label proportions.

    Parameters
    ----------
    labels : torch.Tensor
        Training labels.
    n_clients : int
        Number of clients.
    alpha : float
        Dirichlet concentration; smaller means more skew and more imbalance.
    generator : torch.Generator
        Source of randomness.
    min_client_size : int
        Redraw until every client holds at least this many samples.

    Returns
    -------
    list of torch.Tensor
        One index tensor per client.

    Raises
    ------
    RuntimeError
        If no partition satisfies ``min_client_size`` after the allowed attempts.
    """
    concentration = torch.full((n_clients,), alpha, dtype=torch.float64)
    for _ in range(MAX_PARTITION_ATTEMPTS):
        parts: list[list[torch.Tensor]] = [[] for _ in range(n_clients)]
        for label in labels.unique():
            indices = (labels == label).nonzero().flatten()
            indices = indices[torch.randperm(len(indices), generator=generator)]
            proportions = sample_dirichlet(concentration, generator)
            cuts = (proportions.cumsum(0)[:-1] * len(indices)).long()
            for client, chunk in enumerate(torch.tensor_split(indices, cuts.tolist())):
                parts[client].append(chunk)
        merged = [torch.cat(chunks).sort().values for chunks in parts]
        if min(len(part) for part in merged) >= min_client_size:
            return merged
    raise RuntimeError(
        f"No Dirichlet(alpha={alpha}) partition over {n_clients} clients gives every "
        f"client at least {min_client_size} samples; raise alpha or lower min_client_size"
    )


def sample_dirichlet(
    concentration: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    # torch.distributions.Dirichlet cannot take a generator, so draw the gamma
    # variates directly to keep the partition reproducible from data_seed.
    gamma = torch._standard_gamma(concentration, generator=generator)
    return gamma / gamma.sum()


def iid_partition(
    n_samples: int, n_clients: int, generator: torch.Generator
) -> list[torch.Tensor]:
    """Equal-size random split (control for the non-i.i.d. partition)."""
    shuffled = torch.randperm(n_samples, generator=generator)
    return [part.sort().values for part in torch.tensor_split(shuffled, n_clients)]


def build_mnist_splits(cfg: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    """Load, reduce and split MNIST for one task.

    Parameters
    ----------
    cfg : dict
        Resolved configuration; uses ``dataset.*`` and ``federation.*``.
    dtype : torch.dtype
        Real dtype of the returned pixel tensors.

    Returns
    -------
    dict
        ``train``, ``test`` (and ``validation`` when ``validation_fraction > 0``) as
        ``(pixels, labels)`` tuples, ``client_indices`` (indices into ``train``) and
        a ``summary`` of sizes and per-client label counts.
    """
    dataset = cfg["dataset"]
    classes = list(dataset["classes"])
    directory = mnist_directory(cfg)
    split_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, (image_file, label_file, count) in MNIST_FILES.items():
        images = read_idx(directory / image_file, count, is_image=True)
        labels = read_idx(directory / label_file, count, is_image=False)
        keep = np.isin(labels, classes)
        relabel = {original: index for index, original in enumerate(classes)}
        pixels = torch.from_numpy(downsample_area(images[keep])).to(dtype)
        mapped = torch.tensor([relabel[int(v)] for v in labels[keep]], dtype=torch.long)
        split_tensors[name] = (pixels, mapped)

    generator = torch.Generator().manual_seed(int(dataset["data_seed"]))
    # Reduced (smoke or partial) runs: a seeded random subset of each split.
    for name, limit_key in (
        ("train", "max_train_samples"),
        ("test", "max_test_samples"),
    ):
        limit = dataset[limit_key]
        if limit is not None and limit < len(split_tensors[name][1]):
            chosen = torch.randperm(len(split_tensors[name][1]), generator=generator)
            chosen = chosen[: int(limit)].sort().values
            pixels, mapped = split_tensors[name]
            split_tensors[name] = (pixels[chosen], mapped[chosen])
    fraction = float(dataset["validation_fraction"])
    train_pixels, train_labels = split_tensors["train"]
    if fraction > 0:
        keep_idx, held_idx = _stratified_holdout(train_labels, fraction, generator)
        split_tensors["validation"] = (train_pixels[held_idx], train_labels[held_idx])
        split_tensors["train"] = (train_pixels[keep_idx], train_labels[keep_idx])
    scaling = dataset["feature_scaling"]
    if scaling == "per_feature_minmax":
        # Fitted on the training split only, then applied to every split.
        reference = split_tensors["train"][0]
        low, high = reference.min(dim=0).values, reference.max(dim=0).values
        span = (high - low).clamp_min(1e-6)
        for name, (pixels, labels) in list(split_tensors.items()):
            split_tensors[name] = (((pixels - low) / span).clamp(0.0, 1.0), labels)
    elif scaling != "none":
        raise ValueError(f"Unknown dataset.feature_scaling {scaling!r}")
    train_labels = split_tensors["train"][1]

    federation = cfg["federation"]
    if federation["mode"] == "federated":
        n_clients = int(federation["n_clients"])
        if federation["partition"] == "noniid":
            client_indices = dirichlet_partition(
                train_labels,
                n_clients,
                float(federation["dirichlet_alpha"]),
                generator,
                int(federation["min_client_size"]),
            )
        elif federation["partition"] == "iid":
            client_indices = iid_partition(len(train_labels), n_clients, generator)
        else:
            raise ValueError(
                f"Unknown federation.partition {federation['partition']!r}"
            )
    else:
        client_indices = [torch.arange(len(train_labels))]

    n_classes = len(classes)
    summary = {
        "classes": classes,
        "split_sizes": {name: len(pair[1]) for name, pair in split_tensors.items()},
        "client_sizes": [len(part) for part in client_indices],
        "client_label_counts": [
            torch.bincount(train_labels[part], minlength=n_classes).tolist()
            for part in client_indices
        ],
    }
    logger.info("PARTITION | sizes=%s", summary["client_sizes"])
    return {**split_tensors, "client_indices": client_indices, "summary": summary}


def build_dataloader(
    cfg: dict[str, Any],
    split: str,
    splits: dict[str, Any],
    client: int | None = None,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Return a loader over a split, or over one client's share of the training set.

    Parameters
    ----------
    cfg : dict
        Resolved configuration; ``training.batch_size`` sets the batch size.
    split : str
        ``train``, ``validation`` or ``test``.
    splits : dict
        Output of :func:`build_mnist_splits`.
    client : int | None
        Client index; only valid with ``split="train"``. Default value is None.
    generator : torch.Generator | None
        Shuffling generator for training loaders. Default value is None.

    Returns
    -------
    torch.utils.data.DataLoader
        Batches of ``(pixels, labels)``.
    """
    pixels, labels = splits[split]
    if client is not None:
        if split != "train":
            raise ValueError("client loaders are only defined for the train split")
        indices = splits["client_indices"][client]
        pixels, labels = pixels[indices], labels[indices]
    return DataLoader(
        TensorDataset(pixels, labels),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=split == "train",
        generator=generator if split == "train" else None,
    )
