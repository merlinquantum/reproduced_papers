"""
Data utilities for the QVT reproduction.

  - ClassicalPatchEmbed:       image → n patches → embed → normalise
  - HierarchicalPatchEmbed:    image → r regions × p patches → embed → normalise
  - get_medmnist_loaders:      standard train/val/test splits
"""

from __future__ import annotations
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def normalize_last_dim(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def make_data_loader_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def make_worker_init_fn(base_seed: int | None):
    if base_seed is None:
        return None

    def _seed_worker(worker_id: int) -> None:
        worker_seed = int(base_seed) + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))
        torch.manual_seed(worker_seed)

    return _seed_worker


def extract_dataset_targets(dataset) -> np.ndarray:
    """Return integer class labels for single-label MedMNIST-style datasets."""
    if hasattr(dataset, "labels"):
        labels = np.asarray(dataset.labels)
    elif hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets)
    else:
        labels = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            label = sample[1]
            if isinstance(label, torch.Tensor):
                label = label.detach().cpu().numpy()
            labels.append(label)
        labels = np.asarray(labels)

    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    return labels


def _random_subset_indices(length: int, subset_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.arange(length)
    rng.shuffle(indices)
    return np.sort(indices[:subset_size])


def _stratified_subset_indices(labels: np.ndarray, subset_size: int, seed: int) -> np.ndarray:
    if labels.ndim != 1:
        raise ValueError(
            "Stratified train subsetting only supports single-label datasets; "
            "the current dataset appears to be multi-label."
        )

    classes, counts = np.unique(labels, return_counts=True)
    n_classes = len(classes)
    if subset_size < n_classes:
        raise ValueError(
            f"train_subset_size={subset_size} is too small for stratified sampling "
            f"across {n_classes} classes."
        )

    ideal = subset_size * counts / counts.sum()
    quotas = np.floor(ideal).astype(int)
    quotas = np.minimum(quotas, counts)
    quotas = np.maximum(quotas, 1)

    while quotas.sum() > subset_size:
        candidates = np.where(quotas > 1)[0]
        idx = candidates[np.argmax(quotas[candidates] - ideal[candidates])]
        quotas[idx] -= 1

    while quotas.sum() < subset_size:
        capacity = counts - quotas
        candidates = np.where(capacity > 0)[0]
        idx = candidates[np.argmax(ideal[candidates] - quotas[candidates])]
        quotas[idx] += 1

    rng = np.random.default_rng(seed)
    picked = []
    for cls, quota in zip(classes, quotas, strict=True):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        picked.extend(cls_indices[:quota].tolist())

    picked = np.asarray(picked, dtype=int)
    rng.shuffle(picked)
    return np.sort(picked)


def select_train_subset(dataset, subset_size: int | None, subset_seed: int | None,
                        subset_mode: str = "stratified"):
    if subset_size is None:
        return dataset

    subset_size = int(subset_size)
    if subset_size <= 0:
        raise ValueError(f"train_subset_size must be positive, got {subset_size}.")
    if subset_size >= len(dataset):
        return dataset

    subset_seed = 0 if subset_seed is None else int(subset_seed)
    if subset_mode == "random":
        indices = _random_subset_indices(len(dataset), subset_size, subset_seed)
    elif subset_mode == "stratified":
        labels = extract_dataset_targets(dataset)
        indices = _stratified_subset_indices(labels, subset_size, subset_seed)
    else:
        raise ValueError(
            f"Unknown train_subset_mode '{subset_mode}'. Expected 'random' or 'stratified'."
        )

    return Subset(dataset, indices.tolist())


class ClassicalPatchEmbed(nn.Module):
    """image [B,C,H,W] → [B, n, d] normalised.  Paper: 28×28, patch=7, n=16, d=16."""

    def __init__(self, img_size: int = 28, in_channels: int = 3,
                 patch_size: int = 7, embed_dim: int = 16):
        super().__init__()
        assert img_size % patch_size == 0, f"{img_size} not divisible by {patch_size}"
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_per_side = img_size // patch_size
        self.n_patches = self.n_per_side ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.linear = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        p = self.patch_size
        n = self.n_per_side
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, n * n, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(B, self.n_patches, self.patch_dim)
        return normalize_last_dim(self.linear(patches))


class ImageLinearEmbed(nn.Module):
    """image [B,C,H,W] → [B, 1, d] using a learned image-wide projection.

    When ``grayscale=True``, RGB inputs are averaged to one channel before the
    global projection. This is useful for matching paper baselines that specify
    a 784xd image-wide layer on 28x28 inputs.
    """

    def __init__(
        self,
        img_size: int = 28,
        in_channels: int = 3,
        embed_dim: int = 16,
        grayscale: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grayscale = grayscale
        effective_channels = 1 if grayscale else in_channels
        self.input_dim = effective_channels * img_size * img_size
        self.linear = nn.Linear(self.input_dim, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.grayscale and images.shape[1] != 1:
            images = images.mean(dim=1, keepdim=True)
        flat = images.reshape(images.shape[0], self.input_dim)
        return normalize_last_dim(self.linear(flat)).unsqueeze(1)


class HierarchicalPatchEmbed(nn.Module):
    """
    Two-level hierarchical patch extraction for the 3-photon model.

    image [B,C,H,W] → [B, r, p, d] normalised.

    Default for 28×28:  2×2 regions (14×14) × 2×2 patches (7×7) × d=16
    → r=4, p=4, total=16 patches (same as paper).

    The spatial hierarchy motivates 3-photon encoding: one photon for
    region, one for patch-within-region, one for feature.
    """

    def __init__(self, img_size: int = 28, in_channels: int = 3,
                 n_regions_per_side: int = 2, n_patches_per_side: int = 2,
                 embed_dim: int = 16):
        super().__init__()
        region_size = img_size // n_regions_per_side
        patch_size = region_size // n_patches_per_side
        assert img_size % n_regions_per_side == 0, \
            f"img_size {img_size} not divisible by n_regions_per_side {n_regions_per_side}"
        assert region_size % n_patches_per_side == 0, \
            f"region_size {region_size} not divisible by n_patches_per_side {n_patches_per_side}"

        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.n_regions_per_side = n_regions_per_side
        self.n_patches_per_side = n_patches_per_side
        self.n_regions = n_regions_per_side ** 2
        self.n_patches_per_region = n_patches_per_side ** 2
        self.n_patches = self.n_regions * self.n_patches_per_region
        self.region_size = region_size
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.linear = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, C, H, W] → [B, r, p, d] normalised."""
        B, C, H, W = images.shape
        rn, pn = self.n_regions_per_side, self.n_patches_per_side
        rs, ps = self.region_size, self.patch_size

        # [B, C, H, W] → [B, C, rn, rs, rn, rs]  (split H and W into regions)
        x = images.view(B, C, rn, rs, rn, rs)
        # → [B, rn, rn, C, rs, rs]  (group by region)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # → [B, r, C, rs, rs]
        x = x.view(B, self.n_regions, C, rs, rs)
        # → [B, r, C, pn, ps, pn, ps]  (split region into patches)
        x = x.view(B, self.n_regions, C, pn, ps, pn, ps)
        # → [B, r, pn, pn, C, ps, ps]
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        # → [B, r, p, patch_dim]
        x = x.view(B, self.n_regions, self.n_patches_per_region, self.patch_dim)

        return normalize_last_dim(self.linear(x))


def get_medmnist_loaders(
    dataset_name: str = "retinamnist",
    batch_size: int = 32,
    data_root: str = "data/QVT",
    num_workers: int = 2,
    download: bool = True,
    seed: int | None = None,
    train_subset_size: int | None = None,
    train_subset_seed: int | None = None,
    train_subset_mode: str = "stratified",
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Return (train, val, test) loaders and n_classes."""
    import os
    import medmnist
    from medmnist import INFO
    from torchvision import transforms

    os.makedirs(data_root, exist_ok=True)
    info = INFO[dataset_name]
    n_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])
    tfm = transforms.Compose([transforms.ToTensor()])

    loaders = []
    split_offsets = {"train": 0, "val": 1, "test": 2}
    for split in ("train", "val", "test"):
        ds = DataClass(split=split, transform=tfm, download=download, root=data_root)
        if split == "train":
            ds = select_train_subset(ds, train_subset_size, train_subset_seed, train_subset_mode)
        split_seed = None if seed is None else int(seed) + split_offsets[split]
        loaders.append(DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"), num_workers=num_workers,
            pin_memory=True, drop_last=(split == "train"),
            generator=make_data_loader_generator(split_seed),
            worker_init_fn=make_worker_init_fn(split_seed),
        ))
    return (*loaders, n_classes)
