"""Synthetic Gaussian-mixture dataset for the QEGM rare-event experiment.

The paper (Section VI.C) specifies three Gaussian components with means
{-3, 0, +3} and variances {1, 0.5, 1.5}, where the central mode dominates
with 70% of the mass and the remaining 30% is split between the two
tail components. We follow that specification exactly. The dominant
mode is reweighted via ``weights`` so that tail labels (|x| > tail_threshold)
remain the rare positive class for recall evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GMMDataset:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    tail_threshold: float
    means: tuple
    stds: tuple
    weights: tuple


def sample_gmm(
    n_samples: int,
    means: Sequence[float],
    stds: Sequence[float],
    weights: Sequence[float],
    rng: np.random.Generator,
) -> np.ndarray:
    means_arr = np.asarray(means, dtype=np.float64)
    stds_arr = np.asarray(stds, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()
    components = rng.choice(len(means_arr), size=n_samples, p=weights_arr)
    noise = rng.standard_normal(n_samples)
    return means_arr[components] + stds_arr[components] * noise


def build_gmm_dataset(cfg: dict, seed: int) -> GMMDataset:
    """Materialize the synthetic Gaussian-mixture dataset.

    Parameters
    ----------
    cfg : dict
        Resolved configuration. Expected keys live under ``dataset``.
    seed : int
        Seed for the dataset RNG. Decoupled from training seeds so
        each seed sees the same train/val/test split.

    Returns
    -------
    GMMDataset
        Train / val / test 1-D tensors plus the dataset metadata used
        downstream by metric computations and figure scripts.
    """

    ds_cfg = cfg["dataset"]
    n_samples = int(ds_cfg["n_samples"])
    means = tuple(float(v) for v in ds_cfg["means"])
    stds = tuple(float(v) for v in ds_cfg["stds"])
    weights = tuple(float(v) for v in ds_cfg["weights"])
    tail_threshold = float(ds_cfg["tail_threshold"])
    val_fraction = float(ds_cfg["val_fraction"])
    test_fraction = float(ds_cfg["test_fraction"])

    rng = np.random.default_rng(seed)
    samples = sample_gmm(n_samples, means, stds, weights, rng)
    samples = samples.astype(np.float32).reshape(-1, 1)

    rng.shuffle(samples)
    n_test = int(round(test_fraction * n_samples))
    n_val = int(round(val_fraction * n_samples))
    test = samples[:n_test]
    val = samples[n_test : n_test + n_val]
    train = samples[n_test + n_val :]

    return GMMDataset(
        train=torch.from_numpy(train),
        val=torch.from_numpy(val),
        test=torch.from_numpy(test),
        tail_threshold=tail_threshold,
        means=means,
        stds=stds,
        weights=weights,
    )


def iter_batches(data: torch.Tensor, batch_size: int, *, shuffle: bool, seed: int):
    """Iterate over ``data`` in mini-batches, optionally shuffled."""

    n = data.shape[0]
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator)
    else:
        perm = torch.arange(n)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        yield data[idx]
