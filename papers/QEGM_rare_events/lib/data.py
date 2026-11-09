"""Datasets for the QEGM rare-event experiment.

Synthetic benchmark: the paper (Section VI.C) specifies three Gaussian
components with means {-3, 0, +3} and variances {1, 0.5, 1.5}, where the
central mode dominates with 70% of the mass and the remaining 30% is
split between the two tail components. We follow that specification
exactly. The dominant mode is reweighted via ``weights`` so that tail
labels (|x| > tail_threshold) remain the rare positive class for recall
evaluation.

Real-data extension: standardized S&P 500 daily log-returns (the asset
named in the paper's Sec. VI.D finance experiment), loaded from a
packaged CSV. The paper does not specify preprocessing or
hyper-parameters for its real-world experiments, so this path supports
our own ablation study on real heavy-tailed data rather than a
reproduction of the paper's Sec. VI.D numbers. The tail is defined
two-sided (|standardized return| above a quantile-derived threshold) to
stay consistent with the synthetic benchmark's tail convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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

    def describe(self) -> dict:
        return {
            "name": "gmm3",
            "means": list(self.means),
            "stds": list(self.stds),
            "weights": list(self.weights),
            "tail_threshold": self.tail_threshold,
        }


@dataclass
class SP500Dataset:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    tail_threshold: float
    csv_path: str
    n_samples: int
    tail_quantile: float

    def describe(self) -> dict:
        return {
            "name": "sp500",
            "csv_path": self.csv_path,
            "n_samples": self.n_samples,
            "tail_quantile": self.tail_quantile,
            "tail_threshold": self.tail_threshold,
            "standardized": True,
        }


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


def build_sp500_dataset(cfg: dict, seed: int) -> SP500Dataset:
    """Load standardized S&P 500 daily log-returns from the packaged CSV.

    The CSV (see ``utils/fetch_sp500.py``) has one ``date,log_return`` row
    per trading day. Returns are standardized to zero mean / unit variance
    so the model and metric scales match the synthetic benchmark, and the
    tail threshold is the (1 - tail_quantile) quantile of |standardized
    return| — e.g. ``tail_quantile: 0.05`` marks the 5% most extreme days
    (both signs) as the rare class. The split is a seeded shuffle, matching
    the GMM path: the generative model treats returns as i.i.d. draws.
    """

    ds_cfg = cfg["dataset"]
    csv_path = Path(str(ds_cfg["csv_path"]))
    if not csv_path.is_absolute():
        # Resolve relative to the paper directory so runs work from the
        # repo root and from the paper directory alike.
        csv_path = (Path(__file__).resolve().parents[1] / csv_path).resolve()
    tail_quantile = float(ds_cfg.get("tail_quantile", 0.05))
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError(f"tail_quantile must be in (0, 1); got {tail_quantile}")
    val_fraction = float(ds_cfg["val_fraction"])
    test_fraction = float(ds_cfg["test_fraction"])

    raw = np.genfromtxt(
        csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    returns = np.asarray(raw["log_return"], dtype=np.float64)
    returns = (returns - returns.mean()) / returns.std()
    tail_threshold = float(np.quantile(np.abs(returns), 1.0 - tail_quantile))

    samples = returns.astype(np.float32).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    n_samples = samples.shape[0]
    n_test = int(round(test_fraction * n_samples))
    n_val = int(round(val_fraction * n_samples))

    return SP500Dataset(
        train=torch.from_numpy(samples[n_test + n_val :]),
        val=torch.from_numpy(samples[n_test : n_test + n_val]),
        test=torch.from_numpy(samples[:n_test]),
        tail_threshold=tail_threshold,
        csv_path=str(ds_cfg["csv_path"]),
        n_samples=n_samples,
        tail_quantile=tail_quantile,
    )


def build_dataset(cfg: dict, seed: int) -> GMMDataset | SP500Dataset:
    """Dispatch on ``dataset.name`` (``gmm3`` default, ``sp500``)."""

    name = str(cfg["dataset"].get("name", "gmm3"))
    if name == "gmm3":
        return build_gmm_dataset(cfg, seed)
    if name == "sp500":
        return build_sp500_dataset(cfg, seed)
    raise ValueError(f"Unknown dataset name '{name}' (expected gmm3 or sp500)")


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
