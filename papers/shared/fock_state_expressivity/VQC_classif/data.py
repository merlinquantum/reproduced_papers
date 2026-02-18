"""Shared dataset generation utilities for VQC_classif."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from runtime_lib.data_paths import paper_data_dir
except ModuleNotFoundError:
    _REPO_ROOT = next(
        (p for p in Path(__file__).resolve().parents if (p / "runtime_lib").exists()),
        None,
    )
    if _REPO_ROOT and str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from runtime_lib.data_paths import paper_data_dir

DATASET_ORDER = ("linear", "circular", "moon")


@dataclass
class DatasetSplit:
    x_train: torch.Tensor
    x_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor

    @property
    def full_x(self) -> torch.Tensor:
        return torch.cat((self.x_train, self.x_test))

    @property
    def full_y(self) -> torch.Tensor:
        return torch.cat((self.y_train, self.y_test))

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "x_train": self.x_train,
            "x_test": self.x_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "full_x": self.full_x,
            "full_y": self.full_y,
        }


def _get_linear(num_samples: int, num_features: int, class_sep: float, seed: int):
    x, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=2,
        random_state=seed,
        class_sep=class_sep,
    )
    return x, y


def _get_circle(num_samples: int, noise: float, seed: int):
    x, y = make_circles(n_samples=num_samples, noise=noise, random_state=seed)
    return x, y


def _get_moon(num_samples: int, noise: float, seed: int):
    x, y = make_moons(n_samples=num_samples, noise=noise, random_state=seed)
    return x, y


GENERATOR_MAP = {
    "linear": _get_linear,
    "circular": _get_circle,
    "moon": _get_moon,
}


def _subsample(x: np.ndarray, y: np.ndarray, ratio: float, seed: int):
    if ratio >= 1.0:
        return x, y
    keep = max(1, min(len(x) - 1, int(len(x) * ratio)))
    x_sel, _, y_sel, _ = train_test_split(
        x, y, train_size=keep, random_state=seed, stratify=y
    )
    return x_sel, y_sel


def _build_split(
    name: str,
    spec: dict,
    global_cfg: dict,
    seed: int,
) -> DatasetSplit:
    generator = GENERATOR_MAP[name]
    num_samples = int(spec.get("num_samples", 200))
    class_sep = float(spec.get("class_sep", 1.5))
    noise = float(spec.get("noise", 0.1))
    num_features = int(spec.get("num_features", 2))

    if name == "linear":
        x, y = generator(num_samples, num_features, class_sep, seed)
    else:
        x, y = generator(num_samples, noise, seed)

    subsample_ratio = float(
        spec.get("subsample_ratio", global_cfg.get("subsample_ratio", 1.0))
    )
    x, y = _subsample(x, y, subsample_ratio, seed)

    test_size = float(global_cfg.get("test_size", 0.4))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    if global_cfg.get("standardize", True):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    return DatasetSplit(x_train_t, x_test_t, y_train_t, y_test_t)


def prepare_datasets(cfg: dict) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load cached datasets if available or generate synthetic ones.

    Args:
        cfg (Dict): Configuration dictionary for data handling.

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Train/test splits keyed by dataset name.
    """
    raw_cache = cfg.get("cache_dir")
    if raw_cache:
        cache_dir = Path(raw_cache)
        if not cache_dir.is_absolute():
            cache_dir = (
                paper_data_dir("fock_state_expressivity") / "VQC_classif" / cache_dir
            )
    else:
        cache_dir = paper_data_dir("fock_state_expressivity") / "VQC_classif" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    force_regen = bool(cfg.get("force_regenerate", False))
    dataset_cfg = cfg.get("datasets", {})
    seed = int(cfg.get("seed", 42))

    datasets: dict[str, dict[str, torch.Tensor]] = {}
    for name in cfg.get("order", DATASET_ORDER):
        spec = dataset_cfg.get(name, {})
        cache_path = cache_dir / f"{name}_dataset.pt"
        if cache_path.exists() and not force_regen:
            payload = torch.load(cache_path)
            datasets[name] = payload
            continue

        split = _build_split(name, spec, cfg, seed)
        payload = split.to_dict()
        torch.save(payload, cache_path)
        datasets[name] = payload

    return datasets


__all__ = [
    "DatasetSplit",
    "DATASET_ORDER",
    "prepare_datasets",
]
