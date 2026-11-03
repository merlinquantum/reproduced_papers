"""Semeion data pipeline for the vertical QFL reproduction.

Mirrors the authors' ``create_data_setting.ipynb``: pixels are scaled to
``[0, pi/4]``, the 256 pixel columns are permuted with a seeded ``torch.randperm``
(this is the paper's "random feature distribution"; consecutive 32-column slices
of the permuted vector then go to the clients), and a stratified 80/20
train/test split is drawn with scikit-learn.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from runtime_lib.data_paths import resolve_data_root

logger = logging.getLogger(__name__)

N_PIXELS = 256
N_CLASSES = 10
PIXEL_MAX_ANGLE = math.pi / 4
DEFAULT_TEST_SIZE = 0.2


def semeion_file(cfg: dict[str, Any]) -> Path:
    """Locate ``semeion.data`` under the shared data root (``--data-root`` / ``DATA_DIR``)."""
    dataset_cfg = cfg["dataset"]
    root = resolve_data_root(cfg.get("data_root"))
    path = root / dataset_cfg["subdir"] / dataset_cfg["file"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Semeion file not found at {path}. See LOG.md (Data Acquisition Log) for the "
            "download command, or pass --data-root."
        )
    return path


def load_semeion(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pixels, labels)`` with pixels as binary 0/1 floats, shape ``(1593, 256)``."""
    table = np.loadtxt(path)
    if table.shape != (1593, N_PIXELS + N_CLASSES):
        raise ValueError(f"Unexpected Semeion shape {table.shape} in {path}")
    pixels = torch.from_numpy(table[:, :N_PIXELS]).double()
    labels = torch.from_numpy(table[:, N_PIXELS:].argmax(axis=1)).long()
    return pixels, labels


def build_splits(cfg: dict[str, Any]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Build permuted, angle-encoded, stratified train and test tensors.

    Returns
    -------
    dict
        ``{"train": (pixels, labels), "test": (pixels, labels)}`` with pixels as
        float64 angles in ``{0, pi/4}``; the model casts to its own dtype.
    """
    dataset_cfg = cfg["dataset"]
    pixels, labels = load_semeion(semeion_file(cfg))
    pixels = pixels * PIXEL_MAX_ANGLE

    # The authors seed the global generator and then call randperm.
    torch.manual_seed(dataset_cfg["permutation_seed"])
    permutation = torch.randperm(N_PIXELS)
    pixels = pixels[:, permutation]

    train_x, test_x, train_y, test_y = train_test_split(
        pixels,
        labels,
        test_size=dataset_cfg.get("test_size", DEFAULT_TEST_SIZE),
        stratify=labels,
        random_state=dataset_cfg["partition_seed"],
    )
    splits = {"train": (train_x, train_y), "test": (test_x, test_y)}
    validation_fraction = dataset_cfg.get("validation_fraction")
    if validation_fraction:
        # Held out from the training split for hyperparameter selection; the test split is
        # untouched, so a selection run never sees test labels.
        fit_x, validation_x, fit_y, validation_y = train_test_split(
            train_x,
            train_y,
            test_size=validation_fraction,
            stratify=train_y,
            random_state=dataset_cfg["partition_seed"],
        )
        splits["train"] = (fit_x, fit_y)
        splits["validation"] = (validation_x, validation_y)
        train_x, train_y = fit_x, fit_y
    # Reduced runs (smoke tests) may cap the sample counts; the cap is recorded in the config.
    max_train = dataset_cfg.get("max_train_samples")
    max_test = dataset_cfg.get("max_test_samples")
    if max_train is not None:
        splits["train"] = (train_x[:max_train], train_y[:max_train])
    if max_test is not None:
        splits["test"] = (test_x[:max_test], test_y[:max_test])
    logger.info(
        "Semeion split built: train=%d test=%d permutation_seed=%d partition_seed=%d",
        len(splits["train"][1]),
        len(splits["test"][1]),
        dataset_cfg["permutation_seed"],
        dataset_cfg["partition_seed"],
    )
    return splits


def build_dataloader(
    cfg: dict[str, Any],
    split: str,
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> DataLoader:
    """Batches of ``(pixels, labels)`` for ``split`` in ``{"train", "test"}``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    split : str
        ``"train"`` (shuffled with a generator seeded from ``cfg["seed"]``) or ``"test"``.
    splits : dict, optional
        Pre-built output of :func:`build_splits`, to avoid rebuilding per split.
    """
    if split not in ("train", "validation", "test"):
        raise ValueError(
            f"split must be 'train', 'validation' or 'test', got {split!r}"
        )
    splits = splits if splits is not None else build_splits(cfg)
    pixels, labels = splits[split]
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    return DataLoader(
        TensorDataset(pixels, labels),
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=(split == "train"),
        generator=generator,
    )
