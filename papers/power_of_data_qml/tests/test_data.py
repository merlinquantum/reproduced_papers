"""One-batch check of the real Fashion-MNIST loader (Phase 2 data-loader gate)."""

from __future__ import annotations

from pathlib import Path

import pytest
from common import REPO_ROOT, load_project_defaults
from lib.data import build_dataloader

DATA_ROOT = Path(REPO_ROOT) / "data"


@pytest.mark.skipif(
    not (DATA_ROOT / "power_of_data_qml" / "fashion_mnist").is_dir(),
    reason="Fashion-MNIST not acquired",
)
def test_build_dataloader_returns_expected_batch():
    cfg = load_project_defaults()
    cfg["data_root"] = str(DATA_ROOT)
    loader = build_dataloader(cfg, split="train")
    features, labels = next(iter(loader))
    expected_shape = (cfg["dataset"]["batch_size"], cfg["study"]["system_sizes"][0])
    assert tuple(features.shape) == expected_shape, features.shape
    assert tuple(labels.shape) == (cfg["dataset"]["batch_size"],)
    assert len(loader.dataset) == cfg["study"]["quantum_dataset_train_size"]
    test_loader = build_dataloader(cfg, split="test")
    assert len(test_loader.dataset) == (
        cfg["study"]["subsample_size"] - cfg["study"]["quantum_dataset_train_size"]
    )
