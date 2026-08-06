"""Dataset loading for the BVE photonic dual-rail QNN reproduction.

The dataset (``sem_supervised_dataset.npz``) contains supervised pairs
``(t, x, y, z) -> psi`` sampled from a reference Spectral Element Method
(SEM) solution of the Barotropic Vorticity Equation, as described in
``main.tex`` Section IV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from runtime_lib.data_paths import paper_data_dir

PAPER_NAME = "bve_qnn"
DEFAULT_DATASET_FILENAME = "sem_supervised_dataset.npz"


def resolve_dataset_path(cfg: dict[str, Any]) -> Path:
    dataset_cfg = cfg.get("dataset", {})
    filename = dataset_cfg.get("filename", DEFAULT_DATASET_FILENAME)

    explicit_root = dataset_cfg.get("root")
    if explicit_root:
        candidate = Path(explicit_root) / filename
        if candidate.exists():
            return candidate

    data_dir = paper_data_dir(PAPER_NAME, data_root=cfg.get("data_root"))
    return data_dir / filename


def load_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the BVE dataset and return tensors plus raw reference arrays."""

    dataset_path = resolve_dataset_path(cfg)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"BVE dataset not found at {dataset_path}. Place "
            f"'{DEFAULT_DATASET_FILENAME}' under data/{PAPER_NAME}/ "
            "(see README.md for details)."
        )

    data = np.load(dataset_path)

    supervised_features = data["supervised_features"]
    supervised_targets = data["supervised_targets"]

    features_tensor = torch.tensor(supervised_features, dtype=torch.float64)
    targets_tensor = torch.tensor(supervised_targets, dtype=torch.float64)

    batch_size = int(cfg.get("dataset", {}).get("batch_size", len(features_tensor)))
    dataset = TensorDataset(features_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return {
        "features_tensor": features_tensor,
        "targets_tensor": targets_tensor,
        "dataloader": dataloader,
        "training_hours": data["training_hours"],
        "psi_qcl_training": data["psi_qcl_training"],
        "lat_downsampled": data["lat_downsampled"],
        "lon_downsampled": data["lon_downsampled"],
        "psi_shape": data["psi_qcl_training"].shape,
    }


__all__ = ["load_dataset", "resolve_dataset_path", "PAPER_NAME"]
