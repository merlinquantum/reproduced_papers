"""Thin importer to the shared photonic_QCNN dataset utilities."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papers.shared.photonic_QCNN.data import (  # noqa: E402
    convert_dataset_to_tensor,
    convert_scalar_labels_to_onehot,
    convert_tensor_to_loader,
    get_dataset,
    get_dataset_description,
    save_dataset_description,
)

__all__ = [
    "get_dataset",
    "get_dataset_description",
    "save_dataset_description",
    "convert_dataset_to_tensor",
    "convert_tensor_to_loader",
    "convert_scalar_labels_to_onehot",
]
