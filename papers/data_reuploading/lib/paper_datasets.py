"""Thin importer delegating dataset helpers to the shared module."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papers.shared.data_reuploading.paper_datasets import (  # noqa: E402
    CirclesDataset,
    MoonsDataset,
    OverheadMNISTDataset,
    PaperDataset,
    TetrominoDataset,
)

__all__ = [
    "PaperDataset",
    "CirclesDataset",
    "MoonsDataset",
    "TetrominoDataset",
    "OverheadMNISTDataset",
]
