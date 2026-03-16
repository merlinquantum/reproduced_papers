"""Thin importer delegating QCNN PCA data prep to the shared module."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from papers.shared.QCNN_data_classification.data import make_pca  # type: ignore
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from papers.shared.QCNN_data_classification.data import make_pca  # type: ignore

__all__ = ["make_pca"]
