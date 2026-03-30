"""Thin importer delegating dataset helpers to the shared module."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papers.shared.fock_state_expressivity.VQC_classif.data import (  # noqa E402
    DATASET_ORDER,
    DatasetSplit,
    prepare_datasets,
)

__all__ = ["DatasetSplit", "DATASET_ORDER", "prepare_datasets"]
