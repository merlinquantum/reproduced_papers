"""Thin importer delegating dataset utilities to the shared module."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papers.shared.fock_state_expressivity.q_gaussian_kernel.data import (  # noqa: E402
    GaussianGrid,
    build_gaussian_grid,
    prepare_classification_data,
)

__all__ = ["GaussianGrid", "build_gaussian_grid", "prepare_classification_data"]
