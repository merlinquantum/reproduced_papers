"""Thin importer delegating Fourier dataset helpers to the shared module."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papers.shared.fock_state_expressivity.VQC_fourier_series.data import (  # noqa: E402
    FourierCoefficient,
    generate_dataset,
)

__all__ = ["FourierCoefficient", "generate_dataset"]
