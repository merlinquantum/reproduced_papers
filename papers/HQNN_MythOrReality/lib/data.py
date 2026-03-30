"""Thin importer for shared HQNN spiral dataset utilities."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from papers.shared.HQNN_MythOrReality.data import (  # type: ignore
        SpiralDatasetConfig,
        load_spiral_dataset,
    )
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from papers.shared.HQNN_MythOrReality.data import (  # type: ignore
        SpiralDatasetConfig,
        load_spiral_dataset,
    )

__all__ = ["SpiralDatasetConfig", "load_spiral_dataset"]
