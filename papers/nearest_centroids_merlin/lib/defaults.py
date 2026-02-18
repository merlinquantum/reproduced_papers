"""Default configuration loader for nearest_centroids_merlin.

This module provides access to the default configuration from configs/defaults.json.
"""

from __future__ import annotations

import json
from pathlib import Path


def _defaults_path() -> Path:
    """Return path to the defaults.json config file."""
    return Path(__file__).resolve().parents[1] / "configs" / "defaults.json"


def default_config() -> dict[str, object]:
    """Load and return the default configuration dictionary."""
    with _defaults_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)
