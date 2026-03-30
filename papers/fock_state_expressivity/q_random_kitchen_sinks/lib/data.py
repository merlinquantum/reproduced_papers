"""Thin importer delegating dataset helpers to the shared module."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papers.shared.fock_state_expressivity.q_random_kitchen_sinks.data import (  # noqa: E402
    load_moons,
    target_function,
)

__all__ = ["load_moons", "target_function"]
