"""Make ``runtime_lib`` and the ``photonic_MMD_generative`` package importable in every test."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR.parents[1], PROJECT_DIR.parent):  # repository root, papers/
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
