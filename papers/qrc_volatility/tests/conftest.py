"""Put the paper directory on ``sys.path`` so ``lib.*`` imports resolve.

Tests may be run from the paper folder (``pytest -q``) or from the repository
root (``pytest papers/qrc_volatility``); both need the paper directory itself on
the path, and ``lib/__init__.py`` adds the repository root for ``runtime_lib``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
