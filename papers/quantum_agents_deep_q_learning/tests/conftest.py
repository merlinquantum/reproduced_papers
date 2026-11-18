"""Make ``lib`` (paper-local) and ``runtime_lib`` (repository) importable in tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
for path in (str(PROJECT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
