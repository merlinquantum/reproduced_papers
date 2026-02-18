"""Dataset generators for QLSTM reproduction (delegates to shared module)."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from papers.shared.QLSTM.dataset import data  # type: ignore
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from papers.shared.QLSTM.dataset import data  # type: ignore

__all__ = ["data"]
