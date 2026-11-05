"""Library package for paper-specific modules."""

from __future__ import annotations

import sys
from pathlib import Path

_PAPERS_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (_PAPERS_DIR, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from runtime_lib import config as _shared_config  # noqa: E402

sys.modules[__name__ + ".config"] = _shared_config
config = _shared_config

__all__ = ["config"]
