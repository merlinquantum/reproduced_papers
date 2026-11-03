"""Library package for the Fourier Fingerprints reproduction."""

from __future__ import annotations

import sys
from pathlib import Path

# papers/<name>/lib/__init__.py -> parents[3] is the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime_lib import config as _shared_config  # noqa: E402

sys.modules[__name__ + ".config"] = _shared_config
config = _shared_config

__all__ = ["config"]
