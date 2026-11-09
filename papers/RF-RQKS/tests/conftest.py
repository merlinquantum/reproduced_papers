"""Configure paper-local imports for RF-RQKS tests."""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))
