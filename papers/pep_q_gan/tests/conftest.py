"""Resolve both a flat workspace and an upstream papers/<name> checkout."""

import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ROOT = next(parent for parent in PAPER.parents if (parent / "runtime_lib").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAPER.parent))
