"""Make the paper and shared runtime importable in flat or catalogue layouts."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = next(path for path in PROJECT.parents if (path / "implementation.py").is_file())
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT))
