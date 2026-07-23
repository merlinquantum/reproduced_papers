from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
QVT_DIR = THIS_DIR.parent
REPO_ROOT = QVT_DIR.parent.parent

# Paper dir first (lib, paper-local modules), then repo root (runtime_lib).
if str(QVT_DIR) not in sys.path:
    sys.path.insert(0, str(QVT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))


@pytest.fixture
def load_generate_figures_module():
    script_path = QVT_DIR / "scripts" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("qvt_generate_figures", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
