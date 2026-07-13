from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
QVT_DIR = THIS_DIR.parent

if str(QVT_DIR) not in sys.path:
    sys.path.insert(0, str(QVT_DIR))


@pytest.fixture
def load_generate_figures_module():
    script_path = QVT_DIR / "scripts" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("qvt_generate_figures", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
