from __future__ import annotations

import subprocess
import sys
from pathlib import Path

QVT_DIR = Path(__file__).resolve().parents[1]


def test_list_models_includes_paper_baselines() -> None:
    result = subprocess.run(
        [sys.executable, str(QVT_DIR / "implementation.py"), "--list-models"],
        cwd=str(QVT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "VisionTransformer" in result.stdout
    assert "OrthoFNN" in result.stdout
