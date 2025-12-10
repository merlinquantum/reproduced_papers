from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

from runtime_lib import run_from_project


def test_runtime_smoke(monkeypatch, tmp_path):
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--epochs",
                "1",
                "--outdir",
                str(tmp_path),
                "--batch-size",
                "4",
                "--sequence-length",
                "4",
            ],
        )
    finally:
        os.chdir(original_cwd)

    assert (run_dir / "done.txt").exists()
    assert (run_dir / "metrics.json").exists()
