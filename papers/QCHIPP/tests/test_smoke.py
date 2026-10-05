from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

from runtime_lib import run_from_project


def test_real_shared_runtime_smoke(tmp_path):
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            ["--epochs", "1", "--outdir", str(tmp_path)],
        )
    finally:
        os.chdir(original_cwd)

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "run_status.json").exists()
    assert not (run_dir / "done.txt").exists()
