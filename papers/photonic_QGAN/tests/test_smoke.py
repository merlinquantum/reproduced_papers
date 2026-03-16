from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

from runtime_lib import run_from_project


def test_run_from_project_smoke(tmp_path):
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            ["--outdir", str(tmp_path)],
        )
    finally:
        os.chdir(original_cwd)

    assert (run_dir / "done.txt").exists()
