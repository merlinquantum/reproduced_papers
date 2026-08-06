from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

from runtime_lib import run_from_project


def test_runtime_smoke(tmp_path):
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--total-steps",
                "2",
                "--batch-size",
                "32",
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)

    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "exp1_merlin_results.npz").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "done.txt").exists()
