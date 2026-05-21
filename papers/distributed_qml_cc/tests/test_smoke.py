from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

from runtime_lib import run_from_project


def test_quantum_smoke_run(tmp_path):
    """Run a tiny CC L=2, 10-iter, 64-batch quantum pipeline end-to-end."""
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--scheme", "cc",
                "--n-layers", "2",
                "--iterations", "10",
                "--batch-size", "64",
                "--outdir", str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)
    assert (run_dir / "config_snapshot.json").exists()
    assert (run_dir / "metrics.json").exists()
