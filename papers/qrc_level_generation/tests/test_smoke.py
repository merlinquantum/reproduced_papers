from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

from runtime_lib import run_from_project


def test_smoke_reference_only_run(tmp_path):
    """End-to-end smoke test in reference-only mode.

    Skips the QRC training to keep the test fast; still exercises metric
    computation against the published Aer T=1 sequences and the Markov +
    uncorrelated baselines.
    """
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--config",
                "configs/reference_eval.json",
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "summary.json").exists()
