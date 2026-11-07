from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from common import PROJECT_DIR, SEMEION_FILE

from runtime_lib import run_from_project


@pytest.mark.skipif(not SEMEION_FILE.exists(), reason="Semeion data not downloaded")
def test_reduced_run_writes_evidence(tmp_path):
    original = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            ["--config", "configs/defaults.json", "--outdir", str(tmp_path)],
        )
    finally:
        os.chdir(original)
    for name in ("run.log", "run_status.json", "config_snapshot.json", "metrics.json"):
        assert (run_dir / name).exists(), name
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert (
        len(
            json.loads((run_dir / "metrics.json").read_text())["history"][
                "test_accuracy"
            ]
        )
        == 2
    )
