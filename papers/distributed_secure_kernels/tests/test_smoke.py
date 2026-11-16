from __future__ import annotations

import json
import os
from pathlib import Path

from common import PROJECT_DIR

from runtime_lib import run_from_project


def test_shared_runtime_runs_the_smoke_config(tmp_path):
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            ["--config", "configs/defaults.json", "--outdir", str(tmp_path)],
        )
    finally:
        os.chdir(original_cwd)
    status = json.loads((run_dir / "run_status.json").read_text())
    methods = json.loads((run_dir / "metrics.json").read_text())["methods"]
    assert status["status"] == "COMPLETED"
    # Wine, paper protocol: the classical linear SVM is well above chance (3 classes).
    assert methods["classical_linear"]["kernel_svm"]["accuracy"]["mean"] > 0.9
