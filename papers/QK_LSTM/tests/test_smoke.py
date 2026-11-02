from __future__ import annotations

import json
import os
from pathlib import Path

from common import PROJECT_DIR

from runtime_lib import run_from_project


def test_reduced_qklstm_run_through_shared_runtime(tmp_path):
    """One-epoch Q-B run on 16 training windows, end to end via the runtime."""
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--config",
                "configs/aqi_qklstm_count209_reduced.json",
                "--max-train-windows",
                "16",
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)

    for name in (
        "run.log",
        "run_status.json",
        "config_snapshot.json",
        "metrics.json",
        "predictions.csv",
        "model_state.pt",
    ):
        assert (run_dir / name).exists(), name
    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert metrics["trainable_parameters"] == 209
    assert len(metrics["history"]) == 1
    log = (run_dir / "run.log").read_text()
    for event in (
        "RUN_STARTED",
        "DATASET_READY",
        "TRAINING_STARTED",
        "TRAIN_EPOCH_COMPLETED",
        "EVALUATION_COMPLETED",
        "METRICS_WRITTEN",
        "RUN_COMPLETED",
    ):
        assert event in log, event
