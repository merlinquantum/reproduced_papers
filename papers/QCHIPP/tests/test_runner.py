"""Real-data runs through the runner and the shared runtime."""

import json
import os
from pathlib import Path

import pytest
from common import MODEL_SETTINGS, PROJECT_DIR, make_config
from lib.runner import train_and_evaluate

from runtime_lib import run_from_project


def test_logistic_run_through_runtime_writes_metrics_status_and_log(tmp_path):
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(PROJECT_DIR, ["--outdir", str(tmp_path)])
    finally:
        os.chdir(original_cwd)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    status = json.loads((run_dir / "run_status.json").read_text())
    assert metrics["test_count"] == 999
    assert 0 <= metrics["test"]["f1"] <= 1
    assert status["status"] == "COMPLETED"
    assert status["dataset"]["category_codes"]
    log = (run_dir / "run.log").read_text()
    assert log.count("RUN_STARTED") == 1
    assert log.count("RUN_COMPLETED") == 1
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "config_snapshot.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "merlin",
        "classical",
        "gate_source",
        "gate_corrected",
        "merlin_reference",
        "classical_reference",
    ],
)
def test_trainable_model_runner_persists_checkpoint_and_real_evidence(name, tmp_path):
    train_and_evaluate(make_config(**MODEL_SETTINGS[name]), tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["training"]["updates"] == 1
    assert metrics["parameters"]["total"] > 0
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "losses.csv").exists()


@pytest.mark.parametrize("name", ["logistic_onehot", "rf_onehot"])
def test_onehot_classical_runner_uses_same_population(name, tmp_path):
    train_and_evaluate(make_config(**MODEL_SETTINGS[name]), tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["test_count"] == 999
    assert metrics["input_features"] == 80
    if name == "logistic_onehot":
        assert metrics["parameters"]["total"] == 81


def test_failed_run_records_failure_status(tmp_path):
    cfg = make_config(name="classical")  # classical models require model.hidden
    with pytest.raises(KeyError):
        train_and_evaluate(cfg, tmp_path)
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "FAILED"
    assert status["error"]["type"] == "KeyError"
