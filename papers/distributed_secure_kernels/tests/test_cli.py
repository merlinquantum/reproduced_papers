from __future__ import annotations

import json

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_defaults_run_writes_logging_contract_artifacts(tmp_path):
    from distributed_secure_kernels.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paper_runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert set(metrics["methods"]) == {
        "classical_linear",
        "quantum_centralised",
        "quantum_distributed",
    }
    log = (run_dir / "run.log").read_text()
    for event in (
        "RUN_STARTED",
        "DATASET_READY",
        "TRAINING_STARTED",
        "EVALUATION_COMPLETED",
        "METRICS_WRITTEN",
        "RUN_COMPLETED",
    ):
        assert event in log
