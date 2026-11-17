from __future__ import annotations

import json
from pathlib import Path

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config

from runtime_lib.config import deep_update, load_config
from runtime_lib.dtypes import resolve_config_dtypes


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def _config_with(overlay: str) -> dict:
    cfg = deep_update(
        load_config(PROJECT_DIR / "configs" / "defaults.json"),
        load_config(PROJECT_DIR / "configs" / overlay),
    )
    cfg.setdefault("logging", {"level": "info"})
    return resolve_config_dtypes(cfg)


def test_classification_smoke_writes_evidence(tmp_path: Path):
    from lib import runner as abs_runner

    cfg = _config_with("smoke.json")
    abs_runner.train_and_evaluate(cfg, tmp_path)

    for name in ("run.log", "run_status.json", "metrics.json", "summary.csv"):
        assert (tmp_path / name).exists(), name
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    for arm in ("paper", "tuned"):
        aggregate = metrics["results"]["rbf_svm_x"]["arms"][arm]["aggregate"]
        assert aggregate["n_instances"] == 1
        assert 0.0 <= aggregate["mean_test_accuracy_over_instances"] <= 1.0


def test_platform_a_run_meets_printed_tolerance(tmp_path: Path):
    from lib import runner as abs_runner

    cfg = _config_with("kernel_platform_a_original.json")
    abs_runner.train_and_evaluate(cfg, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["instances"][0]["printed_entries"]["within_tolerance"]


def test_invalid_device_fails_and_records_status(tmp_path: Path):
    from lib import runner as abs_runner

    cfg = load_runtime_ready_config()
    cfg["device"] = "cuda:0"
    with pytest.raises(ValueError, match="unsupported"):
        abs_runner.train_and_evaluate(cfg, tmp_path)
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "FAILED"
    assert status["error"]["type"] == "ValueError"
