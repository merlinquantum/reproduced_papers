from __future__ import annotations

import json

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_contract_artifacts(tmp_path):
    from photonic_trainability.lib import runner as photonic_trainability_runner

    cfg = load_runtime_ready_config()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    photonic_trainability_runner.train_and_evaluate(cfg, run_dir)

    for name in ("run.log", "run_status.json", "config_snapshot.json", "metrics.json"):
        assert (run_dir / name).exists(), f"Missing artifact {name}"
    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["metrics_path"] == "metrics.json"
    assert metrics["all_passed"] is True


def test_unknown_experiment_fails_clearly(tmp_path):
    from photonic_trainability.lib import runner as photonic_trainability_runner

    cfg = load_runtime_ready_config()
    cfg["experiment"] = "not_an_experiment"
    with pytest.raises(KeyError, match="Unknown experiment"):
        photonic_trainability_runner.train_and_evaluate(cfg, tmp_path)
