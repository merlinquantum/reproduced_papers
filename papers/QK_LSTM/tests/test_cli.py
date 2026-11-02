from __future__ import annotations

import json

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("model_type", ["persistence", "ridge"])
def test_closed_form_baselines_write_evidence(tmp_path, model_type):
    from lib import runner as qk_runner

    cfg = load_runtime_ready_config()
    cfg["model"] = {
        "type": model_type,
        "params": {"alpha": 1.0} if model_type == "ridge" else {},
    }
    run_dir = tmp_path / "run_closed_form"
    metrics = qk_runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert metrics["splits"]["test"]["observed_targets"]["n"] == 402
    assert (run_dir / "predictions.csv").exists()
