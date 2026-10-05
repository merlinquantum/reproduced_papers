from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config
from lib import runner as qchipp_runner


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_artifact(tmp_path):
    cfg = load_runtime_ready_config()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    qchipp_runner.train_and_evaluate(cfg, run_dir)

    assert (run_dir / "metrics.json").exists(), "Expected real metric evidence"
