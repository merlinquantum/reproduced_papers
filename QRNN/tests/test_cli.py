from __future__ import annotations

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_artifact(tmp_path):
    from lib import runner

    cfg = load_runtime_ready_config()
    cfg["training"]["epochs"] = 1
    cfg["dataset"]["path"] = str(PROJECT_DIR / "data" / "sample_weather.csv")

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    runner.train_and_evaluate(cfg, run_dir)

    assert (run_dir / "done.txt").exists(), "Expected completion marker to be created"
    assert (run_dir / "metrics.json").exists(), "Expected metrics to be saved"
