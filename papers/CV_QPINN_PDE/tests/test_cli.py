from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_artifact(tmp_path):
    from CV_QPINN_PDE.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["training"]["epochs"] = 2
    cfg["training"]["collocation_points"] = 8
    cfg["model"]["cutoff"] = 5
    cfg["model"]["n_multi_layers"] = 1
    cfg["model"]["n_single_layers"] = 1
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    summary = paper_runner.train_and_evaluate(cfg, run_dir)
    assert (run_dir / "summary.json").exists()
    assert "metrics" in summary
