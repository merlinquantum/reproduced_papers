from __future__ import annotations

import json

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_lists_paper_flags(capsys):
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for flag in (
        "--ticker",
        "--feature-timing",
        "--regime",
        "--split-mode",
        "--vol-ticker",
        "--num-threads",
    ):
        assert flag in help_text


def test_defaults_run_writes_completed_evidence(tmp_path):
    from lib import runner as benchmark_runner

    cfg = load_runtime_ready_config()
    cfg["data_root"] = str(_repo_data_root())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_snapshot.json").write_text(
        json.dumps({"stand_in": True}), encoding="utf-8"
    )

    benchmark_runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert {"leak_free:qnn", "leak_free:ann", "paper:qnn", "majority_class"} <= set(
        metrics["test"]
    )
    assert metrics["test_window"][0].startswith("2020-01")


def _repo_data_root():
    from common import REPO_ROOT

    return REPO_ROOT / "data"
