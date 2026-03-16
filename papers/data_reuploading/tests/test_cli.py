from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_artifact(tmp_path, monkeypatch):
    from lib import runner as dr_runner

    cfg = load_runtime_ready_config()
    markers: dict[str, bool] = {}

    def fake_reproduce(cfg_arg, run_dir):
        markers["train_eval"] = True
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "train_eval_results.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(dr_runner, "run_train_eval", fake_reproduce)

    dr_runner.train_and_evaluate(cfg, tmp_path)

    assert markers.get("train_eval") is True
    assert (tmp_path / "train_eval_results.json").exists()
