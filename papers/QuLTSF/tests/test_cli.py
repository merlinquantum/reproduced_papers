from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_artifacts(tmp_path):
    from papers.QuLTSF.lib import runner

    cfg = load_runtime_ready_config()
    cfg["training"]["epochs"] = 1
    cfg["dataset"]["path"] = str(tmp_path / "weather.csv")
    cfg["dataset"]["sequence_length"] = 3
    cfg["dataset"]["prediction_horizon"] = 2
    cfg["dataset"]["batch_size"] = 2
    cfg["dataset"]["features_mode"] = "MS"
    cfg["dataset"]["target"] = "OT"

    (tmp_path / "weather.csv").write_text(
        "date,p (mbar),T (degC),rh (%),OT\n"
        "2020-01-01 00:10:00,1008.89,0.71,86.1,428.1\n"
        "2020-01-01 00:20:00,1008.76,0.75,85.2,428.0\n"
        "2020-01-01 00:30:00,1008.62,0.49,87.0,427.6\n"
        "2020-01-01 00:40:00,1008.49,0.36,87.2,427.1\n"
        "2020-01-01 00:50:00,1008.37,0.31,87.4,426.7\n"
        "2020-01-01 01:00:00,1008.24,0.32,87.9,426.5\n"
        "2020-01-01 01:10:00,1008.12,0.15,88.9,426.2\n"
        "2020-01-01 01:20:00,1008.01,0.08,89.4,426.0\n"
        "2020-01-01 01:30:00,1007.90,-0.03,90.5,425.8\n"
        "2020-01-01 01:40:00,1007.79,-0.11,91.0,425.7\n"
        "2020-01-01 01:50:00,1007.67,-0.15,91.3,425.6\n"
        "2020-01-01 02:00:00,1007.56,-0.20,91.8,425.4\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    runner.train_and_evaluate(cfg, run_dir)

    assert (run_dir / "done.txt").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.json").exists()
    assert (run_dir / "metadata.json").exists()
