from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_artifact(tmp_path):
    from lib import runner

    cfg = load_runtime_ready_config()
    cfg["training"]["epochs"] = 1
    csv_path = tmp_path / "synthetic_weather.csv"
    csv_path.write_text(
        "timestamp,temperature,humidity\n"
        "2020-01-01 00:00:00,10.0,0.4\n"
        "2020-01-01 01:00:00,10.1,0.401\n"
        "2020-01-01 02:00:00,10.2,0.402\n"
        "2020-01-01 03:00:00,10.3,0.403\n"
        "2020-01-01 04:00:00,10.4,0.404\n",
        encoding="utf-8",
    )

    cfg["dataset"]["path"] = str(csv_path)
    cfg["dataset"]["kaggle_dataset"] = None
    cfg["dataset"]["preprocess"] = None
    cfg["dataset"]["target_column"] = "temperature"
    cfg["dataset"]["feature_columns"] = ["temperature", "humidity"]
    cfg["dataset"]["time_column"] = "timestamp"
    cfg["dataset"]["sequence_length"] = 2
    cfg["dataset"]["prediction_horizon"] = 1
    cfg["dataset"]["batch_size"] = 2

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    runner.train_and_evaluate(cfg, run_dir)

    assert (run_dir / "done.txt").exists(), "Expected completion marker to be created"
    assert (run_dir / "metrics.json").exists(), "Expected metrics to be saved"
    assert (run_dir / "predictions.csv").exists(), (
        "Expected predictions CSV to be saved"
    )
