"""CLI schema and an end-to-end run of the default (synthetic) config."""

from __future__ import annotations

import json

import numpy as np
import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_named_configs_are_valid_json():
    for path in sorted((PROJECT_DIR / "configs").glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        assert "description" in config, path.name


@pytest.mark.parametrize("model_name", ["qultsf", "linear", "persistence"])
def test_train_and_evaluate_writes_evidence(tmp_path, model_name):
    from papers.QuLTSF.lib import runner

    cfg = load_runtime_ready_config()
    cfg["model"]["name"] = model_name
    cfg["training"]["epochs"] = 1
    cfg["training"]["max_batches_per_epoch"] = 3
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_snapshot.json").write_text(
        json.dumps({"test": True}), encoding="utf-8"
    )

    runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETED"
    assert status["metrics_path"] == "metrics.json"
    assert np.isfinite(metrics["test"]["mse"])
    errors = np.load(run_dir / "test_errors.npz")
    assert errors["squared"].shape == (metrics["test"]["n_windows"],)
    assert (run_dir / "best_model.pt").exists() == (model_name != "persistence")
