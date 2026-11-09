from __future__ import annotations

import json

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_artifact(tmp_path):
    """Smoke test: a tiny end-to-end training run completes and writes metrics."""

    from QEGM_rare_events.lib import runner as qegm_runner

    cfg = load_runtime_ready_config()
    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = 32
    cfg["training"]["seeds"] = "0"
    cfg["training"]["models"] = "vae"
    cfg["dataset"]["n_samples"] = 64
    cfg["evaluation"]["n_generated"] = 64

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    qegm_runner.train_and_evaluate(cfg, run_dir)

    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert "summary" in payload
    assert "vae" in payload["summary"]
