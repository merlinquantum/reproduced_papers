from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config
from lib.data import dataset_dir


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_defaults_match_paper_setup():
    cfg = load_runtime_ready_config()
    assert cfg["experiment"]["kind"] == "qstl"
    assert cfg["dataset"]["tickers"] == ["AAPL"]
    assert cfg["dataset"]["context_length"] == 3
    assert cfg["model"]["layers"] == 4
    assert cfg["training"]["epochs"] == 3000
    assert cfg["training"]["lr"] == 0.1
    assert cfg["training"]["spsa_perturbation"] == 0.01
    assert cfg["training"]["shots"] == 10000


def test_named_configs_are_valid_json_with_descriptions():
    for path in sorted((PROJECT_DIR / "configs").glob("*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert cfg.get("description"), path.name


def test_reduced_run_writes_evidence(tmp_path):
    cfg = load_runtime_ready_config()
    try:
        dataset_dir(cfg)
    except FileNotFoundError:
        pytest.skip("substitute price data not present under data/QNN_STOCK")
    from lib import runner

    cfg["training"]["epochs"] = 3
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETED"
    assert metrics["assets"]["AAPL"]["models"]["QSTL"]["n_params"] == 32
    assert (run_dir / "losses.csv").exists()
