from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def _tiny_config():
    cfg = load_runtime_ready_config()
    cfg["data"]["num_windows"] = 150
    cfg["data"]["tasks"] = ["exp_fading"]
    cfg["readout"]["tuning"]["xi_maxiter"] = 5
    cfg["readout"]["tuning"]["nu_grid"] = [2.5]
    cfg["readout"]["train_sizes"] = [60, 120]
    cfg["readout"]["slope_min_size"] = 60
    return cfg


@pytest.mark.parametrize("kind", ["quark", "raw_window_linear", "esn_kernel"])
def test_train_and_evaluate_writes_metrics(tmp_path, kind):
    from QuaRK.lib import runner as paper_runner

    cfg = _tiny_config()
    cfg["model"]["kind"] = kind
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, default=str))

    paper_runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    selections = metrics["tasks"]["exp_fading"]["selections"]
    assert set(selections) == {"validation", "test"}
    assert len(selections["validation"]["training_size_mse_test"]) == 2
    assert (kind == "quark") == ("bound" in selections["validation"])


def test_configs_load():
    for path in sorted((PROJECT_DIR / "configs").glob("*.json")):
        json.loads(path.read_text())
