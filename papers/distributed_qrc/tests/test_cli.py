from __future__ import annotations

import json

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_tiny_grid_writes_evidence(tmp_path):
    """End-to-end: 4-neuron SRSR, all 9 cells, kernels on a few hundred points."""

    from distributed_qrc.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["experiment"]["sizes"] = [4]
    cfg["readout"].update(kernel_selection_subsample=200, kernel_fit_subsample=300)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    paper_runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert len(metrics["cells"]) == 9
    for cell in metrics["cells"]:
        assert 0.0 < cell["metrics"]["test"]["rmse"] < 1.0
    assert (run_dir / "test_predictions.npz").exists()
