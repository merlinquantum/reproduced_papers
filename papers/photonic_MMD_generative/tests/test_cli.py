from __future__ import annotations

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_evidence(tmp_path):
    import json

    from photonic_MMD_generative.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["data_root"] = str(tmp_path / "data")
    cfg["training"]["steps"] = 3
    cfg["evaluation"]["num_model_samples"] = 100
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    paper_runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert metrics["training"]["steps_run"] == 3
    assert set(metrics["sample_metrics"]) == {
        "model",
        "random",
        "marginal",
        "test_to_test",
    }
    for artifact in ("losses.csv", "params.npy", "unitary.npy", "samples.npz"):
        assert (run_dir / artifact).exists(), artifact


def test_non_unitary_model_is_marked_not_deployable(tmp_path):
    import json

    from photonic_MMD_generative.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["data_root"] = str(tmp_path / "data")
    cfg["dataset"].update(m=6, n=2)
    cfg["model"].update(ansatz="butterfly", butterfly_variant="reference")
    cfg["init"] = {"strategy": "random"}
    cfg["training"]["steps"] = 2
    cfg["evaluation"]["num_model_samples"] = 50
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    paper_runner.train_and_evaluate(cfg, run_dir)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert (
        json.loads((run_dir / "run_status.json").read_text())["status"] == "COMPLETED"
    )
    assert metrics["deployable"] is False and metrics["unitarity_error"] > 1e-8
    assert "model" not in metrics["sample_metrics"]
    assert "random" in metrics["sample_metrics"]
