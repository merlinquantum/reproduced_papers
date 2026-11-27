"""End-to-end runner tests: one tiny run per experiment kind, plus the failure path."""

from __future__ import annotations

import hashlib
import json

import pytest
from common import load_named_config
from lib import runner

TINY_OVERRIDES = {
    "e2_fig4_reduced.json": {"fit": {"training": {"steps": 3}}},
    "e1_fig3_classical.json": {"fit": {"seeds": [0], "training": {"steps": 3}}},
    "e3_fig5_original.json": {"coefficients": {"n_samples": 3}},
    "e4_spectrum_original.json": {
        "spectrum": {"random_draws": 2, "pauli_repetitions": [1, 2]}
    },
    "e2_fig4_merlin_reduced.json": {"fit": {"training": {"steps": 2}}},
    "e3_fig5_merlin.json": {"coefficients": {"n_samples": 2}},
    "e6_spectrum_merlin.json": {"photonic_spectrum": {"random_draws": 1}},
    "e8_fig4_improved_merlin.json": {"fit": {"seeds": [0], "training": {"steps": 2}}},
    "e8_fig4_trainable_observable.json": {
        "fit": {"seeds": [0], "training": {"steps": 2, "full_batch": True}}
    },
    "e9_universality_merlin.json": {"fit": {"seeds": [0], "training": {"steps": 2}}},
    "e11_multiscale_square_merlin.json": {
        "fit": {"seeds": [0], "training": {"steps": 2}}
    },
    "e11_multiscale_square_classical.json": {
        "fit": {"seeds": [0], "training": {"steps": 2, "optimizer": "lbfgs"}}
    },
    "e12_fixed_readout_ceiling_merlin.json": {
        "fit": {"seeds": [0], "training": {"steps": 2}}
    },
    "e12_fixed_readout_ceiling_qubit.json": {
        "fit": {"seeds": [0], "training": {"steps": 2}}
    },
    "e7_coefficient_reach_merlin.json": {
        "coefficient_reach": {"steps": 2, "init_seeds": [0]}
    },
    "e5_universality_original.json": {
        "universality": {
            "exact_degrees": [1, 2],
            "evaluation_points": 128,
            "square_wave_qubits": [1, 2, 3],
        }
    },
}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("config_name", sorted(TINY_OVERRIDES))
def test_runner_writes_complete_evidence(config_name, tmp_path):
    cfg = load_named_config(config_name, TINY_OVERRIDES[config_name])
    run_dir = tmp_path / "run_test"
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)

    status = _read_json(run_dir / "run_status.json")
    assert status["status"] == "COMPLETED"
    for key in ("log_path", "config_path", "metrics_path"):
        assert (run_dir / status[key]).exists(), key
    assert status["metrics_sha256"] == _sha256(run_dir / "metrics.json")
    assert status["config_sha256"] == _sha256(run_dir / "config_snapshot.json")

    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert log.count("RUN_STARTED") == 1 and log.count("RUN_COMPLETED") == 1
    assert log.count("DATASET_READY") == 1 and log.count("METRICS_WRITTEN") == 1
    assert "EVALUATION_COMPLETED" in log

    metrics = _read_json(run_dir / "metrics.json")
    assert metrics["experiment"] == cfg["experiment"]
    for artifact in metrics.get("artifacts", {}).values():
        assert (run_dir / artifact).exists(), artifact


def test_fit_run_logs_training_progress_and_reports_b1(tmp_path):
    cfg = load_named_config(
        "e2_fig4_reduced.json", TINY_OVERRIDES["e2_fig4_reduced.json"]
    )
    run_dir = tmp_path / "run_fit"
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert log.count("TRAINING_STARTED") == 1
    assert log.count("TRAIN_STEP_COMPLETED") == 3 * len(cfg["fit"]["cases"])
    metrics = _read_json(run_dir / "metrics.json")
    for run in metrics["runs"]:
        assert len(run["mse_history"]) == 4
        assert run["gap_to_b1"] == pytest.approx(run["final_mse"] - run["b1_mse"])


def test_spectrum_checks_pass_on_tiny_run(tmp_path):
    cfg = load_named_config(
        "e4_spectrum_original.json", TINY_OVERRIDES["e4_spectrum_original.json"]
    )
    run_dir = tmp_path / "run_spectrum"
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)
    metrics = _read_json(run_dir / "metrics.json")
    failed = [check["name"] for check in metrics["checks"] if not check["passed"]]
    assert metrics["all_passed"], failed


def test_failed_run_records_failure(tmp_path):
    cfg = load_named_config(
        "e2_fig4_reduced.json", {"fit": {"models": ["not_a_model"]}}
    )
    run_dir = tmp_path / "run_failed"
    run_dir.mkdir()
    with pytest.raises(ValueError):
        runner.train_and_evaluate(cfg, run_dir)
    status = _read_json(run_dir / "run_status.json")
    assert status["status"] == "FAILED"
    assert status["error"]["type"] == "ValueError"
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert log.count("RUN_FAILED") == 1 and "RUN_COMPLETED" not in log


def test_photonic_spectrum_checks_pass_on_tiny_run(tmp_path):
    cfg = load_named_config(
        "e6_spectrum_merlin.json", TINY_OVERRIDES["e6_spectrum_merlin.json"]
    )
    run_dir = tmp_path / "run_photonic_spectrum"
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)
    metrics = _read_json(run_dir / "metrics.json")
    failed = [check["name"] for check in metrics["checks"] if not check["passed"]]
    assert metrics["all_passed"], failed
    unbunched = [c for c in metrics["checks"] if c["computation_space"] == "unbunched"]
    assert unbunched and not any(c["band_limited"] for c in unbunched)
