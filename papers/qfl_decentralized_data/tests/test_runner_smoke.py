from __future__ import annotations

import json
import math

import pytest
from common import MNIST_DIR, load_named_config

pytest.importorskip("pyscf")

from lib.runner import train_and_evaluate  # noqa: E402

needs_mnist = pytest.mark.skipif(not MNIST_DIR.is_dir(), reason="MNIST not downloaded")


def _run(name: str, tmp_path, **overrides) -> tuple[dict, dict, str]:
    cfg = load_named_config(name, **overrides)
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(cfg, default=str), encoding="utf-8"
    )
    train_and_evaluate(cfg, run_dir)
    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return status, metrics, (run_dir / "run.log").read_text()


def _assert_completed(status: dict, metrics: dict, log: str) -> None:
    assert status["status"] == "COMPLETED" and status["error"] is None
    assert status["metrics_sha256"] and status["config_sha256"]
    assert log.count("RUN_STARTED") == 1 and log.count("RUN_COMPLETED") == 1
    assert "TRAINING_STARTED" in log and "EVALUATION_COMPLETED" in log
    assert math.isfinite(metrics["final_primary_metric"])


def test_qaoa_reduced_run_writes_complete_evidence(tmp_path):
    status, metrics, log = _run("e3_reduced", tmp_path)
    _assert_completed(status, metrics, log)
    assert 0.0 < metrics["final_primary_metric"] <= 1.0  # approximation ratio


def test_vqe_reduced_run_writes_complete_evidence(tmp_path):
    status, metrics, log = _run("e4_reduced", tmp_path)
    _assert_completed(status, metrics, log)
    assert metrics["final_primary_metric"] >= 0.0  # mean absolute energy error


@needs_mnist
def test_vqtn_reduced_run_writes_complete_evidence(tmp_path):
    status, metrics, log = _run(
        "e1_reduced",
        tmp_path,
        dataset={"max_train_samples": 300, "max_test_samples": 100},
        federation={"rounds": 1},
    )
    _assert_completed(status, metrics, log)
    assert status["dataset"]["splits"]["client_sizes"] and len(metrics["history"]) == 1


@needs_mnist
def test_four_class_and_classical_baseline_runs_complete(tmp_path):
    small = {"max_train_samples": 300, "max_test_samples": 100}
    status, metrics, log = _run(
        "e2_reduced", tmp_path, dataset=small, federation={"rounds": 1}
    )
    _assert_completed(status, metrics, log)
    assert len(metrics["final_report"]["per_class_f1"]) == 4
    status, metrics, log = _run(
        "e1_classical", tmp_path, dataset=small, federation={"rounds": 1}
    )
    _assert_completed(status, metrics, log)
    assert metrics["trainable_parameters"] == 21


def test_missing_config_key_fails_with_a_failed_status(tmp_path):
    cfg = load_named_config("e3_reduced")
    del cfg["federation"]["rounds"]
    run_dir = tmp_path / "broken"
    run_dir.mkdir()
    with pytest.raises(KeyError, match="federation.rounds"):
        train_and_evaluate(cfg, run_dir)
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "FAILED" and status["error"]["type"] == "KeyError"
    assert (run_dir / "run.log").read_text().count("RUN_FAILED") == 1
