"""Coordinator bookkeeping for stopped sweeps: PARTIAL closure and resume planning."""

from __future__ import annotations

import importlib.util
import json
import logging

import common
import pytest

SPEC_PATH = common.PROJECT_DIR / "utils" / "run_sweep.py"


@pytest.fixture(scope="module")
def run_sweep():
    spec = importlib.util.spec_from_file_location("run_sweep", SPEC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spec():
    return {
        "sweep_id": "S9_test",
        "fixed_args": [],
        "candidates": [{"name": "A", "args": []}, {"name": "B", "args": []}],
        "seeds": [0, 1],
        "max_parallel": 1,
        "threads_per_run": 1,
        "teacher_sweep": None,
        "selection": {
            "metric": "final_evaluation.test_accuracy",
            "direction": "maximize",
            "split": "test",
            "checkpoint": "final_epoch",
            "aggregation": "mean",
            "tie_tolerance": 0.0,
            "tie_break": "report_all",
        },
    }


def _write_run(sweep_dir, label, status, with_checkpoint):
    run_dir = sweep_dir / "runs" / label / "run_20260101-000000"
    run_dir.mkdir(parents=True)
    if with_checkpoint:
        (run_dir / "checkpoint.pt").write_bytes(b"x")
    if status == "COMPLETED":
        (run_dir / "metrics.json").write_text(
            json.dumps({"final_evaluation": {"test_accuracy": 90.0}})
        )
    return run_dir


def test_resume_spec_plans_only_unfinished_pairs_with_checkpoints(tmp_path, run_sweep):
    sweep_dir = tmp_path / "S9_test_20260101-000000"
    sweep_dir.mkdir()
    spec = _spec()
    (sweep_dir / "spec.json").write_text(json.dumps(spec))
    runs = []
    for name, seed, status, checkpoint in [
        ("A", 0, "COMPLETED", True),
        ("B", 0, "FAILED", True),
        ("A", 1, "RUNNING", True),
    ]:
        _write_run(sweep_dir, f"{name}__seed{seed}", status, checkpoint)
        runs.append({"candidate": name, "seed": seed, "status": status})
    (sweep_dir / "sweep_status.json").write_text(
        json.dumps({"status": "PARTIAL", "runs": runs})
    )

    resumed = run_sweep.resume_spec(sweep_dir)
    assert resumed["sweep_id"] == "S9_test_resume1"
    assert sorted(map(tuple, resumed["only_runs"])) == [("A", 1), ("B", 0), ("B", 1)]
    assert set(resumed["resume_from"]) == {"B__seed0", "A__seed1"}
    assert resumed["resume_from"]["B__seed0"].endswith("run_20260101-000000")


def test_stopped_sweep_with_no_completed_run_is_partial(tmp_path, run_sweep, caplog):
    spec = _spec()
    status = {
        "status": "RUNNING",
        "runs": [
            {"run_id": "A__seed0", "candidate": "A", "seed": 0, "status": "FAILED"}
        ],
    }
    with caplog.at_level(logging.INFO):
        run_sweep.summarize(tmp_path, spec, status, stopped=True)
    assert status["status"] == "PARTIAL"
    assert "SWEEP_PARTIAL" in caplog.text
