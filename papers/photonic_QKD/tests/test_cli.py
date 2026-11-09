from __future__ import annotations

import copy
import json

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def _tiny_config() -> dict:
    cfg = copy.deepcopy(load_runtime_ready_config())
    cfg["dataset"].update(
        {
            "name": "synthetic",
            "train_size": 192,
            "val_size": 128,
            "eval_batch_size": 128,
        }
    )
    cfg["dataset"]["synthetic"]["test_size"] = 128
    cfg["teacher"]["widths"] = [8, 16, 16]
    cfg["photonic"] = {"n_modes": 8, "input_state": [1, 0] * 4}
    cfg["conditioning"].update(
        {"feature_dim": 32, "theta_dim": 14, "reference_evaluations": 4}
    )
    cfg["spsa"].update({"updates_per_epoch": 1, "val_batches": 1, "val_batch_size": 64})
    cfg["training"]["epochs"] = 2
    return cfg


def _status(run_dir):
    return json.loads((run_dir / "run_status.json").read_text())


def test_teacher_then_student_runs_write_contract_artifacts(tmp_path):
    from photonic_QKD.lib import runner as pqkd_runner

    teacher_cfg = _tiny_config()
    teacher_dir = tmp_path / "run_teacher"
    teacher_dir.mkdir()
    pqkd_runner.train_and_evaluate(teacher_cfg, teacher_dir)
    assert _status(teacher_dir)["status"] == "COMPLETED"
    assert (teacher_dir / "teacher.pt").is_file()
    teacher_metrics = json.loads((teacher_dir / "metrics.json").read_text())
    assert len(teacher_metrics["history"]) == 2

    student_cfg = _tiny_config()
    student_cfg["task"] = "student"
    student_cfg["teacher"]["checkpoint"] = str(teacher_dir)
    student_dir = tmp_path / "run_student"
    student_dir.mkdir()
    pqkd_runner.train_and_evaluate(student_cfg, student_dir)
    status = _status(student_dir)
    assert status["status"] == "COMPLETED"
    assert status["candidate"]["conditioning_source"] == "photonic"
    metrics = json.loads((student_dir / "metrics.json").read_text())
    assert metrics["conditioning_source"] == "photonic"
    assert metrics["parameters"]["conditioning_trainable"] == 14
    assert "delta_test_accuracy_vs_zero_feature" in metrics["final_evaluation"]
    assert (student_dir / "feature_trace.npz").is_file()
    log = (student_dir / "run.log").read_text(encoding="utf-8")
    for event in (
        "RUN_STARTED",
        "DATASET_READY",
        "TRAINING_STARTED",
        "TRAIN_EPOCH_COMPLETED",
        "EVALUATION_COMPLETED",
        "METRICS_WRITTEN",
        "RUN_COMPLETED",
    ):
        assert event in log


@pytest.mark.parametrize("variant", ["dense", "dictionary_trainable"])
def test_baseline_student_variants_run(tmp_path, variant):
    from photonic_QKD.lib import runner as pqkd_runner

    teacher_dir = tmp_path / "run_teacher"
    teacher_dir.mkdir()
    teacher_cfg = _tiny_config()
    teacher_cfg["training"]["epochs"] = 1
    pqkd_runner.train_and_evaluate(teacher_cfg, teacher_dir)

    cfg = _tiny_config()
    cfg.update(task="student")
    cfg["training"]["epochs"] = 1
    cfg["teacher"]["checkpoint"] = str(teacher_dir / "teacher.pt")
    cfg["student"]["variant"] = variant
    run_dir = tmp_path / f"run_{variant}"
    run_dir.mkdir()
    pqkd_runner.train_and_evaluate(cfg, run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["conditioning_source"] == "none"
    assert metrics["parameters"]["conditioning_trainable"] == 0


def test_mismatched_teacher_is_rejected(tmp_path):
    from photonic_QKD.lib import runner as pqkd_runner

    teacher_dir = tmp_path / "run_teacher"
    teacher_dir.mkdir()
    teacher_cfg = _tiny_config()
    teacher_cfg["training"]["epochs"] = 1
    pqkd_runner.train_and_evaluate(teacher_cfg, teacher_dir)

    cfg = _tiny_config()
    cfg.update(task="student")
    cfg["teacher"]["checkpoint"] = str(teacher_dir)
    cfg["teacher"]["widths"] = [16, 32, 32]
    run_dir = tmp_path / "run_bad"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="does not match"):
        pqkd_runner.train_and_evaluate(cfg, run_dir)
    assert _status(run_dir)["status"] == "FAILED"
