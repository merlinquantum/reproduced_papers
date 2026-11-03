"""Stop (SIGTERM) and resume: a resumed run must equal an uninterrupted one."""

from __future__ import annotations

import copy
import json
import os
import signal

import pytest
from common import load_runtime_ready_config


def _tiny_config(source: str) -> dict:
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
        {
            "source": source,
            "feature_dim": 32,
            "theta_dim": 14,
            "reference_evaluations": 4,
            "theta_init_mode": "uniform",
        }
    )
    if source.endswith("_native"):
        cfg["conditioning"]["ema"]["enabled"] = False
    cfg["spsa"].update({"updates_per_epoch": 1, "val_batches": 1, "val_batch_size": 64})
    cfg["training"]["epochs"] = 3
    return cfg


def _run(runner, cfg, run_dir):
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)
    return json.loads((run_dir / "metrics.json").read_text())


@pytest.fixture(scope="module")
def teacher_dir(tmp_path_factory):
    from photonic_QKD.lib import runner

    directory = tmp_path_factory.mktemp("teacher") / "run_teacher"
    cfg = _tiny_config("photonic")
    cfg["training"]["epochs"] = 1
    _run(runner, cfg, directory)
    return directory


@pytest.mark.parametrize("source", ["photonic", "photonic_native", "surrogate_native"])
def test_sigterm_stop_then_resume_matches_uninterrupted(
    tmp_path, teacher_dir, monkeypatch, source
):
    from photonic_QKD.lib import runner, training

    def student_cfg():
        cfg = _tiny_config(source)
        cfg.update(task="student")
        cfg["teacher"]["checkpoint"] = str(teacher_dir)
        return cfg

    reference = _run(runner, student_cfg(), tmp_path / "run_reference")

    original_save = training.save_checkpoint

    def save_then_stop(path, state):
        original_save(path, state)
        if state["epoch"] == 2:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(training, "save_checkpoint", save_then_stop)
    stopped_dir = tmp_path / "run_stopped"
    stopped_dir.mkdir()
    with pytest.raises(runner.StopRequestedError):
        runner.train_and_evaluate(student_cfg(), stopped_dir)
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    status = json.loads((stopped_dir / "run_status.json").read_text())
    assert status["status"] == "FAILED"
    assert status["error"]["type"] == "StopRequestedError"
    assert (stopped_dir / "checkpoint.pt").is_file()
    assert "Traceback" in (stopped_dir / "run.log").read_text(encoding="utf-8")

    resumed_cfg = student_cfg()
    resumed_cfg["resume_from"] = str(stopped_dir)
    resumed = _run(runner, resumed_cfg, tmp_path / "run_resumed")
    log = (tmp_path / "run_resumed" / "run.log").read_text(encoding="utf-8")
    assert "TRAIN_RESUME_COMPLETED | restored_epochs=2" in log
    assert log.count("TRAIN_EPOCH_COMPLETED") == 1
    assert [h["epoch"] for h in resumed["history"]] == [1, 2, 3]
    assert (
        resumed["final_evaluation"]["test_accuracy"]
        == reference["final_evaluation"]["test_accuracy"]
    )
    assert resumed["history"][-1]["train_kd_loss"] == pytest.approx(
        reference["history"][-1]["train_kd_loss"], rel=1e-6
    )


def test_resume_refuses_a_different_experiment(tmp_path, teacher_dir, monkeypatch):
    from photonic_QKD.lib import runner

    cfg = _tiny_config("photonic")
    cfg.update(task="student")
    cfg["teacher"]["checkpoint"] = str(teacher_dir)
    cfg["training"]["epochs"] = 1
    first = tmp_path / "run_first"
    _run(runner, cfg, first)
    other = copy.deepcopy(cfg)
    other["conditioning"]["shots"] = 25
    other["resume_from"] = str(first)
    with pytest.raises(ValueError, match="different experiment"):
        _run(runner, other, tmp_path / "run_other")
