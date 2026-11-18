"""Run evidence, checkpointing and resume of the training runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from lib import runner  # noqa: E402


def _config(**overrides) -> dict:
    config = {
        "description": "checkpoint test",
        "feasibility": {"status": "supported"},
        "dataset": "graph_reg",
        "dataset_kwargs": {"num_graphs": 40, "n_range": [5, 8], "p": 0.4, "seed": 0},
        "model": "grit_lite",
        "encoding": "rrwp",
        "rrwp_dim": 4,
        "node_dim": 8,
        "depth": 1,
        "num_heads": 2,
        "head": "graph_reg",
        "epochs": 4,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "train_frac": 0.7,
        "val_frac": 0.15,
        "device": "cpu",
        "seed": 3,
        "checkpoint_every": 1,
    }
    config.update(overrides)
    return config


def test_completed_run_writes_the_evidence_files(tmp_path):
    output = runner.train_and_evaluate(_config(), tmp_path)
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["metrics_path"] == "metrics.json"
    assert status["dataset"]["splits"]["test"] == len(
        np.load(tmp_path / "test_outputs.npz")["absolute_errors"]
    )
    assert (tmp_path / "checkpoint.pt").is_file()
    assert (tmp_path / "best_model.pt").is_file()
    assert output["test_outputs_path"] == "test_outputs.npz"
    errors = np.load(tmp_path / "test_outputs.npz")["absolute_errors"]
    assert errors.mean() == pytest.approx(output["test_metrics"]["mae"])


def test_failed_run_is_recorded_and_resume_reproduces_the_uninterrupted_run(
    tmp_path, monkeypatch
):
    reference = runner.train_and_evaluate(_config(), tmp_path / "reference")

    original_train = runner._train_one_epoch
    calls = {"count": 0}

    def crash_on_third_epoch(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("simulated interruption")
        return original_train(*args, **kwargs)

    monkeypatch.setattr(runner, "_train_one_epoch", crash_on_third_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.train_and_evaluate(_config(), tmp_path / "interrupted")
    status = json.loads((tmp_path / "interrupted" / "run_status.json").read_text())
    assert status["status"] == "FAILED"
    assert status["error"]["type"] == "RuntimeError"
    monkeypatch.setattr(runner, "_train_one_epoch", original_train)

    resumed = runner.train_and_evaluate(
        _config(resume_from=str(tmp_path / "interrupted" / "checkpoint.pt")),
        tmp_path / "resumed",
    )
    assert resumed["resumed_from"].endswith("checkpoint.pt")
    assert resumed["history"]["val_mae"] == pytest.approx(
        reference["history"]["val_mae"], abs=1e-12
    )
    assert resumed["best_epoch"] == reference["best_epoch"]
    assert resumed["test_metrics"]["mae"] == pytest.approx(
        reference["test_metrics"]["mae"], abs=1e-12
    )


def test_resume_rejects_a_checkpoint_from_another_experiment(tmp_path):
    runner.train_and_evaluate(_config(), tmp_path / "first")
    with pytest.raises(ValueError, match="different experiment"):
        runner.train_and_evaluate(
            _config(seed=4, resume_from=str(tmp_path / "first" / "checkpoint.pt")),
            tmp_path / "second",
        )


def test_rejected_config_is_recorded_as_a_failed_run(tmp_path):
    with pytest.raises(ValueError, match="seeds must not contain duplicates"):
        runner.train_and_evaluate(_config(seeds=[1, 1]), tmp_path)
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "FAILED"
    assert status["error"]["type"] == "ValueError"
