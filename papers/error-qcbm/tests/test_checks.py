"""Phase-zero numerical and runtime boundary checks."""

import json
from pathlib import Path

import pytest
from lib.runner import train_and_evaluate

from runtime_lib.dtypes import resolve_config_dtypes


def test_environment_report(tmp_path):
    cfg = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/defaults.json").read_text(
            encoding="utf-8"
        )
    )
    train_and_evaluate(resolve_config_dtypes(cfg), tmp_path)
    report = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    assert report["max_probability_error"] < 1e-6
    assert report["gradient_error"] < 1e-6
    assert report["sampling_disabled_with_grad"]
    assert report["training_implemented"]
    assert not report["training_exercised"]


def test_training_requires_explicit_settings_without_success_artifact(tmp_path):
    with pytest.raises(ValueError, match="Training requires explicit"):
        train_and_evaluate({"task": "train"}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_unknown_task_fails(tmp_path):
    with pytest.raises(ValueError, match="Unknown task"):
        train_and_evaluate({"task": "typo"}, tmp_path)
