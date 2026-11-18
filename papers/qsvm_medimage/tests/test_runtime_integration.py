"""Fast checks for the shared catalogue runtime contract."""

import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_runtime_markers_are_valid():
    defaults = json.loads((PROJECT_ROOT / "configs" / "defaults.json").read_text())
    cli = json.loads((PROJECT_ROOT / "cli.json").read_text())

    assert defaults["scope"] == "merlin_photonic_adaptation"
    assert defaults["device"] == "cpu"
    assert "fix_leakage" not in defaults["experiment"]
    assert cli["arguments"]


def test_runner_reports_missing_dataset(tmp_path):
    from lib.runner import train_and_evaluate

    cfg = {
        "scope": "merlin_photonic_adaptation",
        "seed": 0,
        "dtype": "float32",
        "device": "cpu",
        "data_root": str(tmp_path),
        "dataset": {"filename": "missing.pkl"},
        "experiment": {"pca_dim": 2, "circuit_seed": 0, "max_samples": 100},
    }

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        train_and_evaluate(cfg, tmp_path / "run")


@pytest.mark.parametrize("fix_leakage", [False, True])
def test_runner_only_enables_train_only_preprocessing_when_requested(
    tmp_path, monkeypatch, fix_leakage
):
    from lib.runner import train_and_evaluate

    data_root = tmp_path / "data"
    dataset_path = data_root / "qsvm_medimage" / "dataset.pkl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.touch()

    captured = {}
    merlin_module = types.ModuleType("scripts.merlin_fidelity_kernel")

    def fake_main(arguments):
        captured["arguments"] = arguments

    merlin_module.main = fake_main
    monkeypatch.setitem(sys.modules, "scripts.merlin_fidelity_kernel", merlin_module)

    cfg = {
        "scope": "merlin_photonic_adaptation",
        "seed": 0,
        "dtype": "float32",
        "device": "cpu",
        "data_root": str(data_root),
        "dataset": {"filename": dataset_path.name},
        "experiment": {
            "pca_dim": 2,
            "circuit_seed": 0,
            "max_samples": 100,
            "fix_leakage": fix_leakage,
        },
    }

    train_and_evaluate(cfg, tmp_path / "run")

    assert ("--fix_leakage" in captured["arguments"]) is fix_leakage
