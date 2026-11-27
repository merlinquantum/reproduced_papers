"""Exercise native shots, real phase updates and financial campaign artifacts."""

import json
from pathlib import Path

import numpy as np
import pytest
from lib.data import prepare_data
from lib.runner import train_and_evaluate
from test_financial import financial_fixture
from test_spsa import SETTINGS

from runtime_lib.dtypes import resolve_config_dtypes


def test_financial_campaign_updates_phases_and_preserves_paired_budgets(tmp_path):
    data = tmp_path / "data/error-qcbm/fx"
    data.mkdir(parents=True)
    _, dataset = financial_fixture(data)
    conditions = [
        {
            "name": "lossless",
            "transmittance": 1.0,
            "indistinguishability": 1.0,
            "estimator": "postselection",
        },
        {
            "name": "postselection",
            "transmittance": 0.2,
            "indistinguishability": 1.0,
            "estimator": "postselection",
        },
        {
            "name": "recycling",
            "transmittance": 0.2,
            "indistinguishability": 1.0,
            "estimator": "recycling",
        },
    ]
    cfg = {
        "task": "train",
        "device": "cpu",
        "dtype": "float64",
        "data_root": str(tmp_path / "data"),
        "dataset": dataset,
        "model": {"input_state": [1, 0, 1, 0, 0, 0]},
        "training": {
            "loss": "tvd",
            "seeds": [0],
            "conditions": conditions,
            "steps": 3,
            "record_every": 1,
            "shots": 2000,
            "torch_threads": 1,
            "sampling_seed_offset": 10000,
            "perturbation_seed_offset": 100000,
            "spsa": SETTINGS,
        },
    }
    output = tmp_path / "run"
    output.mkdir()
    preparation_output = tmp_path / "preparation"
    preparation_output.mkdir()
    prepare_data(
        {**cfg, "data_preparation": {"download": False, "timeout_seconds": 1}},
        preparation_output,
    )
    prepared_metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    train_and_evaluate(resolve_config_dtypes(cfg), output)
    trained_metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    for key in ("source_credit", "source_terms_url", "transformation_notice"):
        assert trained_metadata[key] == run_metadata[key] == prepared_metadata[key]
    assert "European Central Bank" in run_metadata["source_credit"]
    report = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["completed_instances"] == 3
    assert report["failed_instances"] == 0
    for case in conditions:
        directory = output / "seed_00" / case["name"]
        result = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
        assert (
            result["objective_calls"] == 12
        )  # initial + calibration 2 + updates 6 + records 3
        assert result["history"][-1]["parameter_change_norm"] > 0
        assert len(result["history"]) == 4
        assert result["sampling_engine"] == "merlin_native_QuantumLayer_shots"
        with np.load(directory / "final.npz") as final:
            assert final["initial_counts"].sum() == 2000
            assert final["final_counts"].sum() == 2000
            assert np.isfinite(final["parameters"]).all()
            assert final["exact_probabilities"].sum() == pytest.approx(1.0, abs=1e-12)
        with np.load(directory / "checkpoint.npz") as checkpoint:
            assert checkpoint["step"] == 3
            assert checkpoint["objective_calls"] == 12


def test_gaussian_kl_training_uses_author_target_and_records_true_loss(tmp_path):
    project = Path(__file__).resolve().parents[1]
    cfg = json.loads((project / "configs/gaussian.json").read_text(encoding="utf-8"))
    cfg["data_root"] = str(tmp_path / "data")
    cfg["model"]["input_state"] = [1, 0, 1, 0, 0, 0]
    cfg["dataset"].update(means=[4, 10], standard_deviations=[2, 2])
    cfg["training"].update(seeds=[0], steps=2, record_every=1, shots=2000)
    output = tmp_path / "gaussian"
    output.mkdir()
    train_and_evaluate(resolve_config_dtypes(cfg), output)
    campaign = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
    assert campaign["status"] == "complete"
    assert campaign["loss_name"] == "kl"
    assert campaign["completed_instances"] == 3
    for case in cfg["training"]["conditions"]:
        directory = output / "seed_00" / case["name"]
        report = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
        with np.load(directory / "final.npz") as final:
            p, q = final["exact_probabilities"], final["target"]
            mask = p > 0
            expected_kl = float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))
        assert report["final_exact_loss"] == pytest.approx(expected_kl, abs=1e-12)
        assert report["history"][-1]["exact_lossless_loss"] == pytest.approx(
            expected_kl
        )
        assert report["history"][-1]["parameter_change_norm"] > 0
        assert report["loss_name"] == "kl"
