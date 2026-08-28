from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

QORC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = QORC_DIR.parents[1]
for import_path in (REPO_ROOT, QORC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

pytest.importorskip("merlin")

from lib import lib_qorc_encoding_and_linear_training as qorc_training
from lib.comparison import plot_qorc_lsvc_comparison
from lib.lib_remote_qorc import (
    _QORCProcessor,
    create_remote_qorc_processor,
)


class FakeMeasurementStrategy:
    @staticmethod
    def probs(**kwargs):
        return kwargs


class FakeReservoirClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.layer = SimpleNamespace(measurement_strategy=None)


def test_factory_configures_merlin_reservoir_classifier(monkeypatch):
    monkeypatch.setattr(
        qorc_training.ML, "ReservoirClassifier", FakeReservoirClassifier
    )
    monkeypatch.setattr(
        qorc_training.ML, "MeasurementStrategy", FakeMeasurementStrategy
    )
    monkeypatch.setattr(
        qorc_training.ML,
        "ComputationSpace",
        SimpleNamespace(FOCK="fock"),
    )

    reservoir = qorc_training.create_qorc_reservoir_classifier(
        n_photons=2,
        n_components=9,
        seed=42,
        device_name="cpu",
        b_no_bunching=False,
    )

    assert reservoir.kwargs["in_features"] == 28 * 28
    assert reservoir.kwargs["out_features"] == 10
    assert reservoir.kwargs["n_photons"] == 2
    assert reservoir.kwargs["reduction"].n_components == 9
    assert reservoir.kwargs["concatenate"] is True
    assert reservoir.kwargs["cache"] is True
    assert reservoir.kwargs["seed"] == 42
    assert reservoir.kwargs["device"] == torch.device("cpu")
    assert reservoir.layer.measurement_strategy == {"computation_space": "fock"}


def test_factory_selects_no_bunching_measurement(monkeypatch):
    monkeypatch.setattr(
        qorc_training.ML, "ReservoirClassifier", FakeReservoirClassifier
    )
    monkeypatch.setattr(
        qorc_training.ML, "MeasurementStrategy", FakeMeasurementStrategy
    )

    reservoir = qorc_training.create_qorc_reservoir_classifier(
        n_photons=2,
        n_components=9,
        seed=42,
        device_name="cpu",
        b_no_bunching=True,
    )

    assert reservoir.layer.measurement_strategy == {}


def test_noise_factory_returns_none_when_disabled():
    assert qorc_training.create_perceval_noise_model(enabled=False) is None


def test_noise_factory_maps_perceval_source_parameters():
    noise = qorc_training.create_perceval_noise_model(
        enabled=True,
        indistinguishability=0.87,
        g2=0.04,
        g2_distinguishable=False,
    )

    assert noise.indistinguishability == 0.87
    assert noise.g2 == 0.04
    assert noise.g2_distinguishable is False


@pytest.mark.parametrize("parameter_name", ["indistinguishability", "g2"])
def test_noise_factory_rejects_invalid_probabilities(parameter_name):
    with pytest.raises(ValueError, match="between 0 and 1"):
        qorc_training.create_perceval_noise_model(
            enabled=True,
            **{parameter_name: 1.1},
        )


def test_remote_processor_adapter_forwards_sample_count():
    calls = []

    class FakeProcessor:
        def forward(self, module, inputs, *, nsample=None):
            calls.append((module, inputs, nsample))
            return inputs + 1

    module = object()
    inputs = torch.zeros(2, 3)
    adapter = _QORCProcessor(FakeProcessor(), nsample=257)

    result = adapter.forward(module, inputs)

    assert torch.equal(result, torch.ones(2, 3))
    assert len(calls) == 1
    assert calls[0][0] is module
    assert calls[0][1] is inputs
    assert calls[0][2] == 257


def test_remote_processor_rejects_unknown_backend(monkeypatch):
    with pytest.raises(ValueError, match="not recognized"):
        create_remote_qorc_processor(
            "qpu:unknown",
            SimpleNamespace(),
            qpu_device_nsample=100,
            logger=logging.getLogger("test_qorc_reservoir"),
        )


def test_comparison_plot_writes_png(tmp_path):
    metrics = {
        "epochs": [1, 2],
        "qorc_train_accuracy": [0.4, 0.7],
        "qorc_test_accuracy": [0.35, 0.65],
        "qorc_train_loss": [1.2, 0.6],
        "qorc_test_loss": [1.3, 0.8],
        "linear_train_accuracy": [0.6, 0.8],
        "linear_test_accuracy": [0.55, 0.75],
        "linear_train_loss": [0.7, 0.4],
        "linear_test_loss": [0.8, 0.5],
    }
    output_path = tmp_path / "comparison.png"

    plot_qorc_lsvc_comparison(metrics, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
