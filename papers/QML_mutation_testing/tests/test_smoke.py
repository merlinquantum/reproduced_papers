from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from common import PROJECT_DIR, load_runtime_ready_config

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import merlin as ml  # noqa: E402
from lib.circuit_ir import build_original_ir, set_weights  # noqa: E402
from lib.data import load_iris_splits  # noqa: E402
from lib.photonic_qnn import PhotonicQnnClassifier, PhotonicQnnSpec  # noqa: E402
from lib.training import (  # noqa: E402
    TrainingConfig,
    build_mutation_test_suite,
    train_photonic_qnn,
)


def _spec(**overrides) -> PhotonicQnnSpec:
    base = {
        "n_features": 4,
        "n_classes": 3,
        "n_modes": 4,
        "n_photons": 2,
        "n_layers": 2,
        "encoding_scale": math.pi,
        "computation_space": "unbunched",
        "input_state": [1, 0, 1, 0],
    }
    base.update(overrides)
    return PhotonicQnnSpec(**base)


def test_iso_parameter_with_real_amplitudes():
    """The photonic ansatz must keep RealAmplitudes' n_qubits * n_layers budget."""
    model = PhotonicQnnClassifier(_spec(n_layers=4))
    assert model.num_trainable_parameters() == 4 * 4


def test_ir_path_matches_circuit_builder():
    """The component-level IR must realise exactly the CircuitBuilder circuit.

    The mutation engine only works on the IR, so a divergence here would make
    every mutant incomparable to the original model of Iteration 1.
    """
    n_modes, n_layers = 4, 3
    weights = [0.1 * index for index in range(n_modes * n_layers)]

    ir = set_weights(
        build_original_ir(n_modes=n_modes, n_features=n_modes, n_layers=n_layers),
        weights,
    )
    ir_model = PhotonicQnnClassifier(
        _spec(n_layers=n_layers, encoding_scale=math.pi), ir
    )

    builder = ml.CircuitBuilder(n_modes=n_modes)
    builder.add_superpositions(trainable=False)
    builder.add_angle_encoding(modes=list(range(n_modes)), scale=math.pi)
    for _ in range(n_layers):
        builder.add_rotations(axis="z", trainable=True)
        builder.add_superpositions(trainable=False)
    reference = ml.QuantumLayer(
        input_size=n_modes,
        builder=builder,
        input_state=[1, 0, 1, 0],
        n_photons=2,
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.UNBUNCHED
        ),
    )
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.tensor(weights))

    x = torch.rand(4, n_modes)
    grouping = ml.LexGrouping(reference.output_size, 3)
    with torch.no_grad():
        expected = grouping(reference(x))
        obtained = ir_model(x)
    assert torch.allclose(obtained, expected, atol=1e-6)


def test_forward_returns_class_distribution():
    model = PhotonicQnnClassifier(_spec())
    probs = model(torch.rand(5, 4))
    assert probs.shape == (5, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-5)


def test_single_photon_is_rejected():
    try:
        PhotonicQnnClassifier(_spec(n_photons=1, input_state=[1, 0, 0, 0]))
    except ValueError as exc:
        assert "at least 2 photons" in str(exc)
    else:
        raise AssertionError("a single-photon circuit must be rejected")


def test_tiny_training_run_builds_a_test_suite():
    cfg = load_runtime_ready_config()
    splits = load_iris_splits(
        seed=cfg["seed"],
        test_size=cfg["dataset"]["test_size"],
        val_size=cfg["dataset"]["val_size"],
        n_suite_samples=cfg["dataset"]["n_suite_samples"],
        feature_max=cfg["dataset"]["feature_max"],
    )
    model, metrics = train_photonic_qnn(
        _spec(), splits, TrainingConfig(epochs=5, lr=0.05, log_every=5), seed=0
    )
    suite = build_mutation_test_suite(model, splits, shots=128)
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert suite["n_candidate_samples"] == 20
    assert suite["n_suite_samples"] <= 20
    assert len(suite["suite_positions"]) == suite["n_suite_samples"]


def test_runner_writes_contract_artifacts(tmp_path: Path):
    from lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["experiment"]["seeds"] = [7]
    cfg["training"]["epochs"] = 3
    cfg["evaluation"]["shots"] = 64
    cfg["mutation_testing"]["enabled"] = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_snapshot.json").write_text("{}", encoding="utf-8")

    paper_runner.train_and_evaluate(cfg, run_dir)

    for name in ("run_status.json", "metrics.json", "summary.json"):
        assert (run_dir / name).exists(), f"missing artifact: {name}"
