import sys

import pytest
import torch
from common import PROJECT_DIR

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

pytest.importorskip("merlin")
pytest.importorskip("perceval")

import merlin as ml
import perceval as pcvl


def test_default_input_state_alternates_modes():
    from lib.photonic import default_input_state

    assert default_input_state(4, 2) == [1, 0, 1, 0]
    assert default_input_state(4, 3) == [1, 1, 1, 0]
    assert default_input_state(5, 3) == [1, 0, 1, 0, 1]
    assert default_input_state(5, 4) == [1, 1, 1, 0, 1]


def test_build_feature_map_assigns_input_and_trainable_parameters():
    from lib.photonic import build_feature_map

    fm = build_feature_map(n_modes=3, n_layers=2, scale=0.5)

    assert fm.input_size == 3
    assert fm.input_parameters == "px"
    assert fm.trainable_parameters == ["el"]


def test_make_kernel_passes_parameter_assignments_to_fidelity_kernel():
    from lib.photonic import make_kernel

    kern, state = make_kernel(
        n_modes=4, n_photons=3, n_layers=1, scale=1.2, device="cpu"
    )

    assert state == [1, 1, 1, 0]
    assert kern.input_state == state
    assert sum(kern.input_state) == 3
    assert getattr(kern, "computation_space", None) == ml.ComputationSpace.UNBUNCHED

    feature_map = getattr(kern, "feature_map", None)
    assert feature_map is not None
    assert feature_map.input_size == 4
    assert feature_map.input_parameters == "px"
    assert feature_map.trainable_parameters == ["el"]


def test_create_perceval_circuit_builds_expected_parameters():
    from lib.photonic_circuits import create_perceval_circuit

    sequence = [("PS", 0, 1, 0.3), ("PS_PI", 1, 0, 0.0), ("BS", 0, 0, 0.0)]
    circuit, input_params, trainable_params = create_perceval_circuit(
        sequence, n_modes=2
    )

    assert isinstance(circuit, pcvl.Circuit)
    assert len(input_params) == 1
    assert len(trainable_params) == 1
    assert input_params[0].name == "theta0"
    assert trainable_params[0].name == "phi0"


def test_create_quantum_module_uses_ps_data_indices_and_trainable_parameters():
    from lib.photonic_circuits import create_quantum_module

    sequence = [("PS", 0, 1, 0.3), ("PS", 1, 0, 0.5), ("PS_PI", 1, 0, 0.0)]
    encoder = create_quantum_module(sequence, n_modes=2)

    assert encoder.ps_data_indices == [1, 0]
    assert encoder.layer.input_size == 2

    trainable_params = [p for p in encoder.layer.parameters() if p.requires_grad][0]
    assert trainable_params.numel() == 2

    x = torch.randn(3, 2, dtype=torch.float32)
    states = encoder(x)
    assert isinstance(states, torch.Tensor)
    assert states.shape[0] == 3
    assert states.ndim == 2


def test_photonic_kernel_svm_accuracy_uses_precomputed_kernel():
    from lib.photonic_kernel_svm import qksvm_accuracy

    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model = DummyModel()
    X_train = [[1.0, 0.0], [0.0, 1.0]]
    y_train = [0, 1]
    X_test = [[1.0, 0.0], [0.0, 1.0]]
    y_test = [0, 1]

    acc = qksvm_accuracy(model, X_train, y_train, X_test, y_test)

    assert acc == 1.0


def test_refine_bias_returns_same_energy_when_no_trainable_parameters():
    import importlib

    photonic_bias = importlib.import_module("lib.photonic_bias")

    sequence = [("PS_PI", 0, 0, 0.0)]
    X = torch.randn(4, 1, dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    encoder, e_before, e_after = photonic_bias.refine_bias(
        sequence,
        X,
        y,
        n_modes=3,
        num_photons=2,
        computation_space=ml.ComputationSpace.UNBUNCHED,
        epochs=1,
        batch_samples=2,
        lr=0.1,
        seed=0,
        device="cpu",
        hidden=None,
        gain=None,
    )

    assert e_before == e_after
    assert hasattr(encoder, "layer")
    assert hasattr(encoder.layer, "trainable_parameters")
    assert len(encoder.layer.trainable_parameters) == 0
