import importlib
import sys

import pytest
import torch
from common import PROJECT_DIR
from lib.photonic_kernel_svm import qksvm_accuracy

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

pytest.importorskip("merlin")
pytest.importorskip("perceval")

import merlin as ml  # noqa: E402
import perceval as pcvl  # noqa: E402


def test_photonic_pool_mirrors_gate_pool_structure():
    """The photonic token pool follows the same 'create a pool' pattern as the gate-based
    pool: data-carrying single-mode gates (PS) plus non-data entanglers (BS)."""
    from lib.photonic_circuits import build_token_pool

    pool = build_token_pool(n_modes=6, num_features=6)
    gates = {t[0] for t in pool}
    assert "PS" in gates and "BS" in gates
    ps = [t for t in pool if t[0] == "PS"]
    assert all(0 <= t[2] < 6 for t in ps)


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
    encoder = create_quantum_module(sequence, num_features=2, n_modes=2)

    assert encoder.ps_data_indices == [1, 0]
    assert encoder.layer.input_size == 4

    x = torch.randn(3, 2, dtype=torch.float32)
    states = encoder(x)
    assert isinstance(states, torch.Tensor)
    assert states.shape[0] == 3
    assert states.ndim == 2


def test_photonic_kernel_svm_accuracy_uses_precomputed_kernel():
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
    photonic_bias = importlib.import_module("lib.photonic_bias")

    sequence = [("PS_PI", 0, 0, 0.0)]
    X = torch.randn(4, 1, dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    encoder, e_before, e_after = photonic_bias.refine_bias(
        sequence,
        X,
        y,
        num_features=4,
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


def test_refine_bias_returns_new_energy():
    photonic_bias = importlib.import_module("lib.photonic_bias")

    sequence = [
        ("BS", 0, 0, torch.pi / 2),
        ("PS", 0, 0, 1 * torch.pi),
        ("BS", 0, 0, torch.pi / 2),
        ("PS", 1, 1, 0.1 * torch.pi),
        ("BS", 1, 0, torch.pi / 2),
    ]
    X = torch.randn(4, 2, dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    encoder, e_before, e_after = photonic_bias.refine_bias(
        sequence,
        X,
        y,
        num_features=2,
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

    assert e_before != e_after
    assert hasattr(encoder, "bias")
    assert len(list(encoder.bias.parameters())) > 0
