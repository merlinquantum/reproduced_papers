from __future__ import annotations

import common  # noqa: F401  (puts the paper package on sys.path)
import numpy as np
import pytest
import torch
from trainable_frequency_qnn.lib.models import (
    build_network,
    circuit_evaluation_cost_factor,
    parameter_summary,
)
from trainable_frequency_qnn.lib.spectrum import (
    accessible_frequencies,
    dominant_frequencies,
    prediction_spectrum,
)


def _toy_model(
    frequency_mode: str, n_qubits: int = 3, feature_map: str = "simple", seed: int = 0
):
    model_cfg = {
        "family": "qnn",
        "n_qubits": n_qubits,
        "layout": "toy",
        "n_layers": 4,
        "entangler": "cnot",
        "feature_map": feature_map,
        "frequency_mode": frequency_mode,
        "frequency_init": 1.0,
    }
    generator = torch.Generator().manual_seed(seed)
    return build_network(model_cfg, 1, torch.float64, generator)


def test_simple_feature_map_has_paper_natural_frequencies():
    # Paper Sec. V: for N = 3 the gaps of sum_m Y_m / 2 are {1, 2, 3}.
    assert accessible_frequencies([1.0, 1.0, 1.0]) == [1.0, 2.0, 3.0]


def test_paper_learned_generator_contains_irregular_gaps():
    # Paper Sec. V: theta_F = {0.89, 1.05, 1.04} gives gaps including 1.05, 1.2 and
    # ~3. The paper also lists 0.95, which these rounded values cannot produce
    # (LOG.md, A16).
    gaps = accessible_frequencies([0.89, 1.05, 1.04])
    for expected in (1.05, 1.2, 2.98):
        assert min(abs(g - expected) for g in gaps) < 1e-9


def test_feature_map_spectral_richness_scaling():
    # Paper Sec. V: simple O(N), tower O(N^2), exponential O(2^N) unique gaps.
    assert len(accessible_frequencies([1, 1, 1, 1])) == 4
    assert len(accessible_frequencies([1, 2, 3, 4])) == 10
    assert len(accessible_frequencies([1, 2, 4, 8])) == 15


def test_tf_model_at_initialisation_equals_ff_model():
    # theta_F starts at 1 and the ansatz draw does not depend on the mode, so a
    # TF model is exactly the FF model with the same seed (paired seeds).
    inputs = torch.linspace(-4, 4, 11, dtype=torch.float64)[:, None]
    outputs = [
        _toy_model(mode)(inputs) for mode in ("fixed", "trainable", "global_scale")
    ]
    assert torch.allclose(outputs[0], outputs[1])
    assert torch.allclose(outputs[0], outputs[2])


def test_ff_prediction_spectrum_only_contains_natural_frequencies():
    model = _toy_model("fixed", seed=3)
    angular, magnitudes = prediction_spectrum(model, 64, 16384, torch.float64, "cpu")
    peaks = dominant_frequencies(angular, magnitudes, 3)
    assert np.allclose(peaks, [1.0, 2.0, 3.0], atol=2 * (angular[1] - angular[0]))


def test_trainable_frequencies_receive_gradients():
    model = _toy_model("trainable")
    inputs = torch.linspace(-4, 4, 5, dtype=torch.float64)[:, None]
    model(inputs).sum().backward()
    assert model.frequency_weights.grad is not None
    assert torch.any(model.frequency_weights.grad != 0)


@pytest.mark.parametrize(
    "mode, expected_frequency_parameters",
    [("fixed", 0), ("trainable", 6), ("global_scale", 3)],
)
def test_navier_stokes_qnn_parameter_counts(mode, expected_frequency_parameters):
    # Paper Sec. VI: 6 qubits, 180 ansatz parameters per QNN, 2 qubits per coordinate.
    model_cfg = {
        "family": "qnn",
        "n_qubits": 6,
        "layout": "ns_parallel",
        "n_layers": 10,
        "entangler": "cz",
        "feature_map": "simple",
        "frequency_mode": mode,
        "frequency_init": 1.0,
    }
    network = build_network(
        model_cfg, 3, torch.float64, torch.Generator().manual_seed(0)
    )
    summary = parameter_summary(network)
    assert summary["ansatz"] == 180
    assert summary["frequency"] == expected_frequency_parameters
    assert network(torch.zeros((2, 3), dtype=torch.float64)).shape == (2,)


@pytest.mark.parametrize(
    "layout, n_qubits, n_inputs, n_layers, entangler",
    [
        ("toy", 3, 1, 4, "cnot"),
        ("ns_parallel", 6, 3, 10, "cz"),
        ("ns_serial", 4, 3, 9, "cnot"),
    ],
)
def test_fused_forward_matches_gatewise_reference(
    layout, n_qubits, n_inputs, n_layers, entangler
):
    model_cfg = {
        "family": "qnn",
        "n_qubits": n_qubits,
        "layout": layout,
        "n_layers": n_layers,
        "entangler": entangler,
        "feature_map": "tower",
        "frequency_mode": "trainable",
        "frequency_init": 1.0,
    }
    network = build_network(
        model_cfg, n_inputs, torch.float64, torch.Generator().manual_seed(1)
    )
    with torch.no_grad():
        network.frequency_weights.uniform_(
            0.5, 1.5, generator=torch.Generator().manual_seed(2)
        )
    inputs = torch.rand(
        (7, n_inputs), generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    inputs = (4 * inputs - 2).requires_grad_(True)
    fused = network(inputs)
    reference = network.forward_gatewise(inputs)
    assert torch.allclose(fused, reference, atol=1e-12)
    fused_gradient = torch.autograd.grad(fused.sum(), inputs)[0]
    reference_gradient = torch.autograd.grad(reference.sum(), inputs)[0]
    assert torch.allclose(fused_gradient, reference_gradient, atol=1e-12)


def test_cost_factor_counts_both_qnns():
    # Two TF QNNs: 12 theta_F over 360 ansatz parameters. The paper's 1.07 divides
    # the 12 theta_F of both QNNs by the 180 ansatz parameters of one (LOG.md, A15).
    summaries = [{"ansatz": 180, "frequency": 6}, {"ansatz": 180, "frequency": 6}]
    assert circuit_evaluation_cost_factor(summaries) == pytest.approx(372 / 360)


def test_appendix_c_serial_qnn_matches_paper_parameter_count():
    model_cfg = {
        "family": "qnn",
        "n_qubits": 4,
        "layout": "ns_serial",
        "n_layers": 67,
        "entangler": "cnot",
        "feature_map": "simple",
        "frequency_mode": "trainable",
        "frequency_init": 1.0,
    }
    network = build_network(
        model_cfg, 3, torch.float64, torch.Generator().manual_seed(0)
    )
    summary = parameter_summary(network)
    assert summary["ansatz"] == 804
    assert summary["frequency"] == 12
    assert sum(1 for step, _ in network._program if step == "fm") == 6


def test_frequency_init_sets_fixed_and_trainable_multipliers():
    for mode in ("fixed", "trainable", "global_scale"):
        model_cfg = {
            "family": "qnn",
            "n_qubits": 3,
            "layout": "toy",
            "n_layers": 2,
            "entangler": "cnot",
            "feature_map": "tower",
            "frequency_mode": mode,
            "frequency_init": 1.5,
        }
        network = build_network(
            model_cfg, 1, torch.float64, torch.Generator().manual_seed(0)
        )
        assert torch.allclose(
            network.encoding_frequencies(),
            torch.tensor([1.5, 3.0, 4.5], dtype=torch.float64),
        )
