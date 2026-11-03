from __future__ import annotations

import torch
from common import PROJECT_DIR  # noqa: F401
from lib.classical import ClassicalClassifier
from lib.qaoa import QAOAModel, generate_graph_batch
from lib.vqtn import TREE_NODES, VQTNClassifier

DTYPE = torch.float64


def _pixels(batch: int = 6) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(batch, 16, generator=generator, dtype=DTYPE)


def test_vqtn_outputs_normalised_log_probabilities():
    for n_classes in (2, 4):
        model = VQTNClassifier(n_classes, 0.3, DTYPE)
        probabilities = model(_pixels()).exp()
        assert probabilities.shape == (6, n_classes)
        # binary read-out is a marginal of one qubit, 4-class of two: both sum to 1
        torch.testing.assert_close(
            probabilities.sum(1), torch.ones(6, dtype=DTYPE), atol=1e-6, rtol=0
        )


def test_vqtn_parameter_count_is_21_and_matches_the_classical_baseline():
    quantum = VQTNClassifier(2, 0.1, DTYPE)
    classical = ClassicalClassifier(2, hidden=1, dtype=DTYPE)
    assert sum(p.numel() for p in quantum.parameters()) == 3 * len(TREE_NODES) == 21
    assert sum(p.numel() for p in classical.parameters()) == 21


def test_parameter_shift_equals_autograd_for_vqtn_angles():
    # The paper trains with parameter-shift (qSGD); this reproduction uses autograd.
    # In noiseless simulation both give the exact derivative of class probabilities.
    model = VQTNClassifier(2, 0.4, DTYPE)
    pixels = _pixels(4)

    def class_one_probability() -> torch.Tensor:
        return model(pixels).exp()[:, 1].sum()

    class_one_probability().backward()
    autograd = model.angles.grad.clone()
    shifted = torch.zeros_like(autograd)
    with torch.no_grad():
        for node in range(autograd.shape[0]):
            for index in range(autograd.shape[1]):
                original = model.angles[node, index].item()
                model.angles[node, index] = original + torch.pi / 2
                plus = class_one_probability()
                model.angles[node, index] = original - torch.pi / 2
                minus = class_one_probability()
                model.angles[node, index] = original
                shifted[node, index] = (plus - minus) / 2
    torch.testing.assert_close(shifted, autograd, atol=1e-6, rtol=0)


def test_qaoa_parameter_shift_equals_autograd():
    diagonal, _ = generate_graph_batch(3, 6, 3, seed=0)
    model = QAOAModel(6, 1, 0.3, DTYPE)
    diagonal = diagonal.to(DTYPE)
    model(diagonal).sum().backward()
    # gamma enters through exp(-i gamma C) whose generator has a wider spectrum than a
    # Pauli, so only beta (a sum of Pauli X rotations) obeys the two-term shift rule;
    # check gamma against central finite differences instead.
    with torch.no_grad():
        eps = 1e-5
        base = model.gamma.item()
        model.gamma.fill_(base + eps)
        plus = model(diagonal).sum()
        model.gamma.fill_(base - eps)
        minus = model(diagonal).sum()
        model.gamma.fill_(base)
    torch.testing.assert_close(
        model.gamma.grad[0], (plus - minus) / (2 * eps), atol=1e-6, rtol=0
    )


def test_qaoa_initial_expected_cut_is_half_the_edges():
    diagonal, max_cut = generate_graph_batch(5, 6, 3, seed=1)
    model = QAOAModel(6, 1, 0.0, DTYPE)
    expected = model(diagonal.to(DTYPE))
    torch.testing.assert_close(expected, torch.full((5,), 4.5, dtype=DTYPE))
    assert (max_cut >= 6).all()


def test_graph_diagonal_uses_the_requested_graph_family():
    diagonal, max_cut = generate_graph_batch(4, 6, 3, seed=2)
    assert diagonal.shape == (4, 64)
    # a 3-regular graph on 6 nodes has 9 edges: no cut exceeds 9, and none is below 5
    assert diagonal.max().item() <= 9 and max_cut.min().item() >= 5


def test_vqtn_depth_multiplies_parameters_and_keeps_probabilities_normalised():
    for layers in (1, 2, 3):
        model = VQTNClassifier(2, 0.3, DTYPE, layers=layers)
        assert sum(p.numel() for p in model.parameters()) == 21 * layers
        torch.testing.assert_close(
            model(_pixels()).exp().sum(1), torch.ones(6, dtype=DTYPE), atol=1e-6, rtol=0
        )


def test_vqtn_rejects_zero_layers():
    import pytest

    with pytest.raises(ValueError, match="layers"):
        VQTNClassifier(2, 0.1, DTYPE, layers=0)


def test_diagnostic_switches_default_to_the_original_circuit():
    default = VQTNClassifier(2, 0.2, DTYPE, layers=2)
    explicit = VQTNClassifier(2, 0.2, DTYPE, 2, 1.0, "rz", "root_marginal")
    explicit.load_state_dict(default.state_dict())
    torch.testing.assert_close(default(_pixels()), explicit(_pixels()))


def test_each_diagnostic_switch_changes_the_output_in_isolation():
    base = VQTNClassifier(2, 0.2, DTYPE)
    for kwargs in ({"encoding_scale": 2.0}, {"encoding_gate": "rx"}):
        other = VQTNClassifier(2, 0.2, DTYPE, **kwargs)
        other.load_state_dict(base.state_dict())
        assert not torch.allclose(base(_pixels()), other(_pixels()), atol=1e-4), kwargs


def test_z_head_readout_adds_a_linear_layer_and_stays_normalised():
    model = VQTNClassifier(2, 0.2, DTYPE, readout="z_head")
    assert sum(p.numel() for p in model.parameters()) == 21 + 8 * 2 + 2
    torch.testing.assert_close(
        model(_pixels()).exp().sum(1), torch.ones(6, dtype=DTYPE), atol=1e-6, rtol=0
    )


def test_invalid_switches_fail_clearly():
    import pytest

    with pytest.raises(ValueError, match="encoding_gate"):
        VQTNClassifier(2, 0.1, DTYPE, encoding_gate="ry")
    with pytest.raises(ValueError, match="readout"):
        VQTNClassifier(2, 0.1, DTYPE, readout="softmax")
