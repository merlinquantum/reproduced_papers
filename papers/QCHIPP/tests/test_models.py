"""Numerical and architecture contracts for the local hierarchical models."""

import math

import pytest
import torch

torch.set_num_threads(1)


def test_residue_local_qrac_padding_and_pair_order():
    from lib.photonic import residue_angles

    bits = torch.tensor([[0, 1, 1, 0, 1] * 4], dtype=torch.float32)
    angles = residue_angles(bits)
    expected = torch.tensor([3 * math.pi / 4, -math.pi / 4, -math.pi / 4])
    assert angles.shape == (1, 4, 3)
    torch.testing.assert_close(angles, expected.expand(1, 4, 3))
    varied = torch.arange(20).remainder(2).reshape(1, 20)
    result = residue_angles(varied)
    pairs = torch.nn.functional.pad(varied.reshape(1, 4, 5), (0, 1)).reshape(1, 4, 3, 2)
    b0, b1 = pairs[..., 0], pairs[..., 1]
    torch.testing.assert_close(
        result, math.pi / 4 + math.pi / 2 * (b1 - b0) + math.pi * b1 * b0
    )


def test_photonic_hierarchy_fock_probabilities_and_parameter_budget():
    import merlin
    from lib.photonic import PhotonicQCNN, residue_angles

    torch.manual_seed(7)
    model = PhotonicQCNN(n_photons=2)
    bits = torch.randint(0, 2, (3, 20)).float()
    observed = []
    handles = [
        cell.register_forward_pre_hook(
            lambda module, args: observed.append(args[0].detach().clone())
        )
        for cell in (model.stage1, model.stage2)
    ]
    logits = model(bits)
    for handle in handles:
        handle.remove()
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    assert isinstance(model.stage1.quantum, merlin.QuantumLayer)
    assert isinstance(model.stage2.quantum, merlin.QuantumLayer)
    assert model.stage1.quantum is not model.stage2.quantum
    torch.testing.assert_close(observed[0], residue_angles(bits).reshape(6, 6))
    with torch.no_grad():
        pooled = model.stage1(observed[0]).reshape(3, 6) * math.pi
    torch.testing.assert_close(observed[1], pooled)
    for cell in (model.stage1, model.stage2):
        assert cell.quantum.output_size == 21
        probs = cell.quantum(torch.randn(4, 6))
        assert probs.shape == (4, 21)
        assert torch.isfinite(probs).all()
        assert (probs >= 0).all()
        torch.testing.assert_close(probs.sum(-1), torch.ones(4), atol=1e-5, rtol=1e-5)
        assert cell.quantum.output_keys is not None
        assert all(sum(key) == 2 and len(key) == 6 for key in cell.quantum.output_keys)
    quantum_count = sum(
        p.numel()
        for cell in (model.stage1, model.stage2)
        for p in cell.quantum.parameters()
    )
    total = sum(p.numel() for p in model.parameters())
    assert quantum_count == 120
    assert total - quantum_count == 136
    assert total == 256


def test_classical_same_input_hierarchy_and_nearly_matched_budget():
    from lib.classical import ClassicalQCNN
    from lib.photonic import residue_angles

    model = ClassicalQCNN(hidden=12)
    bits = torch.randint(0, 2, (4, 20))
    observed = []
    handles = [
        cell.register_forward_pre_hook(
            lambda module, args: observed.append(args[0].detach().clone())
        )
        for cell in (model.stage1, model.stage2)
    ]
    logits = model(bits)
    for handle in handles:
        handle.remove()
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(observed[0], residue_angles(bits).reshape(8, 6))
    torch.testing.assert_close(
        observed[1], model.stage1(observed[0]).reshape(4, 6) * math.pi
    )
    assert model.stage1 is not model.stage2
    assert [type(module) for module in model.stage1] == [
        torch.nn.Linear,
        torch.nn.Tanh,
        torch.nn.Linear,
        torch.nn.Tanh,
    ]
    assert sum(p.numel() for p in model.parameters()) == 250


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_photons": 1},
        {"n_photons": 3},
        {"n_photons": 2.0},
        {"n_photons": True},
        {"input_state": [1, 0, 0, 1, 0]},
        {"input_state": [1, 0, 0, 0, 0, 0]},
        {"input_state": [-1, 0, 0, 3, 0, 0]},
        {"input_state": [1.0, 0, 0, 1, 0, 0]},
        {"input_state": [2, 0, 0, 0, 0, 0]},
        {"input_state": None},
    ],
)
def test_photonic_rejects_unsupported_configuration_without_repair(kwargs):
    from lib.photonic import PhotonicCell, PhotonicQCNN

    for factory in (PhotonicQCNN, PhotonicCell):
        with pytest.raises(ValueError):
            factory(**kwargs)


@pytest.mark.parametrize(
    "bits",
    [
        torch.zeros(2, 19),
        torch.zeros(40),
        torch.zeros(1, 4, 5),
        torch.full((2, 20), 0.5),
        torch.full((2, 20), float("nan")),
        torch.zeros(0, 20),
    ],
)
def test_both_models_reject_invalid_residue_bits(bits):
    from lib.classical import ClassicalQCNN
    from lib.photonic import PhotonicQCNN

    for model in (PhotonicQCNN(), ClassicalQCNN()):
        with pytest.raises(ValueError, match="bits"):
            model(bits)


@pytest.mark.parametrize("hidden", [0, -1, True, 12.0])
def test_classical_rejects_invalid_hidden_width(hidden):
    from lib.classical import ClassicalQCNN

    with pytest.raises(ValueError, match="hidden"):
        ClassicalQCNN(hidden=hidden)


@pytest.mark.parametrize("shots", [-1, 1.5, True, None])
def test_photonic_rejects_invalid_shot_budget(shots):
    from lib.photonic import PhotonicQCNN

    model = PhotonicQCNN().eval()
    with torch.no_grad(), pytest.raises(ValueError, match="shots"):
        model(torch.zeros(2, 20), shots=shots)


def test_finite_shots_require_eval_and_no_grad_even_for_direct_cells():
    from lib.photonic import PhotonicQCNN

    model = PhotonicQCNN()
    bits = torch.zeros(2, 20)
    with torch.no_grad(), pytest.raises(ValueError, match="eval"):
        model(bits, shots=10)
    model.eval()
    with pytest.raises(ValueError, match="no_grad"):
        model(bits, shots=10)
    model.stage1.train()
    with torch.no_grad(), pytest.raises(ValueError, match="eval"):
        model.stage1(torch.zeros(2, 6), shots=10)


def test_eval_finite_shots_sample_every_cell_and_propagate_samples():
    from lib.photonic import PhotonicQCNN

    torch.manual_seed(123)
    model = PhotonicQCNN().eval()
    bits = torch.randint(0, 2, (4, 20))
    measurements = []
    second_inputs = []
    handles = [
        cell.quantum.register_forward_hook(
            lambda module, args, kwargs, output: measurements.append(
                (args[0].clone(), kwargs["shots"], output.clone())
            ),
            with_kwargs=True,
        )
        for cell in (model.stage1, model.stage2)
    ]
    handles.append(
        model.stage2.register_forward_pre_hook(
            lambda module, args: second_inputs.append(args[0].clone())
        )
    )
    with torch.no_grad():
        analytic = model(bits)
        measurements.clear()
        second_inputs.clear()
        sampled = model(bits, shots=31)
        again = model(bits, shots=31)
    for handle in handles:
        handle.remove()
    assert [entry[0].shape for entry in measurements] == [
        (8, 6),
        (4, 6),
        (8, 6),
        (4, 6),
    ]
    assert all(entry[1] == 31 for entry in measurements)
    for _, _, probs in measurements:
        torch.testing.assert_close(probs.sum(-1), torch.ones(probs.shape[0]))
        torch.testing.assert_close(probs * 31, (probs * 31).round(), atol=1e-5, rtol=0)
    expected = (
        torch.tanh(model.stage1.projection(measurements[0][2])).reshape(4, 6) * math.pi
    )
    torch.testing.assert_close(second_inputs[0], expected)
    assert not torch.allclose(sampled, analytic)
    assert not torch.allclose(sampled, again)


@pytest.mark.parametrize("model_name", ["photonic", "classical"])
def test_bce_adam_step_has_nonzero_stage_gradients_and_changes_weights(model_name):
    from lib.classical import ClassicalQCNN
    from lib.photonic import PhotonicQCNN

    torch.manual_seed(19)
    model = PhotonicQCNN() if model_name == "photonic" else ClassicalQCNN()
    bits = torch.randint(0, 2, (8, 20))
    targets = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(bits), targets)
    loss.backward()
    for stage in (model.stage1, model.stage2):
        core = stage.quantum if model_name == "photonic" else stage
        grads = [p.grad for p in core.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        assert sum(grad.abs().sum().item() for grad in grads) > 0
    optimizer.step()
    for stage_name in ("stage1", "stage2"):
        core_prefix = stage_name + (".quantum" if model_name == "photonic" else "")
        assert any(
            not torch.equal(value, before[name])
            for name, value in model.named_parameters()
            if name.startswith(core_prefix)
        )


@pytest.mark.parametrize("model_name", ["photonic", "classical"])
def test_seed_reproduces_initial_weights_and_analytic_output(model_name):
    from lib.classical import ClassicalQCNN
    from lib.photonic import PhotonicQCNN

    factory = PhotonicQCNN if model_name == "photonic" else ClassicalQCNN
    bits = torch.arange(60).remainder(2).reshape(3, 20)
    torch.manual_seed(37)
    first = factory().eval()
    torch.manual_seed(37)
    second = factory().eval()
    for p1, p2 in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(first(bits), second(bits), rtol=0, atol=0)
        if model_name == "photonic":
            torch.manual_seed(999)
            shot1 = first(bits, shots=47)
            torch.manual_seed(999)
            shot2 = second(bits, shots=47)
            torch.testing.assert_close(shot1, shot2, rtol=0, atol=0)
