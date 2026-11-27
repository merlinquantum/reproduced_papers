from __future__ import annotations

import math

import merlin as ml
import pytest
import torch
from common import MNIST_DIR, load_named_config
from lib.vqtn_merlin import PhotonicClassifier, stage_inputs

needs_mnist = pytest.mark.skipif(not MNIST_DIR.is_dir(), reason="MNIST not downloaded")


def _pixels(batch: int = 5) -> torch.Tensor:
    return torch.rand(batch, 16, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("superposition", ["front", "replace"])
@pytest.mark.parametrize("n_photons", [2, 4])
def test_linear_head_outputs_normalised_log_probabilities(superposition, n_photons):
    for n_classes in (2, 4):
        model = PhotonicClassifier(
            n_classes, n_photons, 1.0, superposition, "linear_head"
        )
        out = model(_pixels())
        assert out.shape == (5, n_classes)
        torch.testing.assert_close(out.exp().sum(1), torch.ones(5), atol=1e-5, rtol=0)


def test_parameter_counts_follow_the_superposition_placement():
    counts = {
        placement: sum(
            p.numel()
            for p in PhotonicClassifier(
                2, 4, 1.0, placement, "linear_head"
            ).parameters()
        )
        for placement in ("front", "replace")
    }
    # three or two trainable 8-mode meshes of 56 phases, plus an 8 x 2 + 2 linear head
    assert counts == {"front": 3 * 56 + 18, "replace": 2 * 56 + 18}
    lex = PhotonicClassifier(2, 4, 1.0, "replace", "lex_grouping")
    assert sum(p.numel() for p in lex.parameters()) == 2 * 56  # no classical parameter


def test_superposition_layer_is_fixed_and_first():
    model = PhotonicClassifier(2, 4, 1.0, "replace", "linear_head")
    names = [name for name, p in model.layer.named_parameters() if p.requires_grad]
    assert not any(name.startswith("sup") for name in names)
    first_component = list(model.layer.circuit._components)[0]
    assert first_component[0] == (0, 1)  # the cascade starts on modes (0, 1)


def test_unbunched_threshold_space_is_used():
    model = PhotonicClassifier(2, 4, 1.0, "front", "lex_grouping")
    # collision-free outcomes for 4 photons in 8 modes: C(8, 4) = 70 (FOCK would give 330)
    assert model.layer.output_size == math.comb(8, 4)
    head_model = PhotonicClassifier(2, 4, 1.0, "front", "linear_head")
    click_probabilities = head_model.layer(stage_inputs(_pixels()))
    torch.testing.assert_close(
        click_probabilities.sum(1), torch.full((5,), 4.0), atol=1e-4, rtol=0
    )


def test_input_columns_feed_the_encoding_stages_in_order():
    # MerLin behaviour this module relies on: the first 8 input columns go to stage xa.
    builder = ml.CircuitBuilder(n_modes=4)
    builder.add_superpositions(name="sup")
    builder.add_angle_encoding(modes=[0, 1, 2, 3], name="xa", scale=0.0)
    builder.add_superpositions(name="mix")
    builder.add_angle_encoding(modes=[0, 1, 2, 3], name="xb", scale=1.0)
    builder.add_superpositions(name="out")
    layer = ml.QuantumLayer(
        input_size=8, builder=builder, input_state=[1, 0, 1, 0], n_photons=2,
        measurement_strategy=ml.MeasurementStrategy.probs(ml.ComputationSpace.UNBUNCHED),
    )  # fmt: skip
    x = torch.rand(1, 8, generator=torch.Generator().manual_seed(1))
    changed_first, changed_second = x.clone(), x.clone()
    # Shift a single phase: an equal shift on every mode is a global phase and invisible.
    changed_first[0, 1] += 0.5
    changed_second[0, 5] += 0.5
    torch.testing.assert_close(layer(x), layer(changed_first))
    assert not torch.allclose(layer(x), layer(changed_second))
    pixels = torch.arange(16.0).unsqueeze(0)
    assert stage_inputs(pixels).tolist() == [
        [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
    ]


def test_gradients_reach_the_photonic_phases_and_the_head():
    model = PhotonicClassifier(2, 4, 0.5, "replace", "linear_head")
    loss = torch.nn.functional.nll_loss(model(_pixels()), torch.tensor([0, 1, 0, 1, 1]))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_same_seed_gives_the_same_model():
    torch.manual_seed(3)
    first = PhotonicClassifier(2, 2, 1.0, "front", "linear_head")
    torch.manual_seed(3)
    second = PhotonicClassifier(2, 2, 1.0, "front", "linear_head")
    torch.testing.assert_close(first(_pixels()), second(_pixels()))


def test_invalid_options_fail_clearly():
    with pytest.raises(ValueError, match="n_photons"):
        PhotonicClassifier(2, 3, 1.0, "front", "linear_head")
    with pytest.raises(ValueError, match="superposition"):
        PhotonicClassifier(2, 4, 1.0, "middle", "linear_head")
    with pytest.raises(ValueError, match="readout"):
        PhotonicClassifier(2, 4, 1.0, "front", "softmax")
    with pytest.raises(ValueError, match="divide"):
        PhotonicClassifier(4, 4, 1.0, "front", "lex_grouping")  # 70 % 4 != 0


@needs_mnist
def test_config_values_reach_the_photonic_model(tmp_path):
    from lib.mnist_task import build_mnist_task

    cfg = load_named_config(
        "e1_merlin_reduced",
        model={"n_photons": 2, "encoding_scale": 0.5, "superposition": "replace"},
        dataset={"max_train_samples": 200, "max_test_samples": 50},
    )
    task = build_mnist_task(cfg, torch.float32, torch.device("cpu"))
    model = task.model
    assert isinstance(model, PhotonicClassifier)
    assert (model.n_photons, model.encoding_scale, model.superposition, model.readout) == (
        2, 0.5, "replace", "linear_head",
    )  # fmt: skip
