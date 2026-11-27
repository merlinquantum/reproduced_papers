"""The MerLin photonic adaptation: basis order, readout, shapes, gradients, config."""

from __future__ import annotations

import json

import common  # noqa: F401  (puts the repository root on sys.path)
import merlin as ml
import perceval as pcvl
import pytest
import torch
from common import PROJECT_DIR
from merlin.core.computation_space import ComputationSpace
from merlin.measurement import MeasurementStrategy

from papers.QuLTSF.lib.models import build_model, count_trainable_parameters
from papers.QuLTSF.lib.qultsf_merlin import (
    PhotonicAmplitudeLayer,
    QuLTSFMerlin,
    qultsf_merlin_parameter_count,
    unbunched_occupancy,
)


def _empty_circuit_layer(n_modes, n_photons):
    return ml.QuantumLayer(
        circuit=pcvl.Circuit(n_modes),
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.mode_expectations(
            ComputationSpace.UNBUNCHED
        ),
    )


def test_occupancy_order_matches_merlin_unbunched_basis():
    occupancy = unbunched_occupancy(6, 3)
    basis_states = torch.eye(occupancy.shape[0]).to(torch.complex64)
    readout = _empty_circuit_layer(6, 3)(basis_states)
    torch.testing.assert_close(readout.double(), occupancy, atol=1e-6, rtol=0)


def test_identity_mode_equals_merlin_with_identity_interferometer():
    torch.manual_seed(0)
    features = torch.randn(7, 20)
    ours = PhotonicAmplitudeLayer(6, 3, "identity", torch.float32)(features)
    amplitudes = features / features.norm(dim=-1, keepdim=True)
    merlin = _empty_circuit_layer(6, 3)(amplitudes.to(torch.complex64))
    torch.testing.assert_close(ours, merlin, atol=1e-6, rtol=0)


@pytest.mark.parametrize("interferometer", ["trainable", "identity"])
def test_readout_is_scale_invariant_and_sums_to_photon_number(interferometer):
    torch.manual_seed(0)
    layer = PhotonicAmplitudeLayer(6, 3, interferometer, torch.float32)
    features = torch.randn(5, 20)
    clicks = layer(features)
    assert clicks.shape == (5, 6)
    torch.testing.assert_close(clicks.sum(dim=-1), torch.full((5,), 3.0))
    assert ((clicks >= -1e-6) & (clicks <= 1 + 1e-6)).all()
    torch.testing.assert_close(layer(4.0 * features), clicks, atol=1e-5, rtol=0)


def test_trainable_layer_uses_unbunched_threshold_readout():
    layer = PhotonicAmplitudeLayer(6, 3, "trainable", torch.float32)
    assert layer.layer.computation_space == ComputationSpace.UNBUNCHED
    assert (
        layer.layer.measurement_strategy.computation_space == ComputationSpace.UNBUNCHED
    )
    assert layer.layer.output_size == 6


def test_model_shapes_and_gradients_reach_mesh_and_input_layer():
    torch.manual_seed(0)
    model = QuLTSFMerlin(24, 8, 6, 3, "trainable", torch.float32)
    x = torch.randn(4, 24, 3)
    out = model(x)
    assert out.shape == (4, 8, 3)
    out.pow(2).mean().backward()
    mesh = list(model.hidden_photonic_layer.parameters())
    assert mesh and all(p.grad is not None and p.grad.abs().sum() > 0 for p in mesh)
    assert model.input_classical_layer.weight.grad.abs().sum() > 0


def test_channels_are_processed_independently():
    torch.manual_seed(0)
    model = QuLTSFMerlin(24, 8, 6, 3, "trainable", torch.float32)
    x = torch.randn(2, 24, 3)
    single = model(x[:, :, 1:2])
    torch.testing.assert_close(model(x)[:, :, 1:2], single, atol=1e-6, rtol=0)


def test_seeding_is_deterministic():
    def init(seed):
        torch.manual_seed(seed)
        return QuLTSFMerlin(24, 8, 6, 3, "trainable", torch.float32).state_dict()

    first, second = init(5), init(5)
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    assert any(not torch.equal(first[k], init(6)[k]) for k in first)


@pytest.mark.parametrize(
    ("interferometer", "expected"), [("trainable", 338_959), ("identity", 338_777)]
)
def test_parameter_count_at_paper_scale(interferometer, expected):
    model = QuLTSFMerlin(336, 96, 14, 4, interferometer, torch.float32)
    assert count_trainable_parameters(model) == expected
    assert qultsf_merlin_parameter_count(336, 96, 14, 4, interferometer) == expected


def test_photon_and_interferometer_arguments_are_validated():
    with pytest.raises(ValueError):
        PhotonicAmplitudeLayer(6, 1, "trainable", torch.float32)
    with pytest.raises(ValueError):
        PhotonicAmplitudeLayer(6, 3, "frozen", torch.float32)


@pytest.mark.parametrize(
    ("config", "interferometer"),
    [
        ("table1_qultsf_merlin.json", "trainable"),
        ("ablation_identity_interferometer_merlin.json", "identity"),
    ],
)
def test_configs_reach_the_model_unchanged(config, interferometer):
    cfg = json.loads((PROJECT_DIR / "configs" / config).read_text())
    model = build_model(cfg, torch.float32)
    layer = model.hidden_photonic_layer
    assert isinstance(model, QuLTSFMerlin)
    assert (layer.n_modes, layer.n_photons, layer.interferometer) == (
        14,
        4,
        interferometer,
    )
    assert model.input_classical_layer.out_features == 1001
    assert model.output_classical_layer.out_features == 96


def test_single_chunk_propagation_matches_merlin_default_chunking():
    torch.manual_seed(0)
    layer = PhotonicAmplitudeLayer(6, 3, "trainable", torch.float32)
    features = torch.randn(9, 20)
    amplitudes = features / features.norm(dim=-1, keepdim=True)
    default = layer.layer(amplitudes.to(torch.complex64))
    torch.testing.assert_close(layer(features), default, atol=1e-6, rtol=0)
