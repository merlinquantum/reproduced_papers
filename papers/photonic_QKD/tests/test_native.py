"""Phase 4 MerLin-native sources: gradients, consistency with the PQKD feature, init modes."""

from __future__ import annotations

import copy

import common  # noqa: F401
import pytest
import torch
from lib.conditioning import (
    NativePhotonicSource,
    NativeSurrogateSource,
    build_conditioning_source,
)
from lib.models import ThreeBlockCNN


@pytest.fixture()
def small_cfg():
    cfg = copy.deepcopy(common.load_project_defaults())
    cfg["photonic"] = {"n_modes": 8, "input_state": [1, 0] * 4}
    cfg["conditioning"].update(
        {"feature_dim": 32, "theta_dim": 14, "reference_evaluations": 6}
    )
    return cfg


def _build(cfg, source, mode="uniform"):
    cfg = copy.deepcopy(cfg)
    cfg["conditioning"].update({"source": source, "theta_init_mode": mode})
    return build_conditioning_source(cfg, torch.Generator().manual_seed(0))


def test_native_photonic_gradient_reaches_merlin_angles(small_cfg):
    source = _build(small_cfg, "photonic_native")
    assert isinstance(source, NativePhotonicSource)
    assert source.differentiable and not source.learnable
    parameters = list(source.parameters())
    assert len(parameters) == 1 and parameters[0].numel() == 14
    feature = source.differentiable_feature()
    assert feature.shape == (32,) and feature.requires_grad
    (feature * torch.linspace(-1, 1, 32)).sum().backward()
    assert parameters[0].grad is not None
    assert parameters[0].grad.abs().sum() > 0


def test_native_feature_equals_standardised_exact_pqkd_feature(small_cfg):
    source = _build(small_cfg, "photonic_native")
    angles = source.vector()
    exact = source.feature_map.raw_feature(angles, None, torch.Generator())
    expected = source.standardiser(exact).float()
    torch.testing.assert_close(source.feature(torch.Generator()), expected)


def test_native_surrogate_gradient_and_parameter_count(small_cfg):
    source = _build(small_cfg, "surrogate_native")
    assert isinstance(source, NativeSurrogateSource)
    assert source.parameter_count == 14
    source.differentiable_feature().sum().backward()
    assert source.phi.grad is not None and source.phi.grad.abs().sum() > 0


def test_init_modes(small_cfg):
    zero = _build(small_cfg, "photonic_native", mode="constant").vector()
    uniform = _build(small_cfg, "photonic_native", mode="uniform").vector()
    assert torch.count_nonzero(zero) == 0
    assert uniform.abs().max() <= torch.pi and uniform.std() > 0.5


def test_native_requires_full_angle_budget(small_cfg):
    cfg = copy.deepcopy(small_cfg)
    cfg["conditioning"]["theta_dim"] = 10
    with pytest.raises(ValueError, match="trains every angle"):
        _build(cfg, "photonic_native")


def test_in_graph_feature_matches_stored_mixing():
    model = ThreeBlockCNN(
        1,
        (8, 16, 16),
        0.0,
        compressed=(True, True, False),
        ranks=(4, 4, 4),
        feature_dim=32,
        generator=torch.Generator().manual_seed(0),
    ).eval()
    images = torch.randn(3, 1, 28, 28)
    feature = torch.randn(32)
    in_graph = model(images, feature=feature)
    model.set_feature(feature)
    torch.testing.assert_close(in_graph, model(images))


def test_checkpoint_state_round_trip(small_cfg):
    source = _build(small_cfg, "photonic_native")
    state = source.checkpoint_state()
    with torch.no_grad():
        source.feature_map.angles.add_(1.0)
    source.load_checkpoint_state(state)
    torch.testing.assert_close(source.vector(), state["angles"])
