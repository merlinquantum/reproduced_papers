from __future__ import annotations

import copy

import common  # noqa: F401
import pytest
import torch
import torch.nn.functional as functional
from lib.conditioning import SOURCES, build_conditioning_source
from lib.training import SPSA, kd_loss


def test_kd_loss_reduces_to_cross_entropy_when_lambda_is_one():
    logits = torch.randn(6, 10)
    teacher = torch.randn(6, 10)
    labels = torch.randint(0, 10, (6,))
    torch.testing.assert_close(
        kd_loss(logits, teacher, labels, 3.0, 1.0),
        functional.cross_entropy(logits, labels),
    )


def test_kd_loss_soft_term_matches_eq2():
    logits = torch.randn(6, 10)
    teacher = torch.randn(6, 10)
    labels = torch.randint(0, 10, (6,))
    tau = 3.0
    p_teacher = functional.softmax(teacher / tau, dim=1)
    kl = (
        (p_teacher * (p_teacher.log() - functional.log_softmax(logits / tau, dim=1)))
        .sum(1)
        .mean()
    )
    expected = 0.5 * functional.cross_entropy(logits, labels) + 0.5 * tau**2 * kl
    torch.testing.assert_close(kd_loss(logits, teacher, labels, tau, 0.5), expected)


def test_spsa_descends_a_quadratic():
    spsa = SPSA(
        step_size=0.1,
        perturbation=0.1,
        bound=3.0,
        generator=torch.Generator().manual_seed(0),
    )
    target = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    vector = torch.zeros(3, dtype=torch.float64)
    for _ in range(200):
        vector, _, _ = spsa.step(vector, lambda v: float(((v - target) ** 2).sum()))
    assert torch.allclose(vector, target, atol=0.1)


def test_spsa_respects_bound():
    spsa = SPSA(
        step_size=10.0,
        perturbation=0.5,
        bound=1.0,
        generator=torch.Generator().manual_seed(0),
    )
    vector, _, _ = spsa.step(
        torch.zeros(4, dtype=torch.float64), lambda v: float(v.sum()) * 100
    )
    assert vector.abs().max() <= 1.0


@pytest.fixture(scope="module")
def small_cfg():
    cfg = copy.deepcopy(common.load_project_defaults())
    cfg["photonic"] = {"n_modes": 8, "input_state": [1, 0] * 4}
    cfg["conditioning"].update(
        {"feature_dim": 32, "theta_dim": 14, "reference_evaluations": 6}
    )
    return cfg


@pytest.mark.parametrize("source_name", SOURCES)
def test_every_conditioning_source_builds_and_samples(source_name, small_cfg):
    cfg = copy.deepcopy(small_cfg)
    cfg["conditioning"]["source"] = source_name
    source = build_conditioning_source(cfg, torch.Generator().manual_seed(0))
    feature = source.feature(torch.Generator().manual_seed(1))
    assert feature.shape == (32,)
    assert feature.dtype == torch.float32
    assert torch.isfinite(feature).all()
    if source_name == "zero":
        assert torch.count_nonzero(feature) == 0
    else:
        assert feature.abs().sum() > 0
    assert source.learnable == (source_name in ("photonic", "surrogate"))
    if source.learnable:
        assert source.parameter_count == 14


def test_photonic_feature_depends_on_theta(small_cfg):
    source = build_conditioning_source(
        copy.deepcopy(small_cfg), torch.Generator().manual_seed(0)
    )
    first = source.feature(torch.Generator().manual_seed(1))
    source.set_vector(torch.full((14,), 0.7, dtype=torch.float64))
    second = source.feature(torch.Generator().manual_seed(1))
    assert not torch.allclose(first, second)


def test_frozen_photonic_source_rejects_updates(small_cfg):
    cfg = copy.deepcopy(small_cfg)
    cfg["conditioning"]["source"] = "photonic_fixed"
    source = build_conditioning_source(cfg, torch.Generator().manual_seed(0))
    with pytest.raises(RuntimeError):
        source.set_vector(torch.zeros(14))
