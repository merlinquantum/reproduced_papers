from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)
import pytest
import torch
from lib.models import (
    DictionaryConv2d,
    ThreeBlockCNN,
    conv_parameter_count,
    iso_parameter_widths,
    teacher_parameter_formula,
    trainable_parameter_count,
)


@pytest.mark.parametrize(
    ("widths", "expected"),
    [((32, 64, 128), 94_474), ((48, 96, 128), 154_826), ((64, 128, 128), 224_394)],
)
def test_teacher_parameter_count_matches_paper(widths, expected):
    model = ThreeBlockCNN(1, widths, 0.25)
    assert trainable_parameter_count(model) == expected
    assert teacher_parameter_formula(widths) == expected


def _student(widths, compressed, rank=4, mixing="feature"):
    return ThreeBlockCNN(
        1,
        widths,
        0.25,
        compressed=compressed,
        ranks=(rank, rank, rank),
        feature_dim=512,
        mixing=mixing,
        generator=torch.Generator().manual_seed(0),
    )


@pytest.mark.parametrize(
    ("widths", "compressed", "paper_cx"),
    [
        ((32, 64, 128), (True, True, False), 1.25),
        ((48, 96, 128), (True, True, False), 1.38),
        ((64, 128, 128), (True, True, False), 1.50),
        ((32, 64, 128), (True, True, True), 48.81),
        ((48, 96, 128), (True, True, True), 76.18),
        ((64, 128, 128), (True, True, True), 105.40),
    ],
)
def test_compression_factors_match_table2(widths, compressed, paper_cx):
    # Table 2 uses R=4, dim(theta)=30; the per-layer count R*k^2 + 2*C_out is
    # the inferred accounting (bias + per-channel gain).
    student = _student(widths, compressed)
    overall = teacher_parameter_formula(widths) / (
        trainable_parameter_count(student) + 30
    )
    assert overall == pytest.approx(paper_cx, abs=0.2)


def test_dictionary_layer_trainable_count_and_fixed_projection():
    layer = DictionaryConv2d(
        64,
        128,
        3,
        1,
        rank=4,
        feature_dim=512,
        mixing="feature",
        generator=torch.Generator().manual_seed(0),
    )
    assert trainable_parameter_count(layer) == 4 * 9 + 2 * 128
    assert not layer.projection.requires_grad
    assert layer.projection.shape == (128 * 64 * 4, 512)
    assert layer.projection.var().item() == pytest.approx(1 / 512, rel=0.05)


def test_dictionary_kernel_follows_eq6_and_eq7():
    layer = DictionaryConv2d(
        3,
        5,
        3,
        1,
        rank=2,
        feature_dim=8,
        mixing="feature",
        generator=torch.Generator().manual_seed(1),
    )
    feature = torch.randn(8)
    layer.set_feature(feature)
    mixing = (layer.projection @ feature).view(5, 3, 2)
    expected = torch.einsum("oir,rxy->oixy", mixing, layer.basis) * layer.gain.view(
        -1, 1, 1, 1
    )
    torch.testing.assert_close(layer.kernel(), expected)
    layer.set_feature(torch.zeros(8))
    assert torch.count_nonzero(layer.kernel()) == 0


def test_gradients_reach_basis_but_not_projection():
    student = _student((8, 16, 16), (True, True, False))
    student.set_feature(torch.randn(512))
    student(torch.randn(4, 1, 28, 28)).sum().backward()
    first = student.convs[0]
    assert first.basis.grad is not None and first.basis.grad.abs().sum() > 0
    assert first.gain.grad is not None
    assert first.projection.grad is None


def test_trainable_mixing_baseline_has_more_parameters():
    feature_student = _student((64, 128, 128), (True, True, False))
    trainable_student = _student(
        (64, 128, 128), (True, True, False), mixing="trainable"
    )
    assert not trainable_student.uses_feature
    extra = trainable_parameter_count(trainable_student) - trainable_parameter_count(
        feature_student
    )
    assert extra == 64 * 1 * 4 + 128 * 64 * 4


def test_iso_parameter_widths_are_close_to_target():
    target = (
        trainable_parameter_count(_student((64, 128, 128), (True, True, False))) + 45
    )
    widths = iso_parameter_widths(target, 1, 128)
    assert abs(teacher_parameter_formula(widths) - target) / target < 0.02
    small_target = (
        trainable_parameter_count(_student((64, 128, 128), (True, True, True))) + 45
    )
    small = iso_parameter_widths(small_target, 1, 128)
    assert abs(teacher_parameter_formula(small) - small_target) / small_target < 0.25


def test_conv_parameter_count_excludes_classifier():
    model = ThreeBlockCNN(1, (64, 128, 128), 0.25)
    assert conv_parameter_count(model) == trainable_parameter_count(model) - (
        128 * 10 + 10
    )
