from __future__ import annotations

import math

import common  # noqa: F401
import pytest
import torch
from lib.photonic import TimeBinFeatureMap, fit_angles, n_tilings_for


@pytest.fixture(scope="module")
def small_map():
    # 8 modes / 4 photons keeps the Fock space tiny; same code path as 16 modes.
    return TimeBinFeatureMap(8, theta_dim=14, input_state=[1, 0] * 4)


def test_tiling_count_and_angle_fit():
    assert n_tilings_for(15, 16) == 1
    assert n_tilings_for(30, 16) == 2
    assert n_tilings_for(45, 16) == 3
    assert n_tilings_for(20, 16) == 2
    theta = torch.arange(20, dtype=torch.float64)
    assert fit_angles(theta, 30).tolist() == list(range(20)) + [0] * 10
    assert fit_angles(theta, 15).tolist() == list(range(15))


def test_zero_angles_are_identity(small_map):
    probabilities = small_map.outcome_probabilities(torch.zeros(14))
    keys = small_map.layer.output_keys
    assert keys[int(probabilities.argmax())] == (1, 0, 1, 0, 1, 0, 1, 0)
    assert probabilities.max().item() == pytest.approx(1.0)


@pytest.mark.parametrize("angle_index", [0, 3, 6, 7, 13])
def test_angle_order_matches_physical_beam_splitter(angle_index, small_map):
    # theta = pi is a full swap in Perceval's BS convention, so only the pair of
    # modes coupled by this angle exchanges occupation.
    theta = torch.zeros(14, dtype=torch.float64)
    theta[angle_index] = math.pi
    probabilities = small_map.outcome_probabilities(theta)
    mode = angle_index % 7
    expected = [1, 0] * 4
    expected[mode], expected[mode + 1] = expected[mode + 1], expected[mode]
    assert small_map.layer.output_keys[int(probabilities.argmax())] == tuple(expected)
    assert probabilities.max().item() == pytest.approx(1.0, abs=1e-9)


def test_exact_feature_is_two_normalised_marginals(small_map):
    feature = small_map.raw_feature(
        torch.zeros(14), None, torch.Generator().manual_seed(0)
    )
    assert feature.shape == (32,)
    first, second = feature[:16], feature[16:]
    assert first.sum().item() == pytest.approx(1.0)
    assert second.sum().item() == pytest.approx(1.0)
    # 1010 -> MSB-first index 10 for both halves.
    assert first[10].item() == pytest.approx(1.0)
    assert second[10].item() == pytest.approx(1.0)


def test_shot_feature_converges_to_exact(small_map):
    theta = torch.linspace(-1.0, 1.0, 14, dtype=torch.float64)
    exact = small_map.raw_feature(theta, None, torch.Generator().manual_seed(0))
    errors = []
    for shots in (100, 10_000):
        sampled = small_map.raw_feature(theta, shots, torch.Generator().manual_seed(1))
        assert sampled[:16].sum().item() == pytest.approx(1.0)
        errors.append((sampled - exact).abs().sum().item())
    assert errors[1] < errors[0] / 3


def test_threshold_detection_counts_bunched_outcomes(small_map):
    theta = torch.full((14,), 0.9, dtype=torch.float64)
    probabilities = small_map.outcome_probabilities(theta)
    occupations = torch.tensor(small_map.layer.output_keys)
    bunched_mass = probabilities[(occupations > 1).any(dim=1)].sum().item()
    assert bunched_mass > 0.01  # these outcomes would be lost in the UNBUNCHED space
    feature = small_map.raw_feature(theta, None, torch.Generator())
    assert feature[:16].sum().item() == pytest.approx(1.0)


def test_bits_feature_matches_index_convention(small_map):
    bits = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 0, 1]])
    feature = small_map.raw_feature_from_bits(bits)
    assert feature[10].item() == pytest.approx(1.0)
    assert feature[16 + 1].item() == pytest.approx(1.0)


def test_paper_size_fock_space_is_rejected_when_too_large():
    with pytest.raises(ValueError, match="Fock states"):
        TimeBinFeatureMap(16, 45, [1] * 14 + [0] * 2)
