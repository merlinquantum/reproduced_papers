"""Anchors for the batched solver.

The boson source needs the GPU sampler, so these run the interference-free
sources, which are pure torch: what they check is the batching, the accounting
and the agreement with the scalar CPU path, all of which are source-independent.
"""

import math

import numpy as np
import pytest
import torch
from lib.bbs_gpu import solve_batch
from lib.problems import build_problem
from lib.problems_torch import build_batch
from lib.sampler import ClickSource
from lib.sampler_torch import draw_clicks
from lib.tbi import alternating_input, beamsplitter_layout, shifted_angles, unitary


def test_batched_costs_match_the_scalar_definition():
    seeds = [0, 1, 2]
    for family, m in (("knapsack", 10), ("tsp", 10)):
        batched, optima = build_batch(family, m, seeds, torch.device("cpu"))
        bits = np.random.default_rng(0).integers(0, 2, (len(seeds), 64, m))
        got = batched(torch.tensor(bits)).numpy()
        expected = np.stack(
            [
                build_problem(family, m, seed)[1](bits[slot])
                for slot, seed in enumerate(seeds)
            ]
        )
        assert np.abs(got - expected).max() < 1e-9
        for slot, seed in enumerate(seeds):
            assert np.isclose(optima[slot], build_problem(family, m, seed)[2])


def test_shifted_angles_layout():
    """Row 0 unshifted, then +phi and -phi per angle. A wrong layout would make
    every gradient a mixture of two parameters and still look plausible."""
    theta = torch.tensor([[0.5, 1.5, 2.5]], dtype=torch.float64)
    stacked = shifted_angles(theta, math.pi / 2)
    assert stacked.shape == (1, 7, 3)
    assert torch.allclose(stacked[0, 0], theta[0])
    for k in range(3):
        assert stacked[0, 1 + 2 * k, k] == pytest.approx(
            float(theta[0, k]) + math.pi / 2
        )
        assert stacked[0, 2 + 2 * k, k] == pytest.approx(
            float(theta[0, k]) - math.pi / 2
        )
        others = [j for j in range(3) if j != k]
        assert torch.allclose(stacked[0, 1 + 2 * k, others], theta[0, others])


def test_evaluation_count_equals_appendix_b_bound():
    seeds = [0, 1]
    cost_batch, _ = build_batch("knapsack", 8, seeds, torch.device("cpu"))
    result = solve_batch(cost_batch, 8, seeds, updates=4, samples=4, source="bernoulli")
    assert result["evaluations"] == result["budget"]


def test_batched_and_scalar_sources_agree_in_distribution():
    """The batched torch sources must reproduce lib.sampler's per-mode rates."""
    m, delays = 12, (1, 3, 9)
    theta = (
        np.random.default_rng(0).random(len(beamsplitter_layout(m, delays))) * 2 * np.pi
    )
    input_state = alternating_input(m)
    modes = [i for i, n in enumerate(input_state) if n]
    unitaries = unitary(torch.tensor(theta), m, delays)[None]
    generator = torch.Generator().manual_seed(0)
    shots = 20000
    for source in ("distinguishable", "bernoulli"):
        scalar = (
            ClickSource(m, delays, input_state, source=source)
            .draw(theta, shots, np.random.default_rng(1))
            .mean(0)
        )
        batched = (
            draw_clicks(unitaries, modes, shots, generator, source, m)
            .to(torch.float64)
            .mean(dim=(0, 1))
            .numpy()
        )
        # three standard errors of a Bernoulli rate at this shot count
        assert np.abs(scalar - batched).max() < 3 * math.sqrt(0.25 / shots) + 0.005


def test_freeze_theta_holds_the_interferometer_still():
    """The ablation must actually freeze the angles.

    Expressed as a flag rather than lr_theta = 0 because a falsy learning-rate
    check silently runs the un-ablated model and reports it as the ablation.
    """
    seeds = [0, 1]
    cost_batch, _ = build_batch("knapsack", 8, seeds, torch.device("cpu"))
    frozen = solve_batch(
        cost_batch,
        8,
        seeds,
        updates=6,
        samples=8,
        source="bernoulli",
        freeze_theta=True,
    )
    trained = solve_batch(
        cost_batch,
        8,
        seeds,
        updates=6,
        samples=8,
        source="bernoulli",
        freeze_theta=False,
    )
    assert frozen["evaluations"] == trained["evaluations"]
    assert frozen["history"] != trained["history"]


def test_bernoulli_rate_scale_moves_click_density_only():
    seeds = [0, 1]
    cost_batch, _ = build_batch("knapsack", 10, seeds, torch.device("cpu"))
    densities = []
    for scale in (0.9, 1.0, 1.2):
        result = solve_batch(
            cost_batch,
            10,
            seeds,
            updates=4,
            samples=32,
            source="bernoulli",
            rate_scale=scale,
        )
        densities.append(float(np.mean(result["mean_clicks"])))
    assert densities[0] < densities[1] < densities[2]
