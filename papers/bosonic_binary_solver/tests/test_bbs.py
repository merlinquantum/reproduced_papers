"""Anchors for the solver itself."""

import numpy as np
import pytest
from lib.bbs import BosonicBinarySolver
from lib.problems import build_problem
from lib.sampler import SOURCES, ClickSource
from lib.tbi import alternating_input, beamsplitter_layout


def test_evaluation_count_equals_appendix_b_bound():
    """The run must evaluate exactly N S (2 sum(m-l_i) + 2m + 1) candidates.

    This is the anchor for the accounting question: the bound only comes out
    right if the strings drawn while estimating gradients are counted as
    candidates, which is also what ORCA's released solver does.
    """
    _, cost_batch, _ = build_problem("knapsack", 8, 0)
    solver = BosonicBinarySolver(8, updates=5, samples=4, seed=0)
    result = solver.solve(cost_batch)
    assert result["evaluations"] == result["budget"] == solver.budget()


def test_solver_finds_the_optimum_on_a_small_instance():
    _, cost_batch, optimum = build_problem("knapsack", 10, 0)
    solver = BosonicBinarySolver(10, updates=60, samples=20, seed=0)
    result = solver.solve(cost_batch)
    assert result["best_cost"] == pytest.approx(optimum)


def test_flip_probabilities_start_at_one_half():
    solver = BosonicBinarySolver(6, seed=0)
    assert np.allclose(solver.flip_probabilities, 0.5)
    assert solver.theta.min() >= 0.0 and solver.theta.max() <= 2 * np.pi


@pytest.mark.parametrize("source", SOURCES)
def test_every_source_returns_valid_threshold_patterns(source):
    m, delays = 8, (1, 3)
    rng = np.random.default_rng(0)
    theta = rng.random(len(beamsplitter_layout(m, delays))) * 2 * np.pi
    click_source = ClickSource(m, delays, alternating_input(m), source=source)
    clicks = click_source.draw(theta, 32, rng)
    assert clicks.shape == (32, m)
    assert set(np.unique(clicks)) <= {0, 1}


def test_bernoulli_rate_scale_raises_click_density():
    """The density control must actually move density and nothing else."""
    m, delays = 12, (1, 3, 9)
    theta = (
        np.random.default_rng(0).random(len(beamsplitter_layout(m, delays))) * 2 * np.pi
    )
    densities = []
    for scale in (0.8, 1.0, 1.3):
        source = ClickSource(
            m, delays, alternating_input(m), source="bernoulli", rate_scale=scale
        )
        clicks = source.draw(theta, 400, np.random.default_rng(1))
        densities.append(clicks.mean())
    assert densities[0] < densities[1] < densities[2]
