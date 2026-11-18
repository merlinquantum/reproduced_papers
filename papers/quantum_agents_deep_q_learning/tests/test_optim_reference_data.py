"""Masked Adam, analytic Frozen Lake Q*, environments and replay memory."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from common import PROJECT_DIR
from lib.data import ReplayMemory, build_dataloader, make_environment, state_bits
from lib.optim import MaskedAdam
from lib.reference import DOWN, LEFT, RIGHT, UP, optimal_q_values, terminal_states


def load_config(name: str) -> dict:
    return json.loads((PROJECT_DIR / "configs" / f"{name}.json").read_text())


# ------------------------------------------------------------------- optimiser
def _keras_adam_reference(gradients, lr, amsgrad, beta1=0.9, beta2=0.999, epsilon=1e-7):
    parameter, m, v, vhat = 0.0, 0.0, 0.0, 0.0
    for step, gradient in enumerate(gradients, start=1):
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2
        vhat = max(vhat, v)
        lr_t = lr * np.sqrt(1 - beta2**step) / (1 - beta1**step)
        parameter -= lr_t * m / (np.sqrt(vhat if amsgrad else v) + epsilon)
    return parameter


@pytest.mark.parametrize("amsgrad", [True, False])
def test_masked_adam_matches_keras_rule_and_skips_masked_agents(amsgrad):
    tensor = torch.nn.Parameter(torch.zeros(2, 1, dtype=torch.float64))
    optimizer = MaskedAdam(
        [([tensor], 0.01, amsgrad)], beta1=0.9, beta2=0.999, epsilon=1e-7
    )
    gradients = [0.5, -2.0, 0.1, 3.0]
    masks = [[True, False], [True, True], [True, False], [True, True]]
    for gradient, mask in zip(gradients, masks):
        tensor.grad = torch.full((2, 1), gradient, dtype=torch.float64)
        optimizer.step(torch.tensor(mask))
    expected_agent0 = _keras_adam_reference(gradients, 0.01, amsgrad)
    expected_agent1 = _keras_adam_reference([-2.0, 3.0], 0.01, amsgrad)
    assert tensor[0, 0].item() == pytest.approx(expected_agent0, rel=1e-12)
    assert tensor[1, 0].item() == pytest.approx(expected_agent1, rel=1e-12)


# ------------------------------------------------------------------ reference
def test_frozenlake_optimal_q_values_match_paper_formula():
    gamma = 0.8
    q_star = optimal_q_values(gamma)
    assert q_star[14, RIGHT] == pytest.approx(1.0)
    assert q_star[14, DOWN] == pytest.approx(gamma)
    assert q_star[0, DOWN] == pytest.approx(gamma**5)
    assert q_star[0, LEFT] == pytest.approx(gamma**6)
    assert q_star[1, DOWN] == 0.0  # into hole 5
    assert q_star[10, DOWN] == pytest.approx(gamma)
    assert q_star[13, UP] == pytest.approx(gamma**3)
    for state in terminal_states():
        assert np.all(q_star[state] == 0.0)
    assert sorted(terminal_states()) == [5, 7, 11, 12, 15]


# ---------------------------------------------------------------- environments
def test_environment_limits_match_reference_setting():
    cartpole = make_environment(load_config("e2e3_cartpole_optimal_full_original"))
    assert cartpole.spec.max_episode_steps == 200
    frozen = make_environment(load_config("e1_frozenlake_l5_original"))
    assert frozen.spec.max_episode_steps == 100
    assert frozen.unwrapped.spec.kwargs["is_slippery"] is False
    frozen_long = make_environment(load_config("e1_frozenlake_l5_cap200_variant"))
    assert frozen_long.spec.max_episode_steps == 200


def test_frozenlake_is_deterministic():
    env = make_environment(load_config("e1_frozenlake_l5_original"))
    env.reset(seed=0)
    for action, expected in (
        (RIGHT, 1),
        (RIGHT, 2),
        (DOWN, 6),
        (DOWN, 10),
        (DOWN, 14),
        (RIGHT, 15),
    ):
        state, reward, terminated, _, _ = env.step(action)
        assert state == expected
    assert reward == 1.0 and terminated


def test_state_bits_most_significant_first():
    assert state_bits(np.array([1])).tolist() == [[0, 0, 0, 1]]
    assert state_bits(np.array([8])).tolist() == [[1, 0, 0, 0]]


# ------------------------------------------------------------------ replay
def _filled_memory(capacity, eviction, n):
    memory = ReplayMemory(
        capacity, 1, eviction, "with_replacement", np.random.default_rng(0)
    )
    for index in range(n):
        memory.add([index], 0, 0.0, [index], False)
    return memory


def test_replay_fifo_keeps_latest():
    memory = _filled_memory(5, "fifo", 12)
    assert len(memory) == 5
    assert sorted(memory._states[:, 0].tolist()) == [7, 8, 9, 10, 11]


def test_replay_unbounded_grows():
    memory = _filled_memory(None, "fifo", 5000)
    assert len(memory) == 5000


def test_replay_without_replacement_needs_enough_samples():
    memory = ReplayMemory(
        10, 1, "random", "without_replacement", np.random.default_rng(0)
    )
    memory.add([0], 0, 0.0, [0], False)
    with pytest.raises(ValueError):
        memory.sample(2)


def test_dataloader_batch_shapes():
    for name, shape in (
        ("e2e3_cartpole_optimal_full_original", (16, 4)),
        ("e1_frozenlake_l5_original", (11, 1)),
    ):
        batch = next(iter(build_dataloader(load_config(name), split="train")))
        assert batch[0].shape == shape
        assert batch[1].shape == (shape[0],)


def test_dataloader_rejects_unknown_split():
    with pytest.raises(ValueError):
        next(
            iter(
                build_dataloader(load_config("e1_frozenlake_l5_original"), split="test")
            )
        )


def test_missing_config_key_fails_clearly():
    cfg = load_config("e1_frozenlake_l5_original")
    del cfg["training"]["replay_capacity"]
    with pytest.raises(KeyError, match="training.replay_capacity"):
        next(iter(build_dataloader(cfg, split="train")))


def test_masked_adam_rejects_missing_learning_rate():
    tensor = torch.nn.Parameter(torch.zeros(1, 1))
    with pytest.raises(ValueError):
        MaskedAdam([([tensor], None, False)], 0.9, 0.999, 1e-7)
