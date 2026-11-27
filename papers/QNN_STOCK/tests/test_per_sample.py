"""Per-sample SGD mode (E12): loss, sampling, objective and training loop."""

from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the repository root on sys.path)
from lib.circuits import apply_ansatz, n_params
from lib.model import ContextualTask, per_sample_fidelity
from lib.training import run_spsa, train_qstl


def random_joint(rng, n_contexts=8, levels=2):
    joint = rng.random((n_contexts, levels))
    return joint / joint.sum()


def conditional_circuit(conditional):
    """Linear map ``|c>|0> -> sum_o sqrt(q(o|c)) |c, o>`` (defined on that subspace)."""

    n_contexts, levels = conditional.shape

    def circuit(state):
        output = np.zeros_like(state)
        for context in range(n_contexts):
            output[context * levels : (context + 1) * levels] += state[
                context * levels
            ] * np.sqrt(conditional[context])
        return output

    return circuit


def test_per_sample_swap_loss_is_eq14_of_the_one_hot_fidelity():
    rng = np.random.default_rng(1)
    task = ContextualTask(random_joint(rng))
    params = rng.uniform(0, 2 * np.pi, n_params(4, 2))
    circuit = lambda s: apply_ansatz(s, params, 4)  # noqa: E731
    for context, outcome in [(0, 0), (5, 1), (7, 0)]:
        fidelity = per_sample_fidelity(circuit, context, outcome, 8, 2)
        loss = task.per_sample_swap_loss(circuit, context, outcome, None, rng)
        assert loss == pytest.approx(0.5 * (1 - fidelity), abs=1e-12)


def test_expected_per_sample_fidelity_is_the_window_average():
    rng = np.random.default_rng(2)
    task = ContextualTask(random_joint(rng))
    params = rng.uniform(0, 2 * np.pi, n_params(4, 2))
    circuit = lambda s: apply_ansatz(s, params, 4)  # noqa: E731
    expected = sum(
        task.joint[c, o] * per_sample_fidelity(circuit, c, o, 8, 2)
        for c in range(8)
        for o in range(2)
    )
    assert task.expected_per_sample_fidelity(circuit) == pytest.approx(expected)


def test_sample_window_follows_the_joint():
    rng = np.random.default_rng(3)
    task = ContextualTask(random_joint(rng))
    counts = np.zeros_like(task.joint)
    for _ in range(40_000):
        context, outcome = task.sample_window(rng)
        counts[context, outcome] += 1
    assert np.abs(counts / counts.sum() - task.joint).max() < 0.01


def test_per_sample_objective_prefers_the_mode_over_the_conditional():
    """The per-sample fidelity is linear in q(o|c), so argmax beats calibration."""

    joint = np.array([[0.3, 0.2], [0.1, 0.4]])
    task = ContextualTask(joint)
    conditional = joint / joint.sum(axis=1, keepdims=True)
    mode = np.eye(2)[conditional.argmax(axis=1)]
    calibrated = task.expected_per_sample_fidelity(conditional_circuit(conditional))
    deterministic = task.expected_per_sample_fidelity(conditional_circuit(mode))
    assert calibrated == pytest.approx(0.3 * 0.6 + 0.2 * 0.4 + 0.1 * 0.2 + 0.4 * 0.8)
    assert deterministic == pytest.approx(0.3 + 0.4)
    assert deterministic > calibrated
    # The QBGU fidelity prefers the calibrated conditional instead.
    assert task.fidelity(conditional_circuit(conditional)) == pytest.approx(1.0)
    assert task.fidelity(conditional_circuit(mode)) < 1.0


def test_run_spsa_draws_one_loss_per_iteration_and_thins_history():
    rng = np.random.default_rng(4)
    draws = []

    def draw_loss_fn():
        draws.append(1)
        return lambda params: float(np.sum(params**2))

    history = []
    run_spsa(
        None,
        lambda params: {"norm": float(np.linalg.norm(params))},
        np.ones(3),
        10,
        {"lr": 0.1, "spsa_perturbation": 0.01},
        rng,
        "stage",
        history,
        monitor_every=4,
        draw_loss_fn=draw_loss_fn,
    )
    assert len(draws) == 10
    assert [row["stage_epoch"] for row in history] == [4, 8, 10]
    with pytest.raises(ValueError):
        run_spsa(
            None,
            lambda p: {},
            np.ones(1),
            1,
            {"lr": 0.1, "spsa_perturbation": 0.01},
            rng,
            "s",
            [],
        )


@pytest.mark.parametrize("mode", ["qbgu", "per_sample"])
def test_train_qstl_runs_in_both_modes(mode):
    rng = np.random.default_rng(5)
    task = ContextualTask(random_joint(rng))
    history = []
    model_cfg = {"ansatz": "hea", "layers": 1, "loss": "swap_test_fidelity"}
    training_cfg = {
        "optimizer": "spsa",
        "mode": mode,
        "epochs": 6,
        "monitor_every": 3,
        "init_max_angle": 2 * np.pi,
        "batch_size": 1,
        "lr": 0.1,
        "spsa_perturbation": 0.01,
        "shots": 100,
    }
    train_qstl(task, model_cfg, training_cfg, rng, "qstl", history)
    assert [row["stage_epoch"] for row in history] == [3, 6]
    assert {"fidelity", "per_sample_fidelity", "loss_estimate_mean"} <= set(history[0])


def test_train_qstl_rejects_unknown_mode():
    rng = np.random.default_rng(6)
    task = ContextualTask(random_joint(rng))
    model_cfg = {"ansatz": "hea", "layers": 1, "loss": "swap_test_fidelity"}
    training_cfg = {
        "optimizer": "spsa",
        "mode": "minibatch",
        "epochs": 1,
        "monitor_every": 1,
        "init_max_angle": 2 * np.pi,
        "lr": 0.1,
        "spsa_perturbation": 0.01,
        "shots": None,
    }
    with pytest.raises(ValueError, match="training.mode"):
        train_qstl(task, model_cfg, training_cfg, rng, "qstl", [])


def test_minibatch_loss_is_the_count_weighted_per_sample_loss():
    rng = np.random.default_rng(7)
    task = ContextualTask(random_joint(rng))
    params = rng.uniform(0, 2 * np.pi, n_params(4, 2))
    circuit = lambda s: apply_ansatz(s, params, 4)  # noqa: E731
    counts = rng.multinomial(50, task.joint.reshape(-1)).reshape(task.joint.shape)
    expected = (
        sum(
            counts[c, o] * task.per_sample_swap_loss(circuit, c, o, None, rng)
            for c in range(8)
            for o in range(2)
        )
        / counts.sum()
    )
    assert task.minibatch_swap_loss(circuit, counts, None, rng) == pytest.approx(
        expected
    )
    # With shots the estimate is unbiased and close for many shots.
    estimates = [
        task.minibatch_swap_loss(circuit, counts, 2000, rng) for _ in range(20)
    ]
    assert np.mean(estimates) == pytest.approx(expected, abs=2e-3)


def test_minibatch_requires_shots_divisible_by_batch():
    rng = np.random.default_rng(8)
    task = ContextualTask(random_joint(rng))
    model_cfg = {"ansatz": "hea", "layers": 1, "loss": "swap_test_fidelity"}
    training_cfg = {
        "optimizer": "spsa",
        "mode": "per_sample",
        "batch_size": 3,
        "epochs": 1,
        "monitor_every": 1,
        "init_max_angle": 2 * np.pi,
        "lr": 0.1,
        "spsa_perturbation": 0.01,
        "shots": 100,
    }
    with pytest.raises(ValueError, match="multiple"):
        train_qstl(task, model_cfg, training_cfg, rng, "qstl", [])


def test_train_qstl_runs_with_a_minibatch_on_the_context_preserving_circuit():
    rng = np.random.default_rng(9)
    task = ContextualTask(random_joint(rng))
    model_cfg = {
        "ansatz": "context_controlled",
        "interaction_order": 1,
        "loss": "swap_test_fidelity",
    }
    training_cfg = {
        "optimizer": "spsa",
        "mode": "per_sample",
        "batch_size": 100,
        "epochs": 4,
        "monitor_every": 2,
        "init_max_angle": 2 * np.pi,
        "lr": 0.1,
        "spsa_perturbation": 0.01,
        "shots": 10_000,
    }
    history = []
    train_qstl(task, model_cfg, training_cfg, rng, "qstl", history)
    assert [row["stage_epoch"] for row in history] == [2, 4]
