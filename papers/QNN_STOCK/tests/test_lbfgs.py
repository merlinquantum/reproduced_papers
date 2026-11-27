"""L-BFGS training on the shot-free loss (E14c)."""

import numpy as np
import pytest
from lib.model import ContextualTask
from lib.training import train_qstl

MODEL_CFG = {
    "ansatz": "context_controlled",
    "interaction_order": 1,
    "loss": "swap_test_fidelity",
}


def lbfgs_cfg(**changes):
    cfg = {
        "optimizer": "lbfgs",
        "mode": "qbgu",
        "batch_size": 1,
        "epochs": 200,
        "monitor_every": 5,
        "init_max_angle": 0.3,
        "lr": 0.1,
        "spsa_perturbation": 0.01,
        "shots": None,
    }
    cfg.update(changes)
    return cfg


def random_task(seed):
    rng = np.random.default_rng(seed)
    joint = rng.random((8, 2))
    return ContextualTask(joint / joint.sum())


def test_lbfgs_loss_never_increases_and_ends_at_a_stationary_point():
    task = random_task(3)
    history = []
    circuit, params = train_qstl(
        task, MODEL_CFG, lbfgs_cfg(), np.random.default_rng(0), "qstl", history
    )
    losses = [row["swap_loss_exact"] for row in history]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(losses, losses[1:]))
    assert history[-1]["fidelity"] == pytest.approx(task.fidelity(circuit))

    # At a stationary point no single-angle change of 1e-3 improves the fidelity
    # by more than second order.
    from lib.training import qstl_circuit_factory

    build, _ = qstl_circuit_factory(MODEL_CFG, task.n_qubits, task.levels)
    best = task.fidelity(circuit)
    for index in range(len(params)):
        for sign in (-1.0, 1.0):
            moved = params.copy()
            moved[index] += sign * 1e-3
            assert task.fidelity(build(moved)) <= best + 1e-6


def test_lbfgs_is_not_worse_than_shot_free_spsa():
    task = random_task(4)
    lbfgs_history, spsa_history = [], []
    train_qstl(
        task, MODEL_CFG, lbfgs_cfg(), np.random.default_rng(1), "qstl", lbfgs_history
    )
    train_qstl(
        task,
        MODEL_CFG,
        lbfgs_cfg(optimizer="spsa", epochs=300),
        np.random.default_rng(1),
        "qstl",
        spsa_history,
    )
    assert lbfgs_history[-1]["fidelity"] >= spsa_history[-1]["fidelity"] - 1e-6


@pytest.mark.parametrize("changes", [{"shots": 100}, {"mode": "per_sample"}])
def test_lbfgs_rejects_a_stochastic_loss(changes):
    with pytest.raises(ValueError, match="lbfgs"):
        train_qstl(
            random_task(5),
            MODEL_CFG,
            lbfgs_cfg(**changes),
            np.random.default_rng(2),
            "qstl",
            [],
        )


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError, match="optimizer"):
        train_qstl(
            random_task(6),
            MODEL_CFG,
            lbfgs_cfg(optimizer="adam"),
            np.random.default_rng(2),
            "qstl",
            [],
        )
