"""Finite-shot statistics of the virtual estimator (Phase 5 equal-shot control, C8)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from common import PROJECT_DIR, load_runtime_ready_config
from virtual_qnn.lib import shot_budget

RNG = np.random.default_rng(0)


def _random_state(dim: int) -> np.ndarray:
    a = RNG.normal(size=(dim, dim)) + 1j * RNG.normal(size=(dim, dim))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def test_pauli_decomposition_reconstructs_matrix():
    a = RNG.normal(size=(4, 4)) + 1j * RNG.normal(size=(4, 4))
    hermitian = a + a.conj().T
    terms = shot_budget.pauli_decomposition(hermitian)
    rebuilt = sum(
        h * np.kron(shot_budget._PAULI[label[0]], shot_budget._PAULI[label[1]])
        for label, h in terms.items()
    )
    np.testing.assert_allclose(rebuilt, hermitian, atol=1e-12)


def test_virtual_estimator_mean_and_variance_match_monte_carlo():
    """C8: the quasi-probability estimator is unbiased with per-shot variance γ² − ⟨P⟩²."""
    budget = 2.5
    c1, c2 = (1 + budget) / 2, -(budget - 1) / 2
    rho1, rho2 = _random_state(4), _random_state(4)
    sigma = c1 * rho1 + c2 * rho2
    label = "XZ"
    exp1, exp2 = (
        shot_budget.pauli_expectation(label, rho1),
        shot_budget.pauli_expectation(label, rho2),
    )
    n = 400_000
    branch_two = RNG.random(n) < abs(c2) / budget
    p_plus = np.where(branch_two, (1 + exp2) / 2, (1 + exp1) / 2)
    outcomes = np.where(RNG.random(n) < p_plus, 1.0, -1.0)
    samples = budget * np.where(branch_two, np.sign(c2), np.sign(c1)) * outcomes
    target = shot_budget.pauli_expectation(label, sigma)
    assert samples.mean() == pytest.approx(target, abs=5 * budget / math.sqrt(n))
    assert samples.var() == pytest.approx(budget**2 - target**2, rel=1e-2)


def test_energy_rmse_scaling():
    a = RNG.normal(size=(4, 4))
    hamiltonian = (a + a.T).astype(complex)
    sigma = _random_state(4)
    exact = shot_budget.energy_estimator_rmse(hamiltonian, sigma, 1.0, math.inf, -1.0)
    assert exact["variance"] == 0.0 and exact["rmse"] == pytest.approx(
        abs(exact["bias"])
    )
    v1 = shot_budget.energy_estimator_rmse(hamiltonian, sigma, 1.0, 1000, -1.0)[
        "variance"
    ]
    v2 = shot_budget.energy_estimator_rmse(hamiltonian, sigma, 1.0, 4000, -1.0)[
        "variance"
    ]
    assert v1 == pytest.approx(4 * v2)
    v_virtual = shot_budget.energy_estimator_rmse(hamiltonian, sigma, 3.0, 1000, -1.0)[
        "variance"
    ]
    assert v_virtual > 8 * v1  # ≈ γ² growth (minus the ⟨P⟩² terms)


def test_expected_accuracy_matches_simulation_and_limit():
    mu = np.array([0.4, -0.2, 0.05, -0.6, 1.5, -1.1])
    labels = np.array([1, 0, 1, 1, 1, 0], dtype=float)
    budget, shots = 2.0, 7
    q = (1 + mu / budget) / 2
    n = 200_000
    plus = RNG.binomial(shots, q[:, None], size=(len(mu), n))
    m = budget * (2 * plus - shots) / shots
    simulated = np.mean((m >= 0) == (labels[:, None] == 1))
    assert shot_budget.expected_accuracy(mu, labels, budget, shots) == pytest.approx(
        simulated, abs=3e-3
    )
    assert shot_budget.expected_accuracy(mu, labels, budget, 100_000) == pytest.approx(
        shot_budget.expected_accuracy(mu, labels, budget, math.inf), abs=1e-6
    )


def test_equal_shot_e5_task_on_synthetic_sweep(tmp_path):
    pytest.importorskip("pennylane")
    from virtual_qnn.lib import runner

    sweep = tmp_path / "sweep"
    run_dirs = []
    for i, budget in enumerate((1.0, 2.5)):
        run = sweep / f"r{i}"
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(
            json.dumps({"budget": budget, "half_distance_bohr": 0.8})
        )
        np.savez(run / "virtual_state.npz", sigma=_random_state(4))
        run_dirs.append(
            {"run_id": f"r{i}", "status": "COMPLETED", "seed": 1, "run_dir": f"r{i}"}
        )
    (sweep / "sweep_status.json").write_text(
        json.dumps({"status": "COMPLETED", "runs": run_dirs})
    )
    cfg = load_runtime_ready_config()
    cfg["data_root"] = str(PROJECT_DIR.parents[1] / "data")
    cfg["experiment"]["task"] = "equal_shot_eval"
    cfg["equal_shot"] = {
        "experiment": "e5",
        "sweeps": {"vvqe": str(sweep)},
        "shot_grid": [100, 10000],
        "tie_tolerance_ha": 1e-4,
    }
    out = tmp_path / "out"
    out.mkdir()
    runner.train_and_evaluate(cfg, out)
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["decisions"]) == {"gamma=2.5|S=100", "gamma=2.5|S=10000"}
    assert (out / "equal_shot_e5_rows.csv").is_file()
    assert json.loads((out / "run_status.json").read_text())["status"] == "COMPLETED"
