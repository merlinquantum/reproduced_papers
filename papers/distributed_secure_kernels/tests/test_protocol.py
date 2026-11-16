from __future__ import annotations

import common  # noqa: F401
import numpy as np
from distributed_secure_kernels.lib.protocol import (
    centralised_circuit,
    client_bit_counts,
    distributed_circuit,
    figure2_circuit,
    independent_seed,
    run_ancilla_zeros,
)
from scipy import stats


def test_distributed_and_centralised_circuits_follow_the_ideal_binomial_law():
    rng = np.random.default_rng(0)
    shots, statistic, pairs = 1024, {"distributed": 0.0, "centralised": 0.0}, 6
    for pair in range(pairs):
        a, b = rng.normal(size=4), rng.normal(size=4)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        p0 = 0.5 + 0.5 * (a @ b) ** 2
        for name, circuit in (
            ("distributed", distributed_circuit(a, b)),
            ("centralised", centralised_circuit(a, b)),
        ):
            zeros = run_ancilla_zeros(
                circuit, shots, 0.0, seed=independent_seed(1, pair)
            )
            statistic[name] += (zeros - shots * p0) ** 2 / (shots * p0 * (1 - p0))
    for value in statistic.values():
        assert stats.chi2.sf(value, pairs) > 0.001


def test_figure2_as_drawn_follows_the_ideal_binomial_law():
    # The paper's Fig. 2 (6n + 1 qubits, textbook teleportation, no register
    # reuse) must give the same law as the authors' code on dense, entangled states.
    rng = np.random.default_rng(11)
    shots, statistic, pairs = 1024, 0.0, 6
    for pair in range(pairs):
        a, b = rng.normal(size=4), rng.normal(size=4)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        p0 = 0.5 + 0.5 * (a @ b) ** 2
        zeros = run_ancilla_zeros(
            figure2_circuit(a, b), shots, 0.0, seed=independent_seed(11, pair)
        )
        statistic += (zeros - shots * p0) ** 2 / (shots * p0 * (1 - p0))
    assert stats.chi2.sf(statistic, pairs) > 0.001
    assert figure2_circuit(a, b).num_qubits == 6 * 2 + 1


def test_client_messages_are_uniform_whatever_the_data():
    for a in (np.array([1.0, 0, 0, 0]), np.array([0.5, 0.5, 0.5, 0.5])):
        fractions = client_bit_counts(distributed_circuit(a, a[::-1]), 4096, seed=7)
        for values in fractions.values():
            assert np.max(np.abs(values - 0.5)) < 5 * 0.5 / np.sqrt(4096)


def test_spaced_seeds_give_binomial_dispersion():
    # Consecutive Aer seeds share almost all per-shot random numbers; spaced
    # seeds must give the binomial variance.
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=4), rng.normal(size=4)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    p0, shots = 0.5 + 0.5 * (a @ b) ** 2, 1024
    zeros = np.array(
        [
            run_ancilla_zeros(
                centralised_circuit(a, b), shots, 0.0, seed=independent_seed(5, i)
            )
            for i in range(200)
        ]
    )
    ratio = zeros.var(ddof=1) / (shots * p0 * (1 - p0))
    assert 0.7 < ratio < 1.4
