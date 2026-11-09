from __future__ import annotations

import sys

import numpy as np
from common import PROJECT_DIR

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lib.qrc_qubit import QubitQRC  # noqa: E402


def test_qrc_output_is_probability_vector():
    qrc = QubitQRC(n_qubits=4, num_features=8, n_random_gates=10, seed=0)
    h = qrc.initial_hidden()
    p = qrc.step(x_t=3, h_t=h)
    assert p.shape == (2**4,)
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p >= 0).all()


def test_qrc_depolarizing_makes_state_more_uniform():
    qrc_clean = QubitQRC(n_qubits=3, num_features=4, n_random_gates=20, seed=0)
    qrc_noisy = QubitQRC(
        n_qubits=3, num_features=4, n_random_gates=20, depolarizing_p=0.9, seed=0
    )
    h = qrc_clean.initial_hidden()
    p_clean = qrc_clean.step(x_t=1, h_t=h)
    p_noisy = qrc_noisy.step(x_t=1, h_t=h)
    uniform = np.full_like(p_clean, 1.0 / p_clean.size)
    assert np.linalg.norm(p_noisy - uniform) < np.linalg.norm(p_clean - uniform)


def test_qrc_step_is_deterministic_for_zero_shots():
    qrc = QubitQRC(n_qubits=4, num_features=8, n_random_gates=10, seed=0)
    h = qrc.initial_hidden()
    p1 = qrc.step(x_t=2, h_t=h)
    p2 = qrc.step(x_t=2, h_t=h)
    assert np.allclose(p1, p2)
