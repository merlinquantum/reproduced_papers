"""Context loader (Sec. III, Fig. 4): Hadamard start and the phase-blind MSE."""

from __future__ import annotations

import numpy as np
import pytest
from common import load_project_defaults
from lib.circuits import apply_ansatz
from lib.training import train_context_loader

CONTEXT_MARGINAL = np.array([0.30, 0.07, 0.03, 0.07, 0.07, 0.03, 0.08, 0.35])


def _loader_cfg(**updates):
    cfg = dict(load_project_defaults()["model"]["context_loader"])
    cfg.update(epochs=3, shots=None, **updates)
    return cfg


@pytest.mark.parametrize("initial_hadamard", [False, True])
def test_loader_state_is_ansatz_on_the_configured_start(initial_hadamard):
    history = []
    state, params = train_context_loader(
        CONTEXT_MARGINAL,
        _loader_cfg(initial_hadamard=initial_hadamard),
        np.random.default_rng(0),
        "loader_test",
        history,
    )
    start = np.zeros(8, dtype=np.complex128)
    if initial_hadamard:
        start[:] = 1.0 / np.sqrt(8.0)
    else:
        start[0] = 1.0
    np.testing.assert_allclose(state, apply_ansatz(start, params, 3), atol=1e-12)


def test_magnitude_only_fidelity_bounds_the_state_fidelity():
    history = []
    train_context_loader(
        CONTEXT_MARGINAL,
        _loader_cfg(initial_hadamard=True),
        np.random.default_rng(1),
        "loader_test",
        history,
    )
    last = history[-1]
    assert last["fidelity_magnitudes_only"] >= last["fidelity_to_exact_state"] - 1e-12
    assert last["fidelity_magnitudes_only"] <= 1.0 + 1e-12


def test_initial_hadamard_is_required_and_boolean():
    cfg = _loader_cfg()
    del cfg["initial_hadamard"]
    with pytest.raises(KeyError):
        train_context_loader(
            CONTEXT_MARGINAL, cfg, np.random.default_rng(0), "loader_test", []
        )
    with pytest.raises(ValueError):
        train_context_loader(
            CONTEXT_MARGINAL,
            _loader_cfg(initial_hadamard="yes"),
            np.random.default_rng(0),
            "loader_test",
            [],
        )
