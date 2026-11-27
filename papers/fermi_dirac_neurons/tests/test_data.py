"""Data generator contract from the LOG.md Data Acquisition Log specification."""

from __future__ import annotations

import copy
import json

import common  # noqa: F401  (sets sys.path)
import numpy as np
import pytest
from common import PROJECT_DIR, load_runtime_ready_config
from fermi_dirac_neurons.lib.data import (
    build_dataloader,
    build_states,
    build_target,
    haar_pure_states,
    load_reference_curve,
)

from runtime_lib.config import deep_update, load_config


def _config(name: str | None = None) -> dict:
    cfg = load_runtime_ready_config()
    if name:
        deep_update(cfg, load_config(PROJECT_DIR / "configs" / f"{name}.json"))
    return cfg


def _assert_density_matrices(states):
    np.testing.assert_allclose(states, states.conj().transpose(0, 2, 1), atol=1e-14)
    np.testing.assert_allclose(np.trace(states, axis1=1, axis2=2), 1.0, atol=1e-12)
    assert min(np.linalg.eigvalsh(state).min() for state in states) > -1e-12


@pytest.mark.parametrize("n_qubits", [2, 3])
@pytest.mark.parametrize(
    "state_set", ["code_squared_loss", "paper_squared_loss", "pauli_basis"]
)
def test_state_sets_are_density_matrices_with_expected_counts(n_qubits, state_set):
    cfg = {"set": state_set, "p_values": [0.1, 0.5, 0.9]}
    states = build_states(cfg, n_qubits, np.random.default_rng(0))
    expected = {
        "code_squared_loss": 2 ** (n_qubits + 1) + 5,
        "paper_squared_loss": 2 ** (n_qubits + 1) + 4 + 1 + 3 + 1,
        "pauli_basis": 3 * 2**n_qubits,
    }[state_set]
    assert states.shape == (expected, 2**n_qubits, 2**n_qubits)
    assert states.dtype == np.complex128
    _assert_density_matrices(states)


def test_haar_states_are_pure_with_uniform_overlap():
    states = haar_pure_states(3, 4000, np.random.default_rng(1))
    _assert_density_matrices(states)
    np.testing.assert_allclose(
        np.einsum("mab,mba->m", states, states).real, 1.0, atol=1e-12
    )
    assert np.real(states[:, 0, 0]).mean() == pytest.approx(1 / 8, abs=0.01)


def test_one_batch_loader_shapes():
    cfg = _config()
    states, targets = next(iter(build_dataloader(cfg, split="train")))
    assert states.shape == (13, 4, 4)
    assert targets.shape == (13,)
    states, targets = next(iter(build_dataloader(cfg, split="validation")))
    assert states.shape == (500, 4, 4)


@pytest.mark.parametrize("rule", ["sign_linear", "sign_activation"])
def test_classification_labels_are_signs(rule):
    cfg = _config("fig8_logistic_2q_original")
    cfg["data"]["label_rule"] = rule
    _, targets = next(iter(build_dataloader(cfg, split="train")))
    assert set(np.unique(targets)) == {-1.0, 1.0}


def test_seed_reproducibility_and_pairing():
    first = _config("fig7_tanh_2q_original")
    second = _config("fig7_tanh_2q_classical")
    batch_a = next(iter(build_dataloader(first, "train")))
    batch_b = next(iter(build_dataloader(second, "train")))
    np.testing.assert_array_equal(batch_a[0], batch_b[0])
    np.testing.assert_array_equal(batch_a[1], batch_b[1])
    other = copy.deepcopy(first)
    other["seed"] = 1
    assert not np.allclose(build_target(first).zeta, build_target(other).zeta)


def test_target_coefficients_in_range_and_term_count():
    cfg = _config("fig8_logistic_4q_original")
    target = build_target(cfg)
    assert len(target.zeta) == 6 * 4 - 3
    assert np.all(np.abs(target.zeta) <= 2.0)


def test_reference_curves_are_readable():
    cfg = _config()
    cfg["data_root"] = str(PROJECT_DIR.parent.parent / "data")
    try:
        curve = load_reference_curve(cfg, "mse_tanh_2qubit_ising.csv")
    except FileNotFoundError:
        pytest.skip("authors' reference outputs not cached (see LOG.md)")
    assert len(curve["Quantum_Loss"]) == 2000
    assert curve["Quantum_Loss"][-1] < curve["Classical_Loss"][-1]


def test_default_data_root_is_the_repository_data_folder():
    # The paper folder carries an implementation.py shim, so the runtime's
    # repo-root detection stops at the paper folder; defaults.json must point
    # at the shared data root explicitly.
    from runtime_lib.data_paths import resolve_data_root

    defaults = json.loads((PROJECT_DIR / "configs" / "defaults.json").read_text())
    resolved = resolve_data_root(defaults["data_root"], PROJECT_DIR)
    assert resolved == (PROJECT_DIR.parent.parent / "data").resolve()


def test_reference_outputs_cache_is_under_the_data_root():
    # The authors' outputs are a git-ignored local cache (their repository has
    # no license); on a fresh clone recreate them with the README commands.
    from runtime_lib.data_paths import resolve_data_root

    defaults = json.loads((PROJECT_DIR / "configs" / "defaults.json").read_text())
    resolved = resolve_data_root(defaults["data_root"], PROJECT_DIR)
    outputs = resolved / "fermi_dirac_neurons" / "reference_outputs" / "outputs"
    if not outputs.parent.exists():
        pytest.skip("authors' reference outputs not cached (see README, Data)")
    assert outputs.is_dir()


@pytest.mark.parametrize(
    "state_cfg",
    [
        {"set": "code_squared_loss"},
        {"set": "paper_squared_loss", "p_values": [0.1, 0.5, 0.9]},
        {"set": "pauli_basis"},
        {"set": "computational_basis"},
        {"set": "haar", "n_states": 7},
    ],
)
def test_state_count_matches_build_states(state_cfg):
    from fermi_dirac_neurons.lib.data import state_count

    for n_qubits in (2, 3):
        states = build_states(state_cfg, n_qubits, np.random.default_rng(0))
        assert state_count(state_cfg, n_qubits) == len(states)


def test_explicit_teacher_target_uses_given_labels():
    cfg = _config()
    cfg["data"]["target"] = {
        "term_set": "explicit",
        "term_labels": ["XZ", "IY"],
        "include_identity": False,
        "low": -2.0,
        "high": 2.0,
        "function": "tanh",
    }
    target = build_target(cfg)
    assert target.labels == ["XZ", "IY"]
    assert len(target.zeta) == 2
