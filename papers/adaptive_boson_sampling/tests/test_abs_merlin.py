from __future__ import annotations

import common  # noqa: F401
import numpy as np
import pytest
from lib.abs_merlin import (
    _accepts_detection,
    merlin_state_table,
    noisy_density_table,
    reconstruct_density_matrix,
    simulated_state_table,
    tomography_settings,
)
from lib.circuits import get_platform, random_u0, state_table
from lib.kernels import quantum_kernel


@pytest.mark.parametrize("name, u0_seed", [("A", None), ("B1", None), ("B3", 3)])
def test_merlin_exact_backend_matches_numpy_reference(name, u0_seed):
    platform = get_platform(name)
    u0 = None if u0_seed is None else random_u0(8, u0_seed)
    reference_states, reference_probabilities = state_table(platform, u0)
    states, probabilities = merlin_state_table(platform, u0)
    assert np.array_equal(np.isnan(states), np.isnan(reference_states))
    assert np.nanmax(np.abs(states - reference_states)) < 1e-10
    assert np.max(np.abs(probabilities - reference_probabilities)) < 1e-12


def test_bunched_b3_outcomes_are_post_selected_in_fock_space():
    # B3 encodes 5 bunched outcomes (both photons on one detector); they exist
    # only in the FOCK computation space.
    platform = get_platform("B3")
    states, _ = merlin_state_table(platform, random_u0(8, 3))
    bunched = [i for i, outcome in enumerate(platform.outcomes) if max(outcome) == 2]
    assert len(bunched) == 5
    assert not np.isnan(states[bunched, bunched]).any()


def test_noiseless_tomography_recovers_pure_states():
    platform = get_platform("A")
    density, probabilities = noisy_density_table(
        platform, None, {"indistinguishability": 1.0, "g2": 0.0, "detection": "pnr"}
    )
    states, reference_probabilities = state_table(platform, None)
    pure = np.einsum("cki,ckj->ckij", states, states.conj())
    assert np.nanmax(np.abs(density - pure)) < 1e-10
    assert np.max(np.abs(probabilities - reference_probabilities)) < 1e-12


def test_source_noise_reaches_the_layer_and_mixes_states():
    platform = get_platform("A")
    noise = {"indistinguishability": 0.83, "g2": 0.02, "detection": "threshold"}
    density, _ = noisy_density_table(platform, None, noise)
    purities = [np.trace(density[k, k] @ density[k, k]).real for k in range(3)]
    # Not every state mixes: for some outcomes the distinguishable components
    # leave the same residual state as the interfering ones.
    assert min(purities) < 0.99
    kernel = quantum_kernel(np.array([density[k, k] for k in range(3)]), "fidelity")
    ideal = quantum_kernel(
        state_table(platform, None)[0][[0, 1, 2], [0, 1, 2]], "overlap"
    )
    assert np.abs(kernel - ideal).max() > 1e-3


def test_reconstruction_recovers_known_qutrit_from_exact_frequencies():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho = vectors @ vectors.conj().T
    rho /= np.trace(rho)
    settings = tomography_settings(3)
    assert len(settings) == 7
    frequencies = np.array([np.diag(v @ rho @ v.conj().T).real for v in settings])
    assert np.allclose(reconstruct_density_matrix(settings, frequencies), rho)


def test_threshold_and_pnr_post_selection_rules():
    assert _accepts_detection((2, 1, 0), (1, 1, 0), "threshold")
    assert not _accepts_detection((2, 1, 0), (1, 1, 0), "pnr")
    # Bunched targets are resolved by count (pseudo-PNR) even with threshold.
    assert _accepts_detection((2, 0, 0), (2, 0, 0), "threshold")
    assert not _accepts_detection((3, 0, 0), (2, 0, 0), "threshold")


def test_unknown_backend_and_overlap_on_density_matrices_fail():
    platform = get_platform("A")
    with pytest.raises(ValueError, match="unknown backend"):
        simulated_state_table(platform, None, {"backend": "bogus"})
    with pytest.raises(ValueError, match="fidelity"):
        quantum_kernel(np.zeros((3, 2, 2), dtype=complex), "overlap")
