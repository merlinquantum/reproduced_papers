"""Tests for the MerLin photonic adaptation (Phase 4)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

PAPER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PAPER_ROOT.parents[1]
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT))

from lib import runner  # noqa: E402
from lib.pe_factory import _pe_for_one_graph  # noqa: E402
from lib.photonic import (  # noqa: E402
    _merlin_probabilities,
    _permanent_distribution,
    graph_unitary,
    photonic_cqrw_features,
    photonic_walk_covariance,
    photonic_walk_features,
)
from lib.qpe import adjacency_from_edges, cqrw_features, rrwp  # noqa: E402
from lib.srg import run_srg_photonic_experiment  # noqa: E402

from runtime_lib.data_paths import paper_data_dir  # noqa: E402

SRG_DIRECTORY = paper_data_dir("QPE_GNN", ensure_exists=False) / "srg"


def _random_graph(num_nodes: int = 7, seed: int = 3) -> np.ndarray:
    return nx.to_numpy_array(nx.gnp_random_graph(num_nodes, 0.45, seed=seed))


def test_photonic_unitary_is_unitary():
    A = adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    U = graph_unitary(A, t=0.7)
    np.testing.assert_allclose(U @ U.conj().T, np.eye(4), atol=1e-10)


def test_photonic_1cqrw_matches_xy_cqrw():
    """Single-photon photonic walk IS the 1-CQRW under the XY hamiltonian."""
    A = adjacency_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)])
    times = [0.2, 0.7, 1.4]
    P_phot = photonic_cqrw_features(A, 1, times)
    P_qpe = cqrw_features(A, 1, times)
    np.testing.assert_allclose(P_phot, P_qpe, atol=1e-10)


def test_photonic_cqrw_features_rejects_multiphoton_requests():
    with pytest.raises(ValueError, match="photonic_walk_covariance"):
        photonic_cqrw_features(_random_graph(), 2, [0.3])


@pytest.mark.parametrize("detection", ["threshold", "pnr"])
@pytest.mark.parametrize("n_photons", [1, 2])
def test_closed_form_backend_matches_merlin(detection, n_photons):
    A = _random_graph()
    for t in (0.37, 1.0, 2.3):
        closed_form = photonic_walk_covariance(
            A, t, n_photons=n_photons, detection=detection, backend="permanent"
        )
        simulated = photonic_walk_covariance(
            A, t, n_photons=n_photons, detection=detection, backend="merlin"
        )
        np.testing.assert_allclose(closed_form, simulated, atol=1e-12)


def test_threshold_detection_uses_the_collision_free_postselected_space():
    """UNBUNCHED keys are collision-free and equal FOCK renormalised on them."""
    A = _random_graph()
    unitary = graph_unitary(A, 0.7)
    num_modes = A.shape[0]
    threshold_keys, threshold = _merlin_probabilities(unitary, 2, "UNBUNCHED")
    fock_keys, fock = _merlin_probabilities(unitary, 2, "FOCK")
    assert len(threshold_keys) == math.comb(num_modes, 2)
    assert all(max(key) <= 1 for key in threshold_keys)
    assert len(fock_keys) == math.comb(num_modes + 1, 2)
    fock_index = {key: index for index, key in enumerate(fock_keys)}
    collision_free = fock[[fock_index[key] for key in threshold_keys]]
    np.testing.assert_allclose(
        threshold, collision_free / collision_free.sum(axis=0), atol=1e-12
    )
    np.testing.assert_allclose(threshold.sum(axis=0), 1.0, atol=1e-12)


def test_postselection_success_rate_agrees_between_backends():
    A = _random_graph()
    _, closed_form = photonic_walk_covariance(
        A, 0.7, backend="permanent", return_success=True
    )
    _, simulated = photonic_walk_covariance(
        A, 0.7, backend="merlin", return_success=True
    )
    assert 0.0 < closed_form < 1.0
    assert closed_form == pytest.approx(simulated, abs=1e-12)


def test_covariance_is_permutation_equivariant():
    A = _random_graph(num_nodes=8, seed=5)
    permutation = np.random.default_rng(0).permutation(8)
    C = photonic_walk_covariance(A, 1.1, backend="permanent")
    C_permuted = photonic_walk_covariance(
        A[np.ix_(permutation, permutation)], 1.1, backend="permanent"
    )
    np.testing.assert_allclose(
        C_permuted, C[np.ix_(permutation, permutation)], atol=1e-12
    )


@pytest.mark.parametrize("particles", ["bosons", "distinguishable"])
def test_number_resolving_distribution_is_normalised(particles):
    unitary = graph_unitary(_random_graph(), 0.9)
    _, probabilities, _ = _permanent_distribution(unitary, 2, "pnr", particles)
    np.testing.assert_allclose(probabilities.sum(axis=0), 1.0, atol=1e-12)


def test_distinguishable_photons_have_independent_single_photon_marginals():
    A = _random_graph()
    unitary = graph_unitary(A, 0.9)
    occupation, probabilities, _ = _permanent_distribution(
        unitary, 2, "pnr", "distinguishable"
    )
    np.testing.assert_allclose(probabilities.sum(axis=0), 1.0, atol=1e-12)
    marginals = occupation.T @ probabilities
    single = np.abs(unitary) ** 2  # (output mode, input mode)
    pairs = [(a, b) for a in range(A.shape[0]) for b in range(a + 1, A.shape[0])]
    expected = np.stack([single[:, a] + single[:, b] for a, b in pairs], axis=1)
    np.testing.assert_allclose(marginals, expected, atol=1e-12)


def test_backend_options_are_validated():
    A = _random_graph()
    with pytest.raises(ValueError, match="indistinguishable"):
        photonic_walk_covariance(A, 0.5, particles="distinguishable", backend="merlin")
    with pytest.raises(ValueError, match="one or two photons"):
        photonic_walk_covariance(A, 0.5, n_photons=3, backend="permanent")
    with pytest.raises(ValueError, match="detection"):
        photonic_walk_covariance(A, 0.5, detection="ideal")


def test_photon2_positional_encoding_concatenates_rrwp_and_photonic_channels():
    A = _random_graph()
    times = [0.4, 1.3]
    features = _pe_for_one_graph(
        A, "rrwp+photon2", 5, times=times, rrwp_dim=3, qpe_dim=2
    )
    assert features.shape == (5, 7, 7)
    np.testing.assert_allclose(features[:3], rrwp(A, 3))
    np.testing.assert_allclose(features[3:], photonic_walk_features(A, times))
    with pytest.raises(ValueError, match="one time per QPE channel"):
        _pe_for_one_graph(A, "rrwp+photon2", 5, times=[0.4], rrwp_dim=3, qpe_dim=2)


def test_photon2_run_writes_the_merlin_cross_check(tmp_path):
    config = {
        "description": "photonic smoke",
        "feasibility": {"status": "supported"},
        "dataset": "graph_reg",
        "dataset_kwargs": {"num_graphs": 24, "n_range": [5, 7], "p": 0.4, "seed": 0},
        "model": "grit_lite",
        "encoding": "rrwp+photon2",
        "rrwp_dim": 3,
        "qpe_dim": 2,
        "qpe_initial_distribution": "collision_free_pairs",
        "qpe_min_time": 0.1,
        "qpe_max_time": math.pi,
        "node_dim": 8,
        "depth": 1,
        "num_heads": 2,
        "head": "graph_reg",
        "epochs": 1,
        "batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "train_frac": 0.7,
        "val_frac": 0.15,
        "device": "cpu",
        "seed": 1,
    }
    output = runner.train_and_evaluate(config, tmp_path)
    check = json.loads((tmp_path / "photonic_check.json").read_text())
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    # The training split is smaller than the 64-graph check, so it is checked whole.
    assert check["graphs_checked"] == status["dataset"]["splits"]["train"]
    assert check["times"] == runner._resolve_cqrw_times(config, 2)
    assert check["max_abs_deviation"] <= 1e-8
    assert 0.0 < check["mean_postselection_success"] < 1.0
    assert output["photonic_check"] == check


def test_photon2_config_requires_its_initial_distribution():
    config = json.loads(
        (PAPER_ROOT / "configs" / "zinc_photon2_reduced.json").read_text()
    )
    config["qpe_initial_distribution"] = "local"
    with pytest.raises(ValueError, match="collision_free_pairs"):
        runner._validate_config(config)


@pytest.mark.skipif(
    not (SRG_DIRECTORY / "sr251256.g6").is_file(),
    reason=f"SRG catalogs not downloaded to {SRG_DIRECTORY} (see README, Data)",
)
def test_srg_photonic_experiment_counts_pairs_and_records_settings(tmp_path):
    config = {
        "srg_data_directory": str(SRG_DIRECTORY),
        "evolution_time": 1.0,
        "isomorphic_permutations": 5,
        "tolerance": 1e-9,
        "seed": 0,
        "photonic_methods": [
            {
                "name": "photon2_pnr",
                "n_photons": 2,
                "detection": "pnr",
                "particles": "bosons",
                "backend": "permanent",
            }
        ],
    }
    output = run_srg_photonic_experiment(config, tmp_path)
    family = output["families"]["srg(26,10,3,4)"]
    assert list(family) == ["two_particle_xy", "photon2_pnr"]
    assert family["two_particle_xy"]["distinguished_pairs"] == 45
    # Free photons without postselection cannot separate same-parameter SRGs.
    assert family["photon2_pnr"]["distinguished_pairs"] == 0
    assert family["photon2_pnr"]["controls_pass"]
    assert family["photon2_pnr"]["mean_postselection_success"] == 1.0
    assert (tmp_path / "srg_metrics.json").is_file()
