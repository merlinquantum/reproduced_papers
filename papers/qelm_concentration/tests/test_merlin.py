"""Tests of the Phase 4 MerLin photonic adaptation (`lib.qelm_merlin`, `lib.merlin_tasks`)."""

from __future__ import annotations

import copy
import itertools
import json
import math

import numpy as np
import pytest
import torch
from common import PROJECT_DIR, load_runtime_ready_config
from qelm_concentration.lib import runner as paper_runner
from qelm_concentration.lib.data import landsat_directory
from qelm_concentration.lib.qelm_merlin import (
    PhotonicArchitecture,
    PhotonicReservoir,
    click_readout_features,
    click_statistics,
    fit_click_eigentasks,
    neighbour_click_features,
    single_mode_frequencies,
)

from runtime_lib.config import deep_update, load_config
from runtime_lib.dtypes import resolve_config_dtypes

CONFIG_DIR = PROJECT_DIR / "configs"
TABLE_I_MAPPED = {
    "kind": "brickwall",
    "input_strength": 0.2,
    "b_mean": 0.707,
    "b_std": 0.031,
    "h_mean": 0.683,
    "h_std": 0.034,
    "h_min": 0.1,
}


def _arch(n_modes=6, n_photons=2, n_layers=5):
    return PhotonicArchitecture(
        n_modes=n_modes,
        n_photons=n_photons,
        n_layers=n_layers,
        encoding_period=4,
        input_shift=2,
        angle_scale=2.0,
    )


def _permanent(matrix: np.ndarray) -> complex:
    n = matrix.shape[0]
    return sum(
        np.prod([matrix[row, perm[row]] for row in range(n)])
        for perm in itertools.permutations(range(n))
    )


def _reference_pair_marginals(reservoir, inputs, pairs):
    """Threshold click marginals from permanents of the circuit unitary (no MerLin)."""
    arch = reservoir.arch
    circuit, _ = __import__(
        "qelm_concentration.lib.qelm_merlin", fromlist=["build_photonic_circuit"]
    ).build_photonic_circuit(arch, TABLE_I_MAPPED, 3, inputs.shape[1])
    occupied = [mode for mode, n in enumerate(arch.input_state) if n]
    angles = reservoir.encoding_angles(inputs).double().numpy()
    params = {p.name: p for p in circuit.get_parameters()}
    results = []
    for sample in angles:
        for index, value in enumerate(sample):
            params[f"x{index:04d}"].set_value(float(value))
        unitary = np.array(circuit.compute_unitary())
        clicks = {}
        for outputs in itertools.combinations_with_replacement(
            range(arch.n_modes), arch.n_photons
        ):
            sub = unitary[np.ix_(outputs, occupied)]
            norm = math.prod(
                math.factorial(outputs.count(mode)) for mode in set(outputs)
            )
            probability = abs(_permanent(sub)) ** 2 / norm
            pattern = tuple(sorted(set(outputs)))
            clicks[pattern] = clicks.get(pattern, 0.0) + probability
        marginals = np.zeros((len(pairs), 4))
        for pattern, probability in clicks.items():
            for k, (a, b) in enumerate(pairs):
                marginals[k, 2 * (a in pattern) + (b in pattern)] += probability
        results.append(marginals)
    return np.array(results)


def test_merlin_matches_permanent_reference():
    arch = _arch()
    reservoir = PhotonicReservoir(arch, TABLE_I_MAPPED, 3, 4, torch.float64)
    inputs = np.random.default_rng(0).uniform(-2, 2, size=(3, 4))
    pairs = arch.neighbour_pairs()
    merlin = reservoir.pair_click_probabilities(inputs, pairs, chunk_size=2)
    reference = _reference_pair_marginals(reservoir, inputs, pairs)
    np.testing.assert_allclose(merlin, reference, atol=1e-9)
    # Distinct inputs give distinct outputs, so every encoding parameter is wired.
    assert not np.allclose(merlin[0], merlin[1])


def test_threshold_detection_keeps_all_events():
    arch = _arch(n_modes=8, n_photons=4)
    reservoir = PhotonicReservoir(arch, TABLE_I_MAPPED, 0, 4, torch.float64)
    keys = [tuple(key) for key in reservoir.layer.output_keys]
    expected = sum(math.comb(8, k) for k in range(1, 5))
    assert len(keys) == expected and max(max(key) for key in keys) == 1
    with torch.no_grad():
        probs = reservoir.layer(reservoir.encoding_angles(np.zeros((2, 4)))).numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    # Bunched events show up as patterns with fewer clicks than photons.
    fewer_clicks = [i for i, key in enumerate(keys) if sum(key) < 4]
    assert probs[:, fewer_clicks].sum(axis=1).min() > 1e-3


def test_reservoir_is_seeded_and_frozen():
    arch = _arch(n_modes=8, n_photons=2)
    inputs = np.random.default_rng(1).uniform(-1, 1, size=(4, 4))
    pairs = arch.disjoint_pairs()

    def marginals(seed):
        return PhotonicReservoir(
            arch, TABLE_I_MAPPED, seed, 4, torch.float64
        ).pair_click_probabilities(inputs, pairs, 4)

    np.testing.assert_array_equal(marginals(0), marginals(0))
    assert not np.allclose(marginals(0), marginals(1))
    reservoir = PhotonicReservoir(arch, TABLE_I_MAPPED, 0, 4, torch.float64)
    assert sum(p.numel() for p in reservoir.layer.parameters() if p.requires_grad) == 0


def test_haar_setting_and_missing_keys():
    arch = _arch(n_modes=8, n_photons=2)
    haar = {**TABLE_I_MAPPED, "kind": "haar"}
    out = PhotonicReservoir(arch, haar, 0, 4, torch.float64).pair_click_probabilities(
        np.zeros((2, 4)), arch.disjoint_pairs(), 2
    )
    np.testing.assert_allclose(out.sum(axis=-1), 1.0)
    with pytest.raises(KeyError):
        PhotonicReservoir(arch, {"kind": "brickwall"}, 0, 4, torch.float64)
    with pytest.raises(KeyError):
        PhotonicArchitecture.from_config({"n_modes": 8})
    with pytest.raises(ValueError):
        _arch(n_modes=8, n_photons=3)


def test_click_readouts_shapes_and_eigentask_counts():
    rng = np.random.default_rng(2)
    n_pairs = 4
    train = rng.dirichlet(np.ones(4), size=(60, n_pairs))
    test = rng.dirichlet(np.ones(4), size=(20, n_pairs))
    singles = single_mode_frequencies(train)
    assert singles.shape == (60, 2 * n_pairs, 2)
    np.testing.assert_allclose(singles.sum(axis=-1), 1.0)
    assert click_statistics(train, 1).shape == (60, 2 * n_pairs)
    assert click_statistics(train, 2).shape == (60, 3 * n_pairs)
    basis = fit_click_eigentasks(train, 1000)
    assert basis.nsr.shape == (3 * n_pairs,)
    expected = {
        "w1_click_unit": 2 * n_pairs,
        "w1_et_scaled": 2 * n_pairs,
        "w2_click_scaled": 3 * n_pairs,
        "w2_et_unit": 3 * n_pairs,
    }
    for name, n_features in expected.items():
        train_x, test_x, info = click_readout_features(name, train, test, 1000, 1.0)
        assert train_x.shape == (60, n_features) and test_x.shape == (20, n_features)
        assert info["n_features"] == n_features
    with pytest.raises(ValueError):
        click_readout_features("w2_pauli_unit", train, test, 1000, 1.0)


def test_neighbour_features_layout():
    arch = _arch(n_modes=6, n_photons=2)
    reservoir = PhotonicReservoir(arch, TABLE_I_MAPPED, 0, 3, torch.float64)
    pair_probs = reservoir.pair_click_probabilities(
        np.zeros((2, 3)), arch.neighbour_pairs(), 2
    )
    features = neighbour_click_features(pair_probs)
    assert features.shape == (2, 2 * 6 - 1)
    singles = reservoir.pair_click_probabilities(
        np.zeros((2, 3)), arch.disjoint_pairs(), 2
    )
    np.testing.assert_allclose(features[:, :6], click_statistics(singles, 1), atol=1e-9)


def _config(*overlays):
    cfg = copy.deepcopy(load_runtime_ready_config())
    for overlay in overlays:
        data = (
            load_config(CONFIG_DIR / overlay) if isinstance(overlay, str) else overlay
        )
        cfg = deep_update(cfg, copy.deepcopy(data))
    return resolve_config_dtypes(cfg)


def _run(cfg, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    paper_runner.train_and_evaluate(cfg, run_dir)
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED", status["error"]
    log = (run_dir / "run.log").read_text()
    for event in ("RUN_STARTED", "DATASET_READY", "METRICS_WRITTEN", "RUN_COMPLETED"):
        assert log.count(f"{event} |") == 1, event
    return json.loads((run_dir / "metrics.json").read_text())


def test_landsat_merlin_smoke(tmp_path):
    cfg = _config("smoke_merlin.json", {"dataset": {"landsat": {"n_subsample": 120}}})
    if not (landsat_directory(cfg) / "sat.trn").exists():
        pytest.skip("Landsat data not downloaded (see README Data section)")
    metrics = _run(cfg, tmp_path)
    assert metrics["hardware"]["computation_space"] == "FOCK"
    assert metrics["hardware"]["n_modes"] == cfg["merlin_reservoir"]["n_modes"]
    summary = metrics["summary"]["readouts"]
    assert summary["readout=w2_click_scaled"]["n_features_mean"] == 3 * 4


def test_narma_and_concentration_merlin_smoke(tmp_path):
    narma = _config(
        "narma_scaling_merlin_m16.json",
        {
            "merlin_reservoir": {"n_modes": 8, "n_photons": 2, "n_layers": 4},
            "dataset": {
                "narma": {"orders": [1, 2], "t_init": 20, "t_train": 40, "t_test": 20}
            },
            "narma_scaling": {
                "sequence_seeds": [0],
                "reservoir_seeds": [0],
                "baselines": ["raw_window"],
            },
        },
    )
    metrics = _run(narma, tmp_path / "narma")
    assert metrics["window"] == 4 and metrics["n_modes"] == 8
    assert "qelm|shots_1e5" in metrics["summary"]

    concentration = _config(
        "concentration_merlin.json",
        {
            "concentration_merlin": {
                "sizes": [[8, 2]],
                "n_layers": [4],
                "reservoir_seeds": [0],
                "n_samples": 16,
            }
        },
    )
    metrics = _run(concentration, tmp_path / "concentration")
    assert len(metrics["records"]) == 3
