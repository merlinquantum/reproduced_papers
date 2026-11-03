"""Data construction: one-batch loader check, generator rule, HeH+ Hamiltonian."""

from __future__ import annotations

import numpy as np
import pytest
from common import PROJECT_DIR, load_runtime_ready_config
from virtual_qnn.lib import data

from runtime_lib.config import deep_update, load_config


def _vqc_config():
    cfg = load_runtime_ready_config()
    overlay = load_config(PROJECT_DIR / "configs" / "e4_vqc_original.json")
    cfg = deep_update(cfg, overlay)
    cfg["data_root"] = str(PROJECT_DIR.parents[1] / "data")
    return cfg


def test_one_batch_shapes():
    cfg = _vqc_config()
    x, y = next(iter(data.build_dataloader(cfg, split="train")))
    assert x.shape == (200, 2), x.shape
    assert y.shape == (200,), y.shape
    x_test, _ = next(iter(data.build_dataloader(cfg, split="test")))
    assert x_test.shape == (100, 2)


def test_labels_follow_radius_rule_and_are_deterministic():
    first = data.generate_circle_dataset(200, 100, seed_data=3)
    second = data.generate_circle_dataset(200, 100, seed_data=3)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    radii = np.linalg.norm(first["train_x"], axis=1)
    np.testing.assert_array_equal(radii < data.CIRCLE_RADIUS, first["train_y"] == 0.0)


def test_unknown_split_fails():
    with pytest.raises(ValueError, match="Unknown split"):
        data.build_dataloader(_vqc_config(), split="validation")


def test_heh_hamiltonian_matches_reference_notebook():
    pytest.importorskip("pennylane")
    molecule = data.heh_tapered_hamiltonian(0.8)
    # Value printed by the authors' VVQE notebook at distance 0.8.
    assert molecule["fci_energy"] == pytest.approx(-2.86054203541046, abs=1e-10)
    assert molecule["n_qubits_tapered"] == 2
    assert molecule["hf_energy"] > molecule["fci_energy"]
    assert molecule["separation_angstrom"] == pytest.approx(1.6 * data.BOHR_TO_ANGSTROM)
