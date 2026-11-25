"""Classical baselines: feature maps, wiring of their settings and the matched size."""

from __future__ import annotations

import json

import numpy as np
from common import PROJECT_DIR, load_runtime_ready_config
from QuaRK.lib import runner as paper_runner
from QuaRK.lib.classical import esn_features, raw_window_features
from QuaRK.lib.data import generate_dataset
from QuaRK.lib.measurement import k_local_paulis

ESN_SETTINGS = ("spectral_radius", "leak", "input_scale")


def _overlay(name: str) -> dict:
    return json.loads((PROJECT_DIR / "configs" / f"{name}.json").read_text())


def test_raw_window_features_flatten_each_window_in_time_order():
    windows = np.random.default_rng(0).normal(size=(4, 5, 3))
    features = raw_window_features(windows)
    assert features.shape == (4, 15)
    np.testing.assert_array_equal(features[2, 3:6], windows[2, 1])


def test_esn_features_follow_the_leaky_update():
    """Spectral radius, leak and input scale reach the recursion, started at h = 0."""
    windows = np.tanh(np.random.default_rng(1).normal(size=(6, 7, 3)))
    features = esn_features(
        windows, units=12, spectral_radius=0.7, leak=0.3, input_scale=0.5, seed=4
    )
    rng = np.random.default_rng(4)
    input_weights = rng.uniform(-0.5, 0.5, size=(3, 12))
    recurrent = rng.normal(size=(12, 12))
    recurrent *= 0.7 / np.max(np.abs(np.linalg.eigvals(recurrent)))
    state = np.zeros((6, 12))
    for t in range(7):
        update = np.tanh(windows[:, t] @ input_weights + state @ recurrent)
        state = 0.7 * state + 0.3 * update
    assert features.shape == (6, 12)
    np.testing.assert_allclose(features, state, atol=1e-12)


def test_esn_features_are_seeded_and_fade_old_inputs():
    windows = np.tanh(np.random.default_rng(2).normal(size=(50, 25, 3)))
    params = {"units": 315, "spectral_radius": 0.9, "leak": 1.0, "input_scale": 1.0}
    first = esn_features(windows, seed=7, **params)
    np.testing.assert_array_equal(first, esn_features(windows, seed=7, **params))
    assert not np.array_equal(first, esn_features(windows, seed=8, **params))
    # Echo-state memory, as QuaRK's contraction: flipping the first input of each
    # window moves the final state far less than flipping the last one.
    early, late = windows.copy(), windows.copy()
    early[:, 0] = -early[:, 0]
    late[:, -1] = -late[:, -1]
    drift_early = np.abs(esn_features(early, seed=7, **params) - first).mean()
    drift_late = np.abs(esn_features(late, seed=7, **params) - first).mean()
    assert drift_early < 0.1 * drift_late


def test_baselines_match_the_quantum_feature_count():
    """The ESN has as many units as QuaRK and its photonic version have features."""
    cfg = load_runtime_ready_config()
    quark = cfg["model"]["quark"]
    paulis = k_local_paulis(quark["locality"], quark["num_qubits"])
    quark_dim = quark["num_reservoirs"] * len(paulis)
    merlin = _overlay("quark_merlin")["model"]["merlin"]
    modes = merlin["num_modes"]
    photonic_dim = merlin["num_reservoirs"] * (modes + modes * (modes - 1) // 2)
    assert quark_dim == photonic_dim == cfg["model"]["esn"]["units"] == 315
    for name in ("esn_default_classical", "esn_tuned_classical"):
        assert "units" not in _overlay(name)["model"]["esn"]  # keeps the matched size


def test_esn_settings_reach_the_features():
    """The default ESN uses its configured settings; the tuned one picks from the grid."""
    cfg = load_runtime_ready_config()
    cfg["seed"] = 3
    cfg["data"]["num_windows"] = 120
    cfg["model"]["kind"] = "esn_kernel"
    esn = cfg["model"]["esn"]
    dataset = generate_dataset(cfg["data"], seed=3)

    esn["mode"] = "default"
    features, info = paper_runner.build_features(cfg, dataset, task_index=1)
    settings = {k: float(esn["default"][k]) for k in ESN_SETTINGS}
    expected = esn_features(
        dataset.windows, units=int(esn["units"]), seed=3, **settings
    )
    np.testing.assert_array_equal(features, expected)
    assert info["esn"] == settings

    esn["mode"] = "tuned"
    features, info = paper_runner.build_features(cfg, dataset, task_index=1)
    grid = {
        (float(r), float(leak), float(s))
        for r in esn["spectral_radius"]
        for leak in esn["leak"]
        for s in esn["input_scale"]
    }
    assert len(grid) == 18
    assert tuple(info["esn"][k] for k in ESN_SETTINGS) in grid
    assert features.shape == (120, 315)
