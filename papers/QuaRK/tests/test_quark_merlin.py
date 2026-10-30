"""Photonic MerLin adaptation: circuit semantics, reset mixture, features and wiring."""

from __future__ import annotations

import json
import os
import warnings
from itertools import permutations
from math import factorial
from pathlib import Path

import merlin as ml
import numpy as np
import pytest
import torch
from common import PROJECT_DIR, load_runtime_ready_config
from QuaRK.lib import quark_merlin as qm
from QuaRK.lib.reservoir import (
    ANGLE_SCALE,
    batched_ry_layer,
    draw_reservoir_parameters,
    ising_unitary,
    reservoir_states,
)

import runtime_lib.runtime as runtime_module
from runtime_lib import run_from_project

TINY_MERLIN = {
    "num_modes": 4,
    "num_photons": 2,
    "num_reservoirs": 2,
    "locality": 2,
    "estimator": "exact",
    "shots": 50,
    "distinguishable": False,
}


def _permanent(matrix: np.ndarray) -> complex:
    size = matrix.shape[0]
    return sum(
        np.prod([matrix[i, p[i]] for i in range(size)])
        for p in permutations(range(size))
    )


def _reference_click_probabilities(layer, phases, input_state, distinguishable):
    """Click-pattern probabilities from the layer's Perceval circuit and permanents."""
    circuit = layer.circuit
    params = {p.name: p for p in circuit.get_parameters()}
    num_modes = len(input_state)
    # MerLin names each phase "<step prefix><running input index, from 1>".
    for index, value in enumerate(phases):
        params[f"t{index // num_modes:02d}m{index + 1}"].set_value(float(value))
    unitary = np.asarray(circuit.compute_unitary())
    columns = [m for m, count in enumerate(input_state) for _ in range(count)]
    photons = len(columns)
    probabilities: dict[tuple, float] = {}
    # Every output occupation with `photons` photons, bunched ones included.
    for rows in np.ndindex(*(num_modes,) * photons):
        if list(rows) != sorted(rows):
            continue
        occupation = np.bincount(rows, minlength=num_modes)
        sub = unitary[np.ix_(list(rows), columns)]
        if distinguishable:
            value = _permanent(np.abs(sub) ** 2).real
        else:
            value = abs(_permanent(sub)) ** 2
        value /= np.prod([factorial(int(c)) for c in occupation])
        pattern = tuple(int(c > 0) for c in occupation)
        probabilities[pattern] = probabilities.get(pattern, 0.0) + value
    return probabilities


def _layer_probabilities(layer, phases):
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = layer(torch.tensor(phases[None, :], dtype=torch.float64))[0].numpy()
    return dict(zip([tuple(k) for k in layer.output_keys], probs, strict=True))


@pytest.mark.parametrize("distinguishable", [False, True])
def test_truncated_layer_matches_permanent_reference(distinguishable):
    """Input columns, the fixed meshes and threshold detection against permanents."""
    input_state = qm.spread_input_state(4, 2)
    layer = qm.truncated_layer(
        2,
        num_modes=4,
        mesh_seed=11,
        input_state=input_state,
        distinguishable=distinguishable,
    )
    phases = np.random.default_rng(0).uniform(-np.pi, np.pi, size=8)
    ours = _layer_probabilities(layer, phases)
    reference = _reference_click_probabilities(
        layer, phases, input_state, distinguishable
    )
    assert set(ours) == set(reference)
    for pattern, value in reference.items():
        assert ours[pattern] == pytest.approx(value, abs=1e-10)


def test_interference_changes_the_distribution():
    """The distinguishable-photon control differs from the interfering reservoir."""
    phases = np.random.default_rng(1).uniform(-np.pi, np.pi, size=8)
    probs = [
        _layer_probabilities(
            qm.truncated_layer(
                2,
                num_modes=4,
                mesh_seed=11,
                input_state=[1, 0, 1, 0],
                distinguishable=flag,
            ),
            phases,
        )
        for flag in (False, True)
    ]
    assert max(abs(probs[0][k] - probs[1][k]) for k in probs[0]) > 1e-3


def test_threshold_detection_keeps_bunched_events():
    layer = qm.truncated_layer(
        1, num_modes=4, mesh_seed=5, input_state=[1, 0, 1, 0], distinguishable=False
    )
    probs = _layer_probabilities(layer, np.linspace(-1.0, 1.0, 4))
    assert all(set(key) <= {0, 1} for key in probs)  # click / no click
    assert any(sum(key) == 1 for key in probs)  # two photons in one detector
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-12)  # no postselection


def test_unbunched_space_postselects():
    """Why FOCK: MerLin's UNBUNCHED space renormalizes the collision-free outcomes."""

    def probabilities(space):
        builder = ml.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=False, seed=5)
        layer = ml.QuantumLayer(
            input_size=None,
            builder=builder,
            input_state=[1, 0, 1, 0],
            n_photons=2,
            dtype=torch.float64,
            measurement_strategy=ml.MeasurementStrategy.probs(computation_space=space),
        )
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs = layer().numpy()[0]
        keys = [tuple(int(c) for c in key) for key in layer.output_keys]
        return dict(zip(keys, probs, strict=True))

    unbunched = probabilities(ml.ComputationSpace.UNBUNCHED)
    fock = probabilities(ml.ComputationSpace.FOCK)
    collision_free = {k: v for k, v in fock.items() if max(k) <= 1}
    kept = sum(collision_free.values())
    assert kept < 1 - 1e-3  # bunched outcomes exist and are dropped
    assert set(unbunched) == set(collision_free)
    for key, value in collision_free.items():
        assert unbunched[key] == pytest.approx(value / kept, abs=1e-12)


def test_reset_channel_is_a_mixture_over_memory_lengths():
    """QuaRK's sequential reset channel equals the mixture used by the adaptation."""
    windows = np.tanh(np.random.default_rng(0).normal(size=(4, 5, 3)))
    params = draw_reservoir_parameters(
        input_dim=3, num_qubits=2, num_reservoirs=2, first_contraction=0.1, seed=3
    )
    states = reservoir_states(windows, params)
    angles = ANGLE_SCALE * np.tanh(windows @ params.projection)
    window, dim = windows.shape[1], 4
    initial = np.full((dim, dim), 1.0 / dim, dtype=complex)
    for r in range(2):
        unitary = ising_unitary(
            2, params.zz_angles[r], params.x_angles[r], params.z_angles[r]
        )
        weights = qm.mixture_weights(params.contractions[r], window)
        assert weights.sum() == pytest.approx(1.0)
        for i in range(windows.shape[0]):
            mixture = np.zeros((dim, dim), dtype=complex)
            for depth, weight in enumerate(weights):
                rho = initial.copy()
                for t in range(window - depth, window):
                    step = unitary @ batched_ry_layer(angles[i : i + 1, t])[0]
                    rho = step @ rho @ step.conj().T
                mixture += weight * rho
            np.testing.assert_allclose(mixture, states[i, r], atol=1e-12)


def test_spread_input_state_and_marginal_matrix():
    assert qm.spread_input_state(14, 3) == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        qm.spread_input_state(4, 1)
    keys = [(0, 0, 1), (0, 1, 1), (1, 1, 0)]
    matrix = qm.marginal_matrix(keys, 3)
    # columns: modes 0, 1, 2, then pairs (0, 1), (0, 2), (1, 2)
    expected = [[0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0]]
    np.testing.assert_array_equal(matrix, expected)


def test_photonic_features_shapes_and_seeding():
    windows = np.tanh(np.random.default_rng(2).normal(size=(12, 3, 3)))
    quark = load_runtime_ready_config()["model"]["quark"]
    exact, info = qm.photonic_features(windows, TINY_MERLIN, quark, seed=7)
    assert exact.shape == (12, 2 * (4 + 6))
    assert info["num_observables"] == 10
    assert np.all((exact >= -1e-12) & (exact <= 1 + 1e-12))
    expected_clicks = exact[:, :4].sum(axis=1)  # mean number of clicks, 1 to 2
    assert np.all((expected_clicks > 1 - 1e-9) & (expected_clicks < 2 + 1e-9))
    assert info["contractions"] == pytest.approx(
        draw_reservoir_parameters(
            input_dim=3,
            num_qubits=int(quark["num_qubits"]),
            num_reservoirs=2,
            first_contraction=float(quark["first_contraction"]),
            seed=7,
        ).contractions
    )

    sampled = {**TINY_MERLIN, "estimator": "click_sampling"}
    first, _ = qm.photonic_features(windows, sampled, quark, seed=7)
    again, _ = qm.photonic_features(windows, sampled, quark, seed=7)
    other, _ = qm.photonic_features(windows, sampled, quark, seed=8)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    many, _ = qm.photonic_features(
        windows, {**sampled, "shots": 1_000_000}, quark, seed=7
    )
    np.testing.assert_allclose(many, exact, atol=5e-3)


def test_runtime_applies_merlin_shots_flag(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}

    def fake_import_callable(name: str):
        def _runner(cfg, run_dir: Path):
            recorded["cfg"] = cfg

        return _runner

    monkeypatch.setattr(runtime_module, "import_callable", fake_import_callable)
    original_cwd = Path.cwd()
    try:
        run_from_project(
            PROJECT_DIR,
            [
                "--config",
                "configs/quark_merlin.json",
                "--merlin-shots",
                "100",
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)
    cfg = recorded["cfg"]
    assert cfg["model"]["kind"] == "quark_merlin"
    assert cfg["model"]["merlin"]["shots"] == 100
    assert cfg["model"]["merlin"]["num_photons"] == 3


def test_runner_completes_with_the_photonic_model(tmp_path):
    from QuaRK.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["data"]["num_windows"] = 120
    cfg["data"]["window"] = 4
    cfg["data"]["tasks"] = ["exp_fading"]
    cfg["readout"]["tuning"]["xi_maxiter"] = 5
    cfg["readout"]["tuning"]["nu_grid"] = [2.5]
    cfg["readout"]["train_sizes"] = [48, 96]
    cfg["readout"]["slope_min_size"] = 48
    cfg["model"]["kind"] = "quark_merlin"
    cfg["model"]["merlin"] = dict(TINY_MERLIN, estimator="click_sampling")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, default=str))

    paper_runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    task = metrics["tasks"]["exp_fading"]
    assert task["feature_dim"] == 20
    assert task["model_info"]["input_state"] == [1, 0, 1, 0]
    assert "bound" not in task["selections"]["validation"]
