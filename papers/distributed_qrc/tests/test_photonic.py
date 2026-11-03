"""MerLin photonic adaptation: neuron chips, reservoir wiring, photonic fidelity kernel.

Ground truth is computed independently with Perceval (circuit unitaries + SLOS).
"""

from __future__ import annotations

import itertools

import common  # noqa: F401
import numpy as np
import perceval as pcvl
import pytest
import torch
from common import load_runtime_ready_config
from distributed_qrc.lib.photonic_merlin import (
    PhotonicKernelMap,
    PhotonicReservoir,
    build_neuron_layer,
    evenly_spaced_input_state,
    fit_photonic_kernel_ridge,
    state_gram,
)
from distributed_qrc.lib.reservoirs import sample_quantum_reservoir


def _px(circuit):
    params = [
        p for p in circuit.get_parameters(all_params=True) if p.name.startswith("px")
    ]
    return sorted(params, key=lambda p: int(p.name[2:]))


def _unitary(circuit, values):
    for p, v in zip(_px(circuit), values):
        p.set_value(float(v))
    return np.array(circuit.compute_unitary(use_symbolic=False))


def _amplitude(unitary, state_in, state_out):
    backend = pcvl.BackendFactory.get_backend("SLOS")
    backend.set_circuit(pcvl.Unitary(pcvl.Matrix(unitary)))
    backend.set_input_state(pcvl.BasicState(state_in))
    return backend.prob_amplitude(pcvl.BasicState(state_out))


def test_neuron_chip_is_frozen_and_matches_postselected_perceval():
    scale = 1.0
    layer = build_neuron_layer(scale, np.random.SeedSequence(5))
    assert sum(p.numel() for p in layer.parameters() if p.requires_grad) == 0
    x = np.random.default_rng(1).uniform(-1, 1, (3, 4))
    with torch.no_grad():
        ours = layer(torch.from_numpy(x)).numpy()
    outputs = [o for o in itertools.product(range(3), repeat=4) if sum(o) == 2]
    for row, xi in zip(ours, x):
        unitary = _unitary(layer.circuit, scale * xi)
        probs = {
            o: abs(_amplitude(unitary, [1, 0, 1, 0], list(o))) ** 2 for o in outputs
        }
        # UNBUNCHED = threshold detection postselected on collision-free (2-click) events.
        unbunched = {o: p for o, p in probs.items() if max(o) <= 1}
        norm = sum(unbunched.values())
        expected = [
            sum(p * o[m] for o, p in unbunched.items()) / norm for m in range(4)
        ]
        assert np.allclose(row, expected, atol=1e-10)
        assert row.sum() == pytest.approx(2.0)


def test_photonic_reservoir_outputs_in_range_and_deterministic():
    spec = sample_quantum_reservoir(6, 4, np.random.default_rng(0))
    windows = np.random.default_rng(2).uniform(0, 1, (50, 4))
    a = PhotonicReservoir(spec, 1.0, 3, np.random.SeedSequence(9)).transform(windows)
    b = PhotonicReservoir(spec, 1.0, 3, np.random.SeedSequence(9)).transform(windows)
    assert a.shape == (50, 6)
    assert np.array_equal(a, b)
    assert a.min() >= -1.0 - 1e-12 and a.max() <= 1.0 + 1e-12


@pytest.mark.parametrize("n_features,n_modes", [(5, 5), (15, 10)])
def test_photonic_kernel_matches_perceval_fidelity(n_features, n_modes):
    scale = 2.0
    km = PhotonicKernelMap(n_features, n_modes, 2, scale, 2, np.random.SeedSequence(3))
    x = np.random.default_rng(0).uniform(-1, 1, (3, n_features))
    ours = state_gram(km.states(x)).numpy()
    unitaries = [_unitary(km.layer.circuit, scale * e) for e in km.encode(x)]
    for i, j in itertools.product(range(3), repeat=2):
        v = unitaries[j].conj().T @ unitaries[i]
        expected = abs(_amplitude(v, km.input_state, km.input_state)) ** 2
        assert ours[i, j] == pytest.approx(expected, abs=1e-10)


def test_photonic_kernel_gram_is_valid():
    km = PhotonicKernelMap(12, 10, 2, 2.0, 2, np.random.SeedSequence(4))
    states = km.states(np.random.default_rng(5).uniform(-1, 1, (40, 12)))
    assert torch.allclose(
        (states.abs() ** 2).sum(-1), torch.ones(40, dtype=torch.float64)
    )
    gram = state_gram(states).numpy()
    assert np.allclose(gram, gram.T) and np.allclose(np.diag(gram), 1.0)
    assert np.linalg.eigvalsh(gram).min() > -1e-10
    assert evenly_spaced_input_state(10, 2) == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]


def test_photonic_kernel_ridge_matches_direct_solution():
    rng = np.random.default_rng(6)
    features = {s: rng.uniform(-1, 1, (150, 5)) for s in ("train", "eval", "test")}
    targets = {s: 0.5 + 0.2 * np.sin(x.sum(1)) for s, x in features.items()}
    km = PhotonicKernelMap(5, 5, 2, 2.0, 2, np.random.SeedSequence(7))
    fitted = fit_photonic_kernel_ridge(features, targets, [1e-3], km, "cpu", None, None)
    train, test = km.states(features["train"]), km.states(features["test"])
    gram = state_gram(train).numpy()
    y_mean = targets["train"].mean()
    alpha = np.linalg.solve(gram + 150 * 1e-3 * np.eye(150), targets["train"] - y_mean)
    cross = (test.conj() @ train.T).abs().numpy() ** 2
    assert np.allclose(fitted.predictions["test"], cross @ alpha + y_mean, atol=1e-8)


def test_tiny_photonic_grid_runs(tmp_path):
    from distributed_qrc.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["experiment"].update(
        sizes=[4],
        reservoir_types=["photonic"],
        readout_types=["linear", "photonic_kernel"],
    )
    cfg["readout"].update(kernel_selection_subsample=150, kernel_fit_subsample=200)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paper_runner.train_and_evaluate(cfg, run_dir)
    import json

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert [c["readout_type"] for c in metrics["cells"]] == [
        "linear",
        "photonic_kernel",
    ]
    assert all(0 < c["metrics"]["test"]["rmse"] < 1 for c in metrics["cells"])
    assert metrics["cells"][1]["instances"][0]["n_photons"] == 2
