"""Phase 04 MerLin adaptation: hardware conditions, known answers and wiring."""

from __future__ import annotations

import json

import numpy as np
import perceval as pcvl
import pytest
from common import PROJECT_DIR, load_runtime_ready_config
from photonic_trainability.lib import clements, photonic_sim, runner, training
from photonic_trainability.lib.merlin_hardware import validate_condition
from photonic_trainability.lib.observables import evaluate_monomial, make_monomial


def _condition(name, noise=None, detector="pnr", postselect=False):
    return {
        "name": name,
        "noise": noise or {},
        "detector": detector,
        "postselect": postselect,
    }


def _mesh(modes, seed):
    rng = np.random.default_rng(seed)
    return clements.decompose(photonic_sim.haar_unitary(modes, rng))


def _expectation(mesh, occupation, condition, monomial=None):
    monomial = monomial or make_monomial({0: 1, 1: 1})
    return training.PhotonicExpectation(
        clements.perceval_circuit(mesh),
        occupation,
        lambda patterns: evaluate_monomial(monomial, patterns),
        hardware=condition,
    )


def test_conditions_require_every_key_and_known_values():
    with pytest.raises(KeyError):
        validate_condition({"name": "x", "noise": {}, "detector": "pnr"})
    with pytest.raises(ValueError):
        validate_condition(_condition("x", detector="apd"))
    with pytest.raises(ValueError):
        validate_condition(_condition("x", noise={"g2": 0.1}))


def test_noise_free_conditions_reproduce_the_ideal_model():
    mesh = _mesh(5, 0)
    occupation = (1, 0, 1, 0, 1)
    batch = mesh.parameters()[None, :]
    reference = _expectation(mesh, occupation, None).probabilities(batch)
    for condition in (
        _condition("ideal"),
        _condition("perfect_source", {"indistinguishability": 1.0}),
        _condition("no_loss_postselected", {"transmittance": 1.0}, postselect=True),
    ):
        expectation = _expectation(mesh, occupation, condition)
        np.testing.assert_allclose(
            expectation.probabilities(batch), reference, atol=1e-12
        )


def test_hong_ou_mandel_coincidence_follows_indistinguishability():
    circuit = pcvl.Circuit(2)
    circuit.add(0, pcvl.BS())
    circuit.add(0, pcvl.PS(pcvl.P(clements.parameter_name(0))))
    for visibility in (1.0, 0.9, 0.7):
        expectation = training.PhotonicExpectation(
            circuit,
            (1, 1),
            lambda patterns: evaluate_monomial(make_monomial({0: 1, 1: 1}), patterns),
            hardware=_condition("v", {"indistinguishability": visibility}),
        )
        # n0 n1 equals 1 exactly on the coincidence outcome |1,1>.
        coincidence = expectation.exact_losses(np.zeros((1, 1)))[0]
        assert np.isclose(coincidence, (1 - visibility) / 2, atol=1e-9)


def test_uniform_loss_scales_mean_detected_photons():
    mesh = _mesh(4, 1)
    occupation = (1, 0, 1, 0)
    expectation = training.PhotonicExpectation(
        clements.perceval_circuit(mesh),
        occupation,
        lambda patterns: patterns.sum(axis=1).astype(float),
        hardware=_condition("loss", {"transmittance": 0.7}),
    )
    assert set(expectation.patterns.sum(axis=1)) == {0, 1, 2}
    detected = expectation.exact_losses(mesh.parameters()[None, :])[0]
    assert np.isclose(detected, 0.7 * 2, atol=1e-9)


def test_postselection_keeps_only_full_photon_number():
    mesh = _mesh(4, 2)
    expectation = _expectation(
        mesh, (1, 0, 1, 0), _condition("ps", {"transmittance": 0.7}, postselect=True)
    )
    assert set(expectation.patterns.sum(axis=1)) == {2}
    probabilities = expectation.probabilities(mesh.parameters()[None, :])
    assert np.isclose(probabilities.sum(), 1.0)


def test_threshold_detection_aggregates_pnr_probabilities_into_clicks():
    mesh = _mesh(4, 3)
    occupation = (1, 0, 1, 0)
    batch = mesh.parameters()[None, :]
    pnr = _expectation(mesh, occupation, None)
    threshold = _expectation(mesh, occupation, _condition("th", detector="threshold"))
    assert set(np.unique(threshold.patterns)) <= {0, 1}
    pnr_probabilities = pnr.probabilities(batch)[0]
    aggregated = {}
    for pattern, probability in zip(pnr.patterns, pnr_probabilities):
        key = tuple(int(v > 0) for v in pattern)
        aggregated[key] = aggregated.get(key, 0.0) + probability
    for pattern, probability in zip(
        threshold.patterns, threshold.probabilities(batch)[0]
    ):
        assert np.isclose(
            probability, aggregated[tuple(int(v) for v in pattern)], atol=1e-12
        )


@pytest.mark.parametrize(
    "condition",
    [
        _condition("indist", {"indistinguishability": 0.8}),
        _condition("loss", {"transmittance": 0.7}),
        _condition("threshold", detector="threshold"),
    ],
)
def test_gpsr_with_photon_number_bound_is_exact_under_hardware_effects(condition):
    mesh = _mesh(4, 4)
    occupation = (1, 0, 1, 0)
    expectation = _expectation(mesh, occupation, condition, make_monomial({0: 2, 1: 1}))
    parameters = mesh.parameters()
    gpsr = training.gpsr_gradient(expectation.exact_losses, parameters, sum(occupation))
    np.testing.assert_allclose(
        gpsr, expectation.autograd_gradient(parameters), rtol=1e-6, atol=1e-9
    )


def test_merlin_variance_smoke_runs_and_ideal_matches_exact(tmp_path):
    cfg = load_runtime_ready_config()
    for key in ("cases", "monte_carlo"):
        cfg.pop(key, None)
    cfg.update(json.loads((PROJECT_DIR / "configs" / "merlin_smoke.json").read_text()))
    runner.train_and_evaluate(cfg, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["ideal_matches_exact"] is True
    assert {row["condition"] for row in metrics["rows"]} == {
        condition["name"] for condition in cfg["conditions"]
    }


def test_training_hardware_requires_a_condition(tmp_path):
    cfg = load_runtime_ready_config()
    cfg.update(
        json.loads(
            (
                PROJECT_DIR / "configs" / "merlin_training_hardware_ideal.json"
            ).read_text()
        )
    )
    cfg.update({"iterations": 1, "seeds": [0], "shots": 50})
    del cfg["hardware"]
    with pytest.raises(KeyError):
        runner.train_and_evaluate(cfg, tmp_path)
