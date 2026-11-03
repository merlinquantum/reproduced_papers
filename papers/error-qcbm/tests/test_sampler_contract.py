"""Independent detector checks and process-local sampler isolation."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import perceval as pcvl
import pytest
import torch
from lib import training
from lib.backend import DEMO_INPUT_STATE, PhotonicBackend, initialize_parameters
from lib.data import collision_free_states, gaussian_target
from lib.sampling import count_accumulation
from lib.theory import theoretical_distribution
from merlin.measurement.process import SamplingProcess


def profile(transmittance=1.0, indistinguishability=1.0, detector="threshold"):
    return {
        "name": "contract",
        "transmittance": transmittance,
        "indistinguishability": indistinguishability,
        "detector": detector,
    }


@pytest.mark.parametrize("overlap", [0.0, 0.729, 0.84, 1.0])
def test_three_photon_partial_noise_matches_independent_gram_sum(overlap):
    backend = PhotonicBackend((1, 0, 1, 0, 1, 0), profile(0.63, overlap))
    backend.set_parameters(initialize_parameters(30, [11])[0])
    reference = theoretical_distribution(
        np.asarray(backend.circuit.compute_unitary(), dtype=complex),
        backend.input_state,
        0.63,
        overlap,
    )
    actual = backend.exact()
    assert sum(actual.values()) == pytest.approx(1.0, abs=1e-12)
    assert (
        max(
            abs(actual.get(s, 0) - reference.get(s, 0))
            for s in actual.keys() | reference.keys()
        )
        < 1e-11
    )


def test_twelve_mode_full_fock_precedes_threshold_collision_mapping():
    phases = initialize_parameters(132, [6])[0]
    pnr = PhotonicBackend(DEMO_INPUT_STATE, profile(detector="pnr"))
    threshold = PhotonicBackend(DEMO_INPUT_STATE, profile())
    pnr.set_parameters(phases)
    threshold.set_parameters(phases)
    assert sum(p.numel() for p in pnr.layer.parameters()) == 132
    full = pnr.exact()
    assert len(full) == 1365
    assert all(sum(state) == 4 for state in full)
    reference = pcvl.Processor("SLOS", pnr.circuit.copy())
    reference.with_input(pcvl.BasicState(DEMO_INPUT_STATE))
    probabilities = pcvl.algorithm.Sampler(reference).probs()["results"]
    perceval_distribution = {
        tuple(state): float(value) for state, value in probabilities.items()
    }
    assert (
        max(
            abs(full.get(s, 0) - perceval_distribution.get(s, 0))
            for s in full.keys() | perceval_distribution.keys()
        )
        < 1e-11
    )
    coarsened = {}
    for state, probability in full.items():
        key = tuple(int(n > 0) for n in state)
        coarsened[key] = coarsened.get(key, 0) + probability
    observed = threshold.exact()
    assert (
        max(
            abs(observed.get(s, 0) - coarsened.get(s, 0))
            for s in observed.keys() | coarsened.keys()
        )
        < 1e-11
    )
    assert sum(p for s, p in observed.items() if sum(s) == 3) > 0.01
    assert sum(p for s, p in observed.items() if sum(s) == 4) < 1


@pytest.mark.parametrize("overlap", [0.729, 0.84, 1.0])
def test_paper_scale_lossy_probabilities_match_independent_four_photon_sum(overlap):
    backend = PhotonicBackend(DEMO_INPUT_STATE, profile(0.028, overlap))
    backend.set_parameters(initialize_parameters(132, [6])[0])
    reference = theoretical_distribution(
        np.asarray(backend.circuit.compute_unitary(), dtype=complex),
        backend.input_state,
        0.028,
        overlap,
    )
    actual = backend.exact()
    assert (
        max(
            abs(actual.get(s, 0) - reference.get(s, 0))
            for s in actual.keys() | reference.keys()
        )
        < 1e-11
    )
    states = [s for s in reference if sum(s) == 4]
    expected = np.array([reference[s] for s in states])
    observed = np.array([actual.get(s, 0) for s in states])
    assert np.max(np.abs(expected / expected.sum() - observed / observed.sum())) < 1e-10


def test_project_sampling_serializes_rng_and_restores_after_failure(monkeypatch):
    a = PhotonicBackend((1, 0, 1, 0), profile(0.6, 0.84), native_chunk_shots=1000)
    b = PhotonicBackend(
        (1, 0, 1, 0),
        profile(0.6, 0.84),
        native_chunk_shots=1000,
        native_count_accumulation="vectorized",
    )
    expected = [a.sample(8001, 12), b.sample(8001, 13)]
    state = torch.random.get_rng_state().clone()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(a.sample, 8001, 12), pool.submit(b.sample, 8001, 13)]
        assert [f.result() for f in futures] == expected
    assert torch.equal(state, torch.random.get_rng_state())
    original = SamplingProcess.pcvl_sampler

    def fail(*args, **kwargs):
        raise RuntimeError("intentional forward failure")

    monkeypatch.setattr(b.layer, "forward", fail)
    with pytest.raises(RuntimeError, match="intentional forward"):
        b.sample(100, 99)
    assert SamplingProcess.pcvl_sampler is original
    assert torch.equal(state, torch.random.get_rng_state())


def test_foreign_sampler_patch_is_rejected(monkeypatch):
    monkeypatch.setattr(SamplingProcess, "pcvl_sampler", lambda *args: None)
    with pytest.raises(RuntimeError, match="already patched"):
        with count_accumulation("vectorized"):
            pass


def test_recycling_receives_one_aggregate_after_all_native_chunks(monkeypatch):
    backend = PhotonicBackend((1, 0, 1, 0), profile(0.7, 0.729), native_chunk_shots=300)
    states = collision_free_states(4, 2)
    target = gaussian_target(6, (1, 4), (1, 1))
    calls = []
    original = training.recycling_mitigation

    def observed(counts, *args, **kwargs):
        calls.append(sum(counts.values()))
        return original(counts, *args, **kwargs)

    monkeypatch.setattr(training, "recycling_mitigation", observed)
    objective = training.FiniteShotObjective(
        backend, states, target, "recycling", 1000, 42, metric="kl"
    )
    result = objective(np.zeros(12))
    assert np.isfinite(result)
    assert calls == [1000]
    assert backend.timings["native_chunks"] == 4
