"""Verify exact native draws and checkpoint continuity through count migration."""

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from lib.backend import PhotonicBackend
from lib.resumable import train_instance, verify_accumulation_evidence
from lib.sampling import count_accumulation, vectorized_multinomial
from merlin.measurement.process import SamplingProcess
from test_resumable import CASE, configuration


@pytest.mark.parametrize("batched", [False, True])
def test_vector_counts_keep_native_draws_and_rng(batched):
    probabilities = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    if batched:
        probabilities = torch.stack([probabilities, probabilities.flip(0)])
    original = SamplingProcess.pcvl_sampler
    source = inspect.getsourcefile(SamplingProcess)
    with open(source, "rb") as stream:
        source_hash = hashlib.sha256(stream.read()).hexdigest()
    torch.manual_seed(42)
    expected = SamplingProcess().pcvl_sampler(probabilities, 2345)
    expected_state = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    actual = vectorized_multinomial(SamplingProcess(), probabilities, 2345)
    assert torch.equal(expected, actual)
    assert torch.equal(torch.random.get_rng_state(), expected_state)
    with pytest.raises(RuntimeError), count_accumulation("vectorized"):
        raise RuntimeError("forced forward failure")
    assert SamplingProcess.pcvl_sampler is original
    with open(source, "rb") as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == source_hash


def test_native_chunked_backend_counts_match_all_states():
    profile = {"name": "partial", "transmittance": 0.4, "indistinguishability": 0.729}
    backend = PhotonicBackend([1, 0, 1, 0], profile, native_chunk_shots=1000)
    expected = backend.sample(4567, 123)
    backend.native_count_accumulation = "vectorized"
    assert backend.sample(4567, 123) == expected
    assert backend.timings["native_chunks"] == 5
    assert (
        backend.timings["sample_engine"]
        == "merlin_native_multinomial_vectorized_counts"
    )


def test_resume_python_checkpoint_with_vector_counts_keeps_trajectory(tmp_path):
    cfg = configuration()
    full = train_instance(cfg, tmp_path / "full", 0, CASE)
    paused = train_instance(cfg, tmp_path / "migrated", 0, CASE, max_updates=3)
    assert paused["completed_steps"] == 3
    new = copy.deepcopy(cfg)
    new["training"]["native_count_accumulation"] = "vectorized"
    # Production wrapper validates the real report; this unit fixture tests the transition.
    new["_vectorized_validation"] = {"sha256": "unit-transition-fixture"}
    resumed = train_instance(new, tmp_path / "migrated", 0, CASE)
    assert resumed["history"] == full["history"]
    assert resumed["objective_calls"] == full["objective_calls"]
    transition = resumed["sampler_execution_history"][-1]
    assert transition["mode"] == "vectorized"
    assert transition["at_step"] == 3
    assert transition["verification"] == new["_vectorized_validation"]
    with (
        np.load(tmp_path / "full/final.npz") as a,
        np.load(tmp_path / "migrated/final.npz") as b,
    ):
        for name in ("parameters", "sampled_probabilities", "final_counts"):
            np.testing.assert_array_equal(a[name], b[name])


def test_production_evidence_rejects_any_changed_counts(tmp_path):
    names = (
        "belenos_postselection",
        "belenos_recycling",
        "lossless_partial",
        "lossless_ideal",
    )
    adapter = Path(__file__).resolve().parents[1] / "lib/sampling.py"
    sampler = Path(inspect.getsourcefile(SamplingProcess))
    row = {
        "status": "passed",
        "shots": 50000000,
        "native_chunk_shots": 1000000,
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "original_sampler_sha256": hashlib.sha256(sampler.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "comparisons": [
            {
                "label": label,
                "original_shots": 50000000,
                "vectorized_shots": 50000000,
                "counts_identical": True,
                "max_count_difference": 0,
                "probabilities_identical": True,
                "loss_identical": True,
            }
            for label in ("initial_archived_reference", "trained_fresh_reference")
        ],
    }
    report = {"status": "passed", "cases": {name: copy.deepcopy(row) for name in names}}
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert (
        verify_accumulation_evidence(path)["sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    report["cases"][names[0]]["comparisons"][0]["max_count_difference"] = 1
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="exact comparison failed"):
        verify_accumulation_evidence(path)
