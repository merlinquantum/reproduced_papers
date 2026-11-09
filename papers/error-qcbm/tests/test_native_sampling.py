"""Check the benchmark actually calls native MerLin and preserves counts."""

import json

import numpy as np
import torch
from lib.backend import PhotonicBackend
from lib.runner import train_and_evaluate

from runtime_lib.dtypes import resolve_config_dtypes


def test_native_chunk_counts_preserve_budget_rng_and_theory(monkeypatch):
    backend = PhotonicBackend(
        [1, 1],
        {"name": "chunk_check", "transmittance": 0.7, "indistinguishability": 0.729},
        native_chunk_shots=300,
    )
    original = backend._probabilities
    budgets = []

    def observed(shots=0):
        budgets.append(shots)
        return original(shots)

    monkeypatch.setattr(backend, "_probabilities", observed)
    rng = torch.random.get_rng_state().clone()
    counts = backend.sample(1000, 42)
    assert budgets == [300, 300, 300, 100]
    assert sum(counts.values()) == 1000
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert backend.sample(1000, 42) == counts
    assert backend.timings["native_chunks"] == 4
    theory = backend.exact()
    assert sum(abs(counts.get(k, 0) / 1000 - p) for k, p in theory.items()) / 2 < 0.1


def test_native_task_never_uses_numpy_random_sampling(tmp_path, monkeypatch):
    def reject_aggregate(*args, **kwargs):
        raise AssertionError("Aggregate sampling is forbidden in this task")

    monkeypatch.setattr(np.random, "default_rng", reject_aggregate)
    data = tmp_path / "data"
    hub = data / "error-qcbm/hub"
    hub.mkdir(parents=True)
    (hub / "fixture.json").write_text(
        json.dumps(
            {
                "platform": "qpu:belenos",
                "http_status": 200,
                "queried_at_utc": "2026-10-17T00:00:00+00:00",
                "calibration_timestamp": None,
                "processor": {
                    "perfs": {
                        "Transmittance (%)": 50,
                        "HOM (%)": 72.9,
                        "g2 (%)": 1.1,
                        "Clock (MHz)": 4.94,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "task": "native_sampling_check",
        "device": "cpu",
        "dtype": "float64",
        "data_root": str(data),
        "model": {"input_state": [1, 1]},
        "native_sampling": {
            "profiles": ["belenos_loss", "belenos_partial"],
            "initialization_seed": 0,
            "sampling_seed": 123,
            "torch_threads": 1,
            "snapshot_file": "fixture.json",
            "schedule": [{"shots": 100, "repeats": 1}, {"shots": 200, "repeats": 1}],
        },
    }
    output = tmp_path / "run"
    output.mkdir()
    train_and_evaluate(resolve_config_dtypes(cfg), output)
    report = json.loads((output / "native_sampling.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert not report["numpy_sampling_used"]
    assert not report["library_patched"]
    assert len(report["records"]) == 4
    assert all(
        row["observed_shots"] == row["initial_shots"] for row in report["records"]
    )
