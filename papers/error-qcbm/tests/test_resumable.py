"""Verify deterministic native-shot resumption, failures and completed-run reuse."""

import copy
import json

import numpy as np
import pytest
from lib.resumable import train_instance
from test_spsa import SETTINGS


def configuration():
    """Return a small native MerLin experiment with paired random streams."""
    return {
        "model": {"input_state": [1, 0, 1, 0]},
        "device": "cpu",
        "dtype": "float64",
        "dataset": {"name": "gaussian", "means": [1, 4], "standard_deviations": [1, 1]},
        "training": {
            "steps": 5,
            "record_every": 2,
            "shots": 2000,
            "loss": "kl",
            "sampling_seed_offset": 10000,
            "perturbation_seed_offset": 100000,
            "spsa": SETTINGS,
            "segment_seconds": 3600,
            "torch_threads": 1,
            "native_chunk_shots": 1000,
        },
    }


CASE = {
    "name": "small_native",
    "transmittance": 0.7,
    "indistinguishability": 0.729,
    "estimator": "postselection",
}


def test_native_resume_matches_continuous_even_after_unrecorded_update(tmp_path):
    cfg = configuration()
    full = train_instance(cfg, tmp_path / "full", 0, CASE)
    paused = train_instance(cfg, tmp_path / "resume", 0, CASE, max_updates=3)
    assert paused["status"] == "paused"
    assert paused["completed_steps"] == 3
    assert [r["step"] for r in paused["history"]] == [0, 2]
    restored = train_instance(cfg, tmp_path / "resume", 0, CASE)
    assert restored["status"] == full["status"] == "complete"
    assert restored["history"] == full["history"]
    assert restored["objective_calls"] == full["objective_calls"] == 16
    for filename in ("final.npz", "checkpoint.npz"):
        with (
            np.load(tmp_path / "full" / filename) as a,
            np.load(tmp_path / "resume" / filename) as b,
        ):
            for key in ("parameters",):
                np.testing.assert_array_equal(a[key], b[key])
    ledger = tmp_path / "resume/sampling_ledger.jsonl"
    before = ledger.read_bytes()
    assert train_instance(cfg, tmp_path / "resume", 0, CASE)["status"] == "complete"
    assert ledger.read_bytes() == before
    changed = copy.deepcopy(cfg)
    changed["training"]["shots"] += 1
    with pytest.raises(ValueError, match="configuration differs"):
        train_instance(changed, tmp_path / "resume", 0, CASE)


def test_crash_after_sampling_replays_update_from_committed_rng(tmp_path, monkeypatch):
    cfg = configuration()
    expected = train_instance(cfg, tmp_path / "full", 0, CASE)
    # Runtime discovery unloads lib.*; patch the function's actual namespace.
    namespace = train_instance.__globals__
    original = namespace["_save_checkpoint"]

    def crash_before_commit(directory, optimizer, *args):
        if optimizer.step_number == 1:
            raise KeyboardInterrupt("simulated scheduler kill")
        return original(directory, optimizer, *args)

    monkeypatch.setitem(namespace, "_save_checkpoint", crash_before_commit)
    with pytest.raises(KeyboardInterrupt):
        train_instance(cfg, tmp_path / "crashed", 0, CASE)
    monkeypatch.setitem(namespace, "_save_checkpoint", original)
    resumed = train_instance(cfg, tmp_path / "crashed", 0, CASE)
    assert resumed["history"] == expected["history"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "crashed/sampling_ledger.jsonl")
        .read_text()
        .splitlines()
    ]
    assert (
        sum(row["call_index"] == 3 and row["status"] == "complete" for row in rows) == 2
    )


def test_numeric_failure_is_not_silently_retried(tmp_path):
    cfg = configuration()
    cfg["training"]["shots"] = 1
    case = {**CASE, "transmittance": 0.001}
    with pytest.raises(ValueError, match="Empty collision-free"):
        train_instance(cfg, tmp_path / "failed", 0, case)
    with pytest.raises(RuntimeError, match="automatic numerical retry disabled"):
        train_instance(cfg, tmp_path / "failed", 0, case)


def test_shared_runtime_routes_resumable_with_bundled_public_profile(tmp_path):
    from lib.runner import train_and_evaluate

    from runtime_lib.dtypes import resolve_config_dtypes

    cfg = configuration()
    cfg.update(
        task="train",
        data_root=str(tmp_path / "data"),
        noise_profile="configs/noise/belenos_20261017.json",
    )
    cfg["training"].update(
        resumable=True,
        seeds=[0],
        state_dir=str(tmp_path / "state"),
        conditions=[
            {
                "name": "lossless_ideal",
                "transmittance": 1.0,
                "indistinguishability": 1.0,
                "estimator": "postselection",
            }
        ],
    )
    output = tmp_path / "run"
    output.mkdir()
    train_and_evaluate(resolve_config_dtypes(cfg), output)
    campaign = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
    assert campaign["status"] == "complete"
    assert campaign["qpu_jobs_submitted"] == 0
    assert campaign["noise_provenance"]["platform"] == "qpu:belenos"
    assert campaign["noise_provenance"]["schema"] == "error_qcbm.noise_profile.v1"
    assert not (tmp_path / "data").exists()
    assert campaign["instances"][0]["native_chunk_shots"] == 1000
