"""Check committed optimizer state across finite-shot interruptions and reuse."""

import copy
import json

import numpy as np
import pytest
from lib.resumable import train_instance
from test_resumable import CASE, configuration


def assert_same_checkpoint(left, right):
    """Compare scientific state, allowing wall-clock and execution logs to differ."""
    with (
        np.load(left, allow_pickle=False) as a,
        np.load(right, allow_pickle=False) as b,
    ):
        for name in a.files:
            if name not in ("report_json", "sampling_seconds"):
                np.testing.assert_array_equal(a[name], b[name], err_msg=name)
        ra, rb = (json.loads(str(saved["report_json"])) for saved in (a, b))
        assert ra["history"] == rb["history"]
        assert ra["calibrated_gain"] == rb["calibrated_gain"]


@pytest.mark.parametrize("mode", ["python", "vectorized"])
def test_recycling_resume_preserves_all_optimizer_and_sampling_state(tmp_path, mode):
    cfg = configuration()
    cfg["training"]["native_count_accumulation"] = mode
    if mode == "vectorized":
        # This fixture tests continuation, not the independent full-budget gate.
        cfg["_vectorized_validation"] = {"sha256": "continuation-unit-fixture"}
    case = {**CASE, "estimator": "recycling"}
    full = train_instance(cfg, tmp_path / "full", 0, case)
    paused = train_instance(cfg, tmp_path / "split", 0, case, max_updates=3)
    assert paused["completed_steps"] == 3
    resumed = train_instance(cfg, tmp_path / "split", 0, case)
    assert full["history"] == resumed["history"]
    assert_same_checkpoint(
        tmp_path / "full/checkpoint.npz", tmp_path / "split/checkpoint.npz"
    )
    with (
        np.load(tmp_path / "full/final.npz") as a,
        np.load(tmp_path / "split/final.npz") as b,
    ):
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_recycling_implementation_change_is_rejected_before_sampling(
    tmp_path, monkeypatch
):
    cfg = configuration()
    case = {**CASE, "estimator": "recycling"}
    directory = tmp_path / "split"
    train_instance(cfg, directory, 0, case, max_updates=1)
    status_before = (directory / "run_status.json").read_bytes()
    ledger_before = (directory / "sampling_ledger.jsonl").read_bytes()
    monkeypatch.setitem(
        train_instance.__globals__,
        "recycling_provenance",
        lambda: {"implementation": "different_estimator", "perceval_version": "test"},
    )
    with pytest.raises(ValueError, match="recycling implementation"):
        train_instance(cfg, directory, 0, case)
    assert (directory / "run_status.json").read_bytes() == status_before
    assert (directory / "sampling_ledger.jsonl").read_bytes() == ledger_before


def test_torn_ledger_record_requires_explicit_recovery_before_sampling(tmp_path):
    cfg = configuration()
    directory = tmp_path / "state"
    train_instance(cfg, directory, 0, CASE, max_updates=1)
    ledger = directory / "sampling_ledger.jsonl"
    with ledger.open("ab") as stream:
        stream.write(b'{"status": "started", "call_index":')
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="Incomplete sampling ledger tail"):
        train_instance(cfg, directory, 0, CASE)
    assert ledger.read_bytes() == before


def test_invalid_saved_gain_is_rejected_without_overwriting_status(tmp_path):
    cfg = configuration()
    directory = tmp_path / "state"
    train_instance(cfg, directory, 0, CASE, max_updates=1)
    before = (directory / "run_status.json").read_bytes()
    with np.load(directory / "checkpoint.npz") as saved:
        values = {key: saved[key] for key in saved.files}
    values["gain"] = np.nan
    np.savez(directory / "checkpoint.npz", **values)
    with pytest.raises(ValueError, match="SPSA state is invalid"):
        train_instance(cfg, directory, 0, CASE)
    assert (directory / "run_status.json").read_bytes() == before


@pytest.mark.parametrize("interrupt_call", [2, 3, 4, 7])
def test_interrupt_during_calibration_gradient_or_record_replays_exactly(
    tmp_path, monkeypatch, interrupt_call
):
    cfg = configuration()
    case = {**CASE, "estimator": "recycling"}
    expected = train_instance(cfg, tmp_path / "full", 0, case)
    objective_type = train_instance.__globals__["AuditedObjective"]
    original = objective_type.__call__

    def interrupted(self, parameters):
        call = self.calls
        loss = original(self, parameters)
        if call == interrupt_call:
            raise KeyboardInterrupt("kill after sampling, before update commit")
        return loss

    monkeypatch.setattr(objective_type, "__call__", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train_instance(cfg, tmp_path / "split", 0, case)
    monkeypatch.setattr(objective_type, "__call__", original)
    resumed = train_instance(cfg, tmp_path / "split", 0, case)
    assert resumed["history"] == expected["history"]
    assert_same_checkpoint(
        tmp_path / "full/checkpoint.npz", tmp_path / "split/checkpoint.npz"
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "split/sampling_ledger.jsonl").read_text().splitlines()
    ]
    assert (
        sum(
            row["call_index"] == interrupt_call and row["status"] == "complete"
            for row in rows
        )
        == 2
    )
    replayed = {2: 3, 3: 1, 4: 2, 7: 3}[interrupt_call]
    assert resumed["shot_accounting"]["replayed_completed_calls"] == replayed
    assert (
        resumed["shot_accounting"]["known_fully_sampled_initial_shots"]
        == (16 + replayed) * 2000
    )
    assert resumed["total_initial_shots"] == 16 * 2000


def test_checkpoint_commit_precedes_progress_publication(tmp_path, monkeypatch):
    cfg = configuration()
    expected = train_instance(cfg, tmp_path / "full", 0, CASE)
    namespace = train_instance.__globals__
    original = namespace["_publish"]

    def interrupted(directory, report):
        if report["completed_steps"] == 1:
            raise KeyboardInterrupt("kill after checkpoint replace, before progress")
        return original(directory, report)

    monkeypatch.setitem(namespace, "_publish", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train_instance(cfg, tmp_path / "split", 0, CASE)
    monkeypatch.setitem(namespace, "_publish", original)
    resumed = train_instance(cfg, tmp_path / "split", 0, CASE)
    assert resumed["history"] == expected["history"]
    assert_same_checkpoint(
        tmp_path / "full/checkpoint.npz", tmp_path / "split/checkpoint.npz"
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "split/sampling_ledger.jsonl").read_text().splitlines()
    ]
    assert sum(row["status"] == "complete" for row in rows) == 16


@pytest.mark.parametrize("change", ["spsa", "chunk", "case", "target", "perturbation"])
def test_scientific_changes_rejected_before_any_sampling(tmp_path, change):
    cfg = configuration()
    directory = tmp_path / "state"
    train_instance(cfg, directory, 0, CASE, max_updates=1)
    before = (directory / "sampling_ledger.jsonl").read_bytes()
    new, case = copy.deepcopy(cfg), dict(CASE)
    if change == "spsa":
        new["training"]["spsa"]["c"] = 0.2
    elif change == "chunk":
        new["training"]["native_chunk_shots"] = 500
    elif change == "case":
        case["indistinguishability"] = 1.0
    elif change == "target":
        new["dataset"]["means"][0] += 1
    else:
        new["training"]["perturbation_seed_offset"] += 1
    with pytest.raises(ValueError, match="configuration differs"):
        train_instance(new, directory, 0, case)
    assert (directory / "sampling_ledger.jsonl").read_bytes() == before


def test_completed_skip_requires_a_matching_committed_checkpoint(tmp_path):
    cfg = configuration()
    directory = tmp_path / "state"
    train_instance(cfg, directory, 0, CASE)
    (directory / "checkpoint.npz").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        train_instance(cfg, directory, 0, CASE)


def test_corrupt_next_seed_is_rejected_without_numerical_failure_status(tmp_path):
    cfg = configuration()
    directory = tmp_path / "state"
    train_instance(cfg, directory, 0, CASE, max_updates=1)
    before = (directory / "run_status.json").read_bytes()
    with np.load(directory / "checkpoint.npz") as checkpoint:
        values = {key: checkpoint[key] for key in checkpoint.files}
    values["next_sampling_seed"] += 1
    np.savez(directory / "checkpoint.npz", **values)
    with pytest.raises(ValueError, match="sampling ledger"):
        train_instance(cfg, directory, 0, CASE)
    assert (directory / "run_status.json").read_bytes() == before


def test_io_interruption_preserves_checkpoint_and_can_be_explicitly_resumed(
    tmp_path, monkeypatch
):
    cfg = configuration()
    expected = train_instance(cfg, tmp_path / "full", 0, CASE)
    namespace = train_instance.__globals__
    original = namespace["_publish"]
    failed_once = False

    def locked_file(directory, report):
        nonlocal failed_once
        if report["completed_steps"] == 1 and not failed_once:
            failed_once = True
            raise PermissionError("simulated locked progress file")
        return original(directory, report)

    monkeypatch.setitem(namespace, "_publish", locked_file)
    with pytest.raises(PermissionError):
        train_instance(cfg, tmp_path / "split", 0, CASE)
    state = json.loads((tmp_path / "split/run_status.json").read_text())
    assert state["status"] == "interrupted"
    assert state["error_type"] == "PermissionError"
    events = (tmp_path / "split/execution_events.jsonl").read_bytes()
    monkeypatch.setitem(namespace, "_publish", original)
    resumed = train_instance(cfg, tmp_path / "split", 0, CASE)
    assert resumed["history"] == expected["history"]
    assert_same_checkpoint(
        tmp_path / "full/checkpoint.npz", tmp_path / "split/checkpoint.npz"
    )
    assert (tmp_path / "split/execution_events.jsonl").read_bytes() == events
