"""Verify finite-shot loss supports, failure budgets and replay accounting."""

import json

import numpy as np
import pytest
from lib.artifacts import planned_objective_calls, summarize_sampling_ledger
from lib.data import collision_free_states
from lib.training import FiniteShotObjective, conditional_distribution


class CountFixture:
    """Supply deterministic counts to isolate estimator failure semantics."""

    input_state = (1, 0, 1, 0)

    def __init__(self, counts):
        self.counts = counts
        self.calls = 0
        self.timings = {"sample_seconds": 0.0}

    def set_parameters(self, parameters):
        pass

    def sample(self, shots, seed):
        self.calls += 1
        return self.counts


def test_postselection_renormalizes_only_on_the_fixed_target_support():
    states = collision_free_states(4, 2)
    counts = {states[0]: 2, states[-1]: 6, (1, 0, 0, 0): 92}
    np.testing.assert_array_equal(
        conditional_distribution(counts, states), [0.25, 0, 0, 0, 0, 0.75]
    )


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_empty_recycling_sector_consumes_one_budget_and_never_resamples(missing):
    counts = {(1, 1, 0, 0): 10, (1, 0, 0, 0): 20, (0, 0, 0, 0): 30}
    del counts[list(counts)[missing]]
    backend = CountFixture(counts)
    states = collision_free_states(4, 2)
    objective = FiniteShotObjective(
        backend, states, np.full(6, 1 / 6), "recycling", sum(counts.values()), 42, "kl"
    )
    with pytest.raises(ValueError, match="nonempty zero/one/two"):
        objective(np.zeros(12))
    assert backend.calls == objective.calls == 1
    assert objective.last_diagnostics["sampling_seed"] == 42


@pytest.mark.parametrize(
    "steps,interval,total", [(1000, 10, 2103), (5, 2, 16), (3, 10, 10)]
)
def test_planned_budget_includes_initial_and_calibration(steps, interval, total):
    assert planned_objective_calls(steps, interval)["total"] == total


def test_ledger_separates_replays_failures_and_unknown_interrupted_sampling(tmp_path):
    path = tmp_path / "sampling_ledger.jsonl"
    records = []
    for index, status in [(0, "complete"), (0, "complete"), (1, "failed"), (2, None)]:
        row = {"call_index": index, "sampling_seed": 42 + index, "initial_shots": 2000}
        records.append({**row, "status": "started"})
        if status:
            records.append({**row, "status": status, "sampling_completed": True})
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    report = summarize_sampling_ledger(path)
    assert report["started_attempts"] == 4
    assert report["completed_attempts"] == 2
    assert report["failed_attempts"] == report["unfinished_attempts"] == 1
    assert report["known_fully_sampled_initial_shots"] == 6000
    assert report["unknown_completion_initial_shots_upper_bound"] == 2000
    assert report["replayed_completed_calls"] == 1
    assert report["replayed_completed_initial_shots"] == 2000


def test_legacy_failed_call_does_not_claim_a_known_shot_cost(tmp_path):
    path = tmp_path / "ledger.jsonl"
    row = {"call_index": 0, "sampling_seed": 42, "initial_shots": 50_000_000}
    path.write_text(
        json.dumps({**row, "status": "started"})
        + "\n"
        + json.dumps({**row, "status": "failed"})
        + "\n"
    )
    report = summarize_sampling_ledger(path)
    assert report["known_fully_sampled_initial_shots"] == 0
    assert report["unknown_completion_initial_shots_upper_bound"] == 50_000_000


def test_malformed_complete_ledger_record_fails_clearly(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("broken-json\n")
    with pytest.raises(ValueError, match="Malformed sampling ledger"):
        summarize_sampling_ledger(path)
