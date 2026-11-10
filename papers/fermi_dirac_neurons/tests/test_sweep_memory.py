"""Memory estimates and memory-budget scheduling of utils/run_sweep.py."""

from __future__ import annotations

import threading
import time

import common  # noqa: F401  (sets sys.path)
import pytest
from fermi_dirac_neurons.utils import run_sweep

# Peak resident memory measured on 2026-11-09 (single-threaded workers),
# after the state-array preallocation fix.
MEASURED_PEAK_GIB = {
    "e11_tanh_7q_code": 0.58,
    "e11_tanh_8q_code": 1.46,
    "e11_tanh_8q_haar": 1.46,
    "e11_tanh_9q_code": 6.58,
    "e10_tfim_9q_a05": 0.48,
    "e10_tfim_10q_a05": 0.74,
    "e10_tfim_11q_a05": 1.37,
    # Phase 4 photonic tasks, measured 2026-11-10 (single-threaded).
    "p4_learn_m8_n4_haar": 0.45,
    "p4_grad_m20_n3": 1.04,
    "p4_grad_m10_n5": 1.72,
    "p4_kerr_m12_n3": 0.41,
    "p4_kerr_m10_n5": 1.12,
    # Phase 5 B4 reference, measured 2026-11-10 (single-threaded).
    "p5_ref_fig8_logistic_7q": 1.07,
}


@pytest.mark.parametrize("config, measured", sorted(MEASURED_PEAK_GIB.items()))
def test_memory_estimate_never_underestimates_measurements(config, measured):
    estimate = run_sweep.estimate_memory_gib(run_sweep.resolved_config(config))
    assert measured <= estimate <= 1.5 * measured + 0.5


def _coordinator(configs, budget, attempts_used=None, extra=None):
    spec = {
        "sweep_id": "memory_test",
        "configs": configs,
        "seeds": [0, 1],
        "max_workers": 4,
        "plan": {},
    }
    if attempts_used is not None:
        spec["attempts_used"] = attempts_used
    if extra is not None:
        spec["authorized_extra_attempts"] = extra
    return run_sweep.Coordinator(
        spec, run_sweep.PROJECT_DIR / "configs" / "defaults.json", budget
    )


def test_planning_refuses_a_run_larger_than_the_budget():
    with pytest.raises(ValueError, match="memory budget"):
        _coordinator(["e11_tanh_9q_code"], budget=4.0)


def test_planning_refuses_runs_without_attempts_left():
    with pytest.raises(ValueError, match="explicit authorization"):
        _coordinator(
            ["e11_tanh_2q_code"], budget=10.0, attempts_used={"e11_tanh_2q_code__s0": 2}
        )


def test_scheduler_keeps_concurrent_estimates_within_budget(monkeypatch):
    coordinator = _coordinator(
        ["e11_tanh_9q_code", "e11_tanh_8q_code", "e11_tanh_2q_code"], budget=10.0
    )
    lock = threading.Lock()
    in_use = {"memory": 0.0, "peak": 0.0, "started": []}

    def fake_execute(run):
        with lock:
            in_use["memory"] += run["memory_estimate_gib"]
            in_use["peak"] = max(in_use["peak"], in_use["memory"])
            in_use["started"].append(run["config"])
        time.sleep(0.05)
        with lock:
            in_use["memory"] -= run["memory_estimate_gib"]

    monkeypatch.setattr(coordinator, "_execute", fake_execute)
    coordinator._schedule()
    assert len(in_use["started"]) == 6
    assert in_use["peak"] <= 10.0
    # The two 9-qubit runs (~7 GiB each) never overlap, but lighter runs fill the gap.
    assert in_use["started"][:2] == ["e11_tanh_9q_code", "e11_tanh_8q_code"]


def test_recorded_authorization_allows_exactly_one_more_attempt():
    grant = {
        "attempts": 1,
        "authorized_by": "user",
        "quote": "you can relaunch",
        "date": "2026-11-10",
    }
    coordinator = _coordinator(
        ["e11_tanh_2q_code"],
        budget=10.0,
        attempts_used={"e11_tanh_2q_code__s0": 2},
        extra={"e11_tanh_2q_code__s0": grant},
    )
    run = next(run for run in coordinator.runs if run["seed"] == 0)
    assert run["attempt_limit"] == 3
    assert run["attempts_used_before_this_sweep"] == 2


def test_authorization_without_a_quote_is_refused():
    with pytest.raises(ValueError, match="lacks"):
        _coordinator(
            ["e11_tanh_2q_code"],
            budget=10.0,
            extra={"e11_tanh_2q_code__s0": {"attempts": 1}},
        )
