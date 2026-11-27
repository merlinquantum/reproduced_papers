from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR, REPO_ROOT  # noqa: F401
from lib.sweep import candidate_name, expand_runs, load_plan, run_sweep, summarize

PLAN = {
    "sweep_id": "unit",
    "purpose": "test",
    "paper": "qfl_decentralized_data",
    "base_config": "configs/e3_reduced.json",
    "fixed_overrides": {},
    "axes": {"model.layers": [1, 2], "training.lr": [0.01, 0.1]},
    "seeds": [1, 2, 3],
    "workers": 2,
    "threads_per_worker": 1,
    "selection": {
        "metric": "test_approximation_ratio", "split": "test", "checkpoint": "last",
        "direction": "maximize", "aggregation": "mean", "tie_tolerance": 0.005,
    },
}  # fmt: skip


def test_expand_runs_is_the_full_candidate_by_seed_matrix():
    runs = expand_runs(PLAN)
    assert len(runs) == 2 * 2 * 3
    assert len({run["name"] for run in runs}) == len(runs)
    assert runs[0]["name"] == "layers1_lr0.01_seed1"
    assert candidate_name({"model.layers": 2, "training.lr": 0.1}) == "layers2_lr0.1"


def test_shipped_calibration_plan_is_valid_and_has_36_runs():
    plan = load_plan(
        PROJECT_DIR / "utils" / "sweep_plans" / "calibration_vqtn_e1_0v9.json"
    )
    assert len(expand_runs(plan)) == 36


def test_plan_without_a_selection_rule_is_rejected():
    broken = {key: value for key, value in PLAN.items() if key != "selection"}
    path = PROJECT_DIR / "outdir" / "_broken_plan.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(broken))
    try:
        with pytest.raises(KeyError, match="selection"):
            load_plan(path)
    finally:
        path.unlink()


def _write_run(sweep_dir, name, value):
    run_dir = sweep_dir / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "history": [
                    {"test_approximation_ratio": 0.0},
                    {"test_approximation_ratio": value},
                ]
            }
        )
    )
    return f"runs/{name}/metrics.json"


def _status(sweep_dir, values_by_candidate, failed=()):
    runs = []
    for run in expand_runs(PLAN):
        key = (run["candidate_name"], run["seed"])
        if key in failed:
            status, path = "FAILED", None
        else:
            status = "COMPLETED"
            path = _write_run(
                sweep_dir,
                run["name"],
                values_by_candidate[run["candidate_name"]][run["seed"] - 1],
            )
        runs.append({**run, "status": status, "metrics_path": path})
    return {"runs": runs}


def test_summary_selects_the_highest_mean_and_reports_a_complete_sweep(tmp_path):
    values = {c["candidate_name"]: [0.5, 0.5, 0.5] for c in expand_runs(PLAN)}
    values["layers2_lr0.1"] = [0.9, 0.92, 0.94]
    summary = summarize(PLAN, _status(tmp_path, values), tmp_path)
    assert summary["coverage"] == "COMPLETE SWEEP"
    assert summary["selected_candidates"] == ["layers2_lr0.1"]
    assert summary["margin"]["exceeds_variability"] is True
    best = next(
        row for row in summary["candidates"] if row["candidate"] == "layers2_lr0.1"
    )
    assert best["aggregate"] == pytest.approx(0.92)
    assert best["std"] == pytest.approx(0.02)


def test_summary_reports_ties_within_the_tolerance(tmp_path):
    values = {c["candidate_name"]: [0.5, 0.5, 0.5] for c in expand_runs(PLAN)}
    values["layers1_lr0.1"] = [0.8, 0.8, 0.8]
    values["layers2_lr0.1"] = [0.802, 0.802, 0.802]
    summary = summarize(PLAN, _status(tmp_path, values), tmp_path)
    assert set(summary["selected_candidates"]) == {"layers1_lr0.1", "layers2_lr0.1"}


def test_a_missing_run_makes_the_sweep_partial_and_not_a_winner(tmp_path):
    values = {c["candidate_name"]: [0.5, 0.5, 0.5] for c in expand_runs(PLAN)}
    values["layers2_lr0.1"] = [0.9, 0.9, 0.9]
    summary = summarize(
        PLAN, _status(tmp_path, values, failed={("layers2_lr0.1", 3)}), tmp_path
    )
    assert summary["coverage"] == "PARTIAL SWEEP"
    assert summary["selection_wording"] == "observed aggregate only"
    row = next(r for r in summary["candidates"] if r["candidate"] == "layers2_lr0.1")
    assert (row["successful_runs"], row["expected_runs"], row["status"]) == (
        2,
        3,
        "PARTIAL",
    )


def test_end_to_end_sweep_on_the_fast_qaoa_config(tmp_path):
    plan = {**PLAN, "axes": {"training.lr": [0.01, 0.1]}, "seeds": [1, 2]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    summary = run_sweep(plan_path, tmp_path / "sweep", REPO_ROOT)
    status = json.loads((tmp_path / "sweep" / "sweep_status.json").read_text())
    assert status["status"] == "COMPLETED" and summary["coverage"] == "COMPLETE SWEEP"
    assert [run["status"] for run in status["runs"]] == ["COMPLETED"] * 4
    log = (tmp_path / "sweep" / "sweep.log").read_text()
    assert log.count("SWEEP_STARTED") == 1 and log.count("SWEEP_COMPLETED") == 1
    assert log.count("CANDIDATE_COMPLETED") == 4
    ledger = json.loads((tmp_path / "sweep" / "ledger.json").read_text())["attempts"]
    assert len(ledger) == 4 and all(a["attempt"] == 1 for a in ledger)


def test_named_candidate_plans_expand_one_at_a_time():
    plan = {key: value for key, value in PLAN.items() if key != "axes"}
    plan["candidates"] = [
        {"name": "baseline", "overrides": {}},
        {"name": "wide", "overrides": {"model.encoding_scale": 2.0}},
    ]
    runs = expand_runs(plan)
    assert [run["name"] for run in runs[:3]] == [
        "baseline_seed1", "baseline_seed2", "baseline_seed3",
    ]  # fmt: skip
    assert len(runs) == 6 and runs[3]["axis_values"] == {"model.encoding_scale": 2.0}


def test_plan_with_both_axes_and_candidates_is_rejected(tmp_path):
    plan = {**PLAN, "candidates": [{"name": "x", "overrides": {}}]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(KeyError, match="exactly one"):
        load_plan(path)
