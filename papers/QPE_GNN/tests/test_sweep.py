"""Sweep plans, the sweep coordinator and the sweep summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from utils.run_sweep import expand_plan, run_sweep  # noqa: E402
from utils.summarise_sweep import summarise_sweep, verify_summary  # noqa: E402


@pytest.mark.parametrize(
    ("plan_name", "expected_runs"), [("zinc_reduced", 16), ("synthetic", 20)]
)
def test_committed_plans_expand_to_the_planned_runs(plan_name, expected_runs):
    plan = json.loads(
        (PAPER_ROOT / "configs" / "sweeps" / f"{plan_name}.json").read_text()
    )
    runs = expand_plan(plan)
    assert len(runs) == expected_runs
    for run in runs:
        assert (PAPER_ROOT / run["args"][1]).is_file()
    assert [run["seed"] for run in runs[:4]] == [0, 0, 0, 0]


def test_reduced_zinc_configs_differ_only_where_planned():
    def load(name):
        return json.loads((PAPER_ROOT / "configs" / f"{name}.json").read_text())

    for encoding in ("rrwp", "cqrw1", "qirw2"):
        original, reduced = (
            load(f"zinc_{encoding}_original"),
            load(f"zinc_{encoding}_reduced"),
        )
        changed = {
            key
            for key in original.keys() | reduced.keys()
            if original.get(key) != reduced.get(key)
        }
        assert changed == {
            "description",
            "feasibility",
            "epochs",
            "warmup_epochs",
            "checkpoint_every",
        }
        assert reduced["epochs"] == 200 and reduced["warmup_epochs"] == 5
    control = load("zinc_rrwp41_reduced")
    quantum = load("zinc_qirw2_reduced")
    assert control["encoding"] == "rrwp"
    assert control["rrwp_dim"] == quantum["rrwp_dim"] + quantum["qpe_dim"]


def test_run_seeds_keep_the_full_repetition_index():
    plan = {
        "entries": [
            {"name": "a", "config": "c.json", "seeds": [0, 1, 2], "run_seeds": [2]}
        ]
    }
    (run,) = expand_plan(plan)
    assert run["run_id"] == "a_seed2" and run["repetition"] == 2
    with pytest.raises(ValueError, match="not in seeds"):
        expand_plan({"entries": [{**plan["entries"][0], "run_seeds": [5]}]})


def _tiny_config(path: Path, encoding: str, rrwp_dim: int) -> None:
    path.write_text(
        json.dumps(
            {
                "description": "sweep test",
                "feasibility": {"status": "supported"},
                "dataset": "graph_reg",
                "dataset_kwargs": {
                    "num_graphs": 30,
                    "n_range": [5, 7],
                    "p": 0.4,
                    "seed": 0,
                },
                "model": "grit_lite",
                "encoding": encoding,
                "rrwp_dim": rrwp_dim,
                "node_dim": 8,
                "depth": 1,
                "num_heads": 2,
                "head": "graph_reg",
                "epochs": 2,
                "batch_size": 8,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "train_frac": 0.7,
                "val_frac": 0.15,
                "split_seed": 11,
                "device": "cpu",
            }
        )
    )


def _write_tiny_plan(tmp_path: Path) -> Path:
    _tiny_config(tmp_path / "small.json", "rrwp", 3)
    _tiny_config(tmp_path / "wide.json", "rrwp", 5)
    plan = {
        "sweep_id": "test_sweep",
        "max_epochs": 2,
        "metric": "test_mae",
        "direction": "minimize",
        "split": "test",
        "checkpoint_rule": "best validation MAE",
        "aggregation": "mean and sample std",
        "tie_rule": "margin > 2 std and Wilcoxon p < 0.05",
        "selection_rule": "none",
        "entries": [
            {"name": "small", "config": str(tmp_path / "small.json"), "seeds": [0, 1]},
            {"name": "wide", "config": str(tmp_path / "wide.json"), "seeds": [0, 1]},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    return plan_path


def test_plan_split_by_seed_combines_into_one_complete_summary(tmp_path):
    plan_path = _write_tiny_plan(tmp_path)
    first, second = tmp_path / "launch_seed0", tmp_path / "launch_seed1"
    assert run_sweep(plan_path, first, "cpu", sys.executable, 2, run_seeds=[0]) == 0
    assert run_sweep(plan_path, second, "cpu", sys.executable, 2, run_seeds=[1]) == 0

    half = json.loads((first / "sweep_summary.json").read_text())
    assert half["verdict"] == "PARTIAL SWEEP" and half["observed_runs"] == 2
    ledger = json.loads((second / "sweep_status.json").read_text())
    assert ledger["run_seeds"] == [1]
    assert [run["repetition"] for run in ledger["runs"]] == [1, 1]

    combined_dir = tmp_path / "combined"
    summary_path = summarise_sweep([first, second], plan_path, combined_dir)
    combined = json.loads(summary_path.read_text())
    assert combined["verdict"] == "COMPLETE SWEEP"
    assert combined["observed_runs"] == combined["expected_runs"] == 4
    assert [launch["run_seeds"] for launch in combined["launches"]] == [[0], [1]]
    assert combined["candidates"][0]["run_dirs"][1].startswith("launch_seed1/")
    assert verify_summary([first, second], plan_path, summary_path)
    with pytest.raises(ValueError, match="appears twice"):
        summarise_sweep([first, first], plan_path, tmp_path / "duplicate")
    with pytest.raises(ValueError, match="not planned"):
        run_sweep(plan_path, tmp_path / "bad", "cpu", sys.executable, 1, run_seeds=[7])


def test_sweep_runs_every_planned_run_and_its_summary_verifies(tmp_path):
    plan_path = _write_tiny_plan(tmp_path)
    sweep_dir = tmp_path / "sweep"

    assert run_sweep(plan_path, sweep_dir, "cpu", sys.executable, workers=2) == 0

    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert [run["status"] for run in status["runs"]] == ["COMPLETED"] * 4
    summary = json.loads((sweep_dir / "sweep_summary.json").read_text())
    assert summary["verdict"] == "COMPLETE SWEEP"
    assert [c["observed_runs"] for c in summary["candidates"]] == [2, 2]
    (comparison,) = summary["comparisons"]
    assert comparison["test"] == "wilcoxon_signed_rank_seed_averaged_absolute_errors"
    assert 0.0 <= comparison["p_value"] <= 1.0
    assert verify_summary(sweep_dir, plan_path, sweep_dir / "sweep_summary.json")
    with pytest.raises(FileExistsError):
        run_sweep(plan_path, sweep_dir, "cpu", sys.executable, workers=1)

    # A run whose metrics no longer match its recorded hash is excluded.
    metrics_path = sweep_dir / status["runs"][0]["metrics_path"]
    metrics_path.write_text(metrics_path.read_text() + " ")
    summarise_sweep(sweep_dir, plan_path, tmp_path / "resummarised")
    resummarised = json.loads(
        (tmp_path / "resummarised" / "sweep_summary.json").read_text()
    )
    assert resummarised["verdict"] == "PARTIAL SWEEP"
    assert resummarised["excluded_runs"][0]["run_id"] == status["runs"][0]["run_id"]
