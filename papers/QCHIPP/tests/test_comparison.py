"""A comparison must account for every planned run with portable run paths."""

import json
import sys

from common import PROJECT_DIR

sys.path.insert(0, str(PROJECT_DIR / "utils"))

import run_comparison  # noqa: E402


def test_small_comparison_records_verified_real_run(tmp_path):
    output = run_comparison.main(
        [
            "--epochs", "1",
            "--seeds", "49",
            "--models", "logistic",
            "--tasks", "binding",
            "--output", str(tmp_path / "comparison"),
        ]
    )  # fmt: skip
    status = json.loads((output / "sweep_status.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["expected_runs"] == 1
    assert len(status["runs"]) == 1
    assert summary["rows"][0]["completed_runs"] == 1
    recorded = summary["runs"][0]["run_dir"]
    assert not recorded.startswith("/") and ":" not in recorded
    assert (output / recorded / "metrics.json").exists()


def test_config_for_maps_binding_logistic_to_defaults():
    assert run_comparison.config_for("binding", "logistic") == "configs/defaults.json"
    assert (
        run_comparison.config_for("immunogenic", "merlin_reference")
        == "configs/immunogenic_merlin_reference.json"
    )
    for task in ("binding", "immunogenic"):
        for model in run_comparison.CONFIG_STEMS:
            assert (PROJECT_DIR / run_comparison.config_for(task, model)).exists()
