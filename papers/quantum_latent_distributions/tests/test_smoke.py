"""End-to-end run through the shared runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path

from common import PROJECT_DIR

from runtime_lib import run_from_project


def test_synthetic_smoke_run_writes_metrics(tmp_path):
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--config",
                str(PROJECT_DIR / "configs" / "smoke.json"),
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)

    assert (run_dir / "config_snapshot.json").exists()
    assert json.loads((run_dir / "done.json").read_text())["status"] == "complete"

    records = json.loads((run_dir / "metrics.json").read_text())
    assert {r["latent"] for r in records} == {"gaussian", "boson"}
    for record in records:
        assert record["l1_nearest_int"] >= 0.0


def test_sampler_validation_smoke_run(tmp_path):
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--experiment",
                "sampler_validation",
                "--seed",
                "3",
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)

    summary = json.loads((run_dir / "sampler_validation.json").read_text())
    convergence = summary["convergence"]
    # An unbiased sampler must get closer to the exact distribution with more shots.
    assert convergence[-1]["tvd"] < convergence[0]["tvd"]
    assert summary["control"]["max_mean_occupancy_difference"] < 0.05
    # First moment matches, second moment does not -- see the challenger study.
    # The ratio is what holds at any size; the absolute values depend on filling.
    assert (
        summary["control"]["fano_quantum"]
        > 1.2 * summary["control"]["fano_distinguishable"]
    )
    assert summary["control"]["tvd_distinguishable_vs_quantum"] > 0.1
