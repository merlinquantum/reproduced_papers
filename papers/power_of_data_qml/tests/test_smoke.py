"""Tiny end-to-end run through the shared runtime, then the logging validator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from common import PROJECT_DIR, REPO_ROOT

from runtime_lib import run_from_project

DATA_ROOT = Path(REPO_ROOT) / "data"
VALIDATOR = Path(
    os.environ.get(
        "ALPS_LOGGING_VALIDATOR",
        Path.home()
        / ".claude/plugins/cache/alps-paper-reproduction/alps-paper-reproduction/0.1.0/"
        "skills/experiment-logging/scripts/validate_logging.py",
    )
)


@pytest.mark.skipif(
    not (DATA_ROOT / "power_of_data_qml" / "fashion_mnist").is_dir(),
    reason="Fashion-MNIST not acquired",
)
def test_tiny_study_run_writes_valid_evidence(tmp_path):
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--system-sizes",
                "3",
                "--seeds",
                "0",
                "--cv-system-sizes",
                "3",
                "--n-jobs",
                "4",
                "--outdir",
                str(tmp_path),
                "--data-root",
                str(DATA_ROOT),
            ],
        )
    finally:
        os.chdir(original_cwd)
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    components = {row["component"] for row in metrics["summary"]}
    assert components == {
        "dimension_geometry",
        "quantum_dataset",
        "classical_dataset",
        "bound",
        "engineered_pqk",
        "engineered_qk",
    }
    protocols = {row["protocol"] for row in metrics["summary"]}
    assert protocols == {None, "paper", "cv"}
    if VALIDATOR.is_file():
        result = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                "--run-dir",
                str(run_dir),
                "--require-training",
                "--require-evaluation",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not (DATA_ROOT / "power_of_data_qml" / "fashion_mnist").is_dir(),
    reason="Fashion-MNIST not acquired",
)
def test_photonic_scaling_probe_writes_one_record_per_point(tmp_path):
    config_path = tmp_path / "probe.json"
    config_path.write_text(
        json.dumps(
            {
                "description": "Tiny photonic scaling probe (test only).",
                "experiment": "photonic_scaling",
                "study": {"embeddings": ["PA", "PB"], "system_sizes": [4, 6]},
                "photonic_scaling": {
                    "subsample_seed": 0,
                    "points": [
                        {"embedding": "PA", "n": 4, "compare_cpu": True},
                        {"embedding": "PB", "n": 6, "max_photons": 2},
                    ],
                },
            }
        )
    )
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--config",
                str(config_path),
                "--outdir",
                str(tmp_path),
                "--data-root",
                str(DATA_ROOT),
            ],
        )
    finally:
        os.chdir(original_cwd)
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    points = json.loads((run_dir / "metrics.json").read_text())["points"]
    assert [p["status"] for p in points] == ["measured", "measured"]
    assert points[0]["max_difference_to_cpu_complex128"]["fidelity_kernel"] == 0.0
    assert points[1]["n_photons"] == 2 and points[1]["fock_dimension"] == 28
