from __future__ import annotations

import json
import subprocess
import sys

from common import PROJECT_DIR

CURATE = PROJECT_DIR / "utils" / "curate_results.py"


def make_run_dir(tmp_path, name="run_19990101-000000"):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    summary = {
        "experiment": "poisson_qpinn",
        "n_params": 48,
        "metrics": {"rmse": 1e-3, "mae": 8e-4, "l_inf": 2e-3, "nmse": 1e-4},
        "best": {"epoch": 10, "loss": 1e-2},
        "wall_time_sec": 12.5,
        "cfg": {
            "seed": 42,
            "model": {"cutoff": 8, "n_multi_layers": 2, "n_single_layers": 2},
            "training": {
                "epochs": 20,
                "lr": 0.01,
                "collocation_points": 16,
                "log_every": 5,
            },
            "logging": {"level": "info"},
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return run_dir


def curate(run_dir, results_dir, label, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(CURATE),
            str(run_dir),
            "--label",
            label,
            "--results-dir",
            str(results_dir),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_curate_writes_compact_json(tmp_path):
    run_dir = make_run_dir(tmp_path)
    results_dir = tmp_path / "results"
    proc = curate(run_dir, results_dir, "smoke_test_label")
    assert proc.returncode == 0, proc.stderr

    compact = json.loads((results_dir / "smoke_test_label.json").read_text())
    assert compact["label"] == "smoke_test_label"
    assert compact["experiment"] == "poisson_qpinn"
    assert compact["source_run"] == run_dir.name
    assert compact["seed"] == 42
    assert compact["metrics"]["rmse"] == 1e-3
    assert compact["model"]["cutoff"] == 8
    # Compact schema: key hyper-parameters only, no raw dumps.
    assert "cfg" not in compact
    assert compact["training"] == {"epochs": 20, "lr": 0.01, "collocation_points": 16}


def test_curate_resolves_nested_launcher_layout(tmp_path):
    parent = tmp_path / "poisson_sweep_seed42"
    make_run_dir(parent, name="run_19990101-000000")
    results_dir = tmp_path / "results"
    proc = curate(parent, results_dir, "nested_layout")
    assert proc.returncode == 0, proc.stderr
    compact = json.loads((results_dir / "nested_layout.json").read_text())
    assert compact["source_run"] == "run_19990101-000000"


def test_curate_rejects_ambiguous_parent(tmp_path):
    parent = tmp_path / "sweep"
    make_run_dir(parent, name="run_19990101-000000")
    make_run_dir(parent, name="run_19990101-000001")
    proc = curate(parent, tmp_path / "results", "ambiguous")
    assert proc.returncode != 0
    assert "contains 2 runs" in proc.stderr


def test_curate_with_arrays_writes_run_record(tmp_path):
    run_dir = make_run_dir(tmp_path)
    history = [{"epoch": 0, "total": 0.123456789}, {"epoch": 1, "total": 0.01}]
    predictions = {"x": [0.0, 1.0], "u_pred": [0.1234567891, 0.2], "u_ref": [0.1, 0.2]}
    (run_dir / "history.json").write_text(json.dumps(history))
    (run_dir / "predictions.json").write_text(json.dumps(predictions))
    results_dir = tmp_path / "results"
    proc = curate(run_dir, results_dir, "with_arrays", "--with-arrays")
    assert proc.returncode == 0, proc.stderr

    compact = json.loads((results_dir / "with_arrays.json").read_text())
    assert "history" not in compact
    record = json.loads((results_dir / "runs" / "with_arrays.json").read_text())
    assert record["label"] == "with_arrays"
    assert record["metrics"]["rmse"] == 1e-3
    assert [h["epoch"] for h in record["history"]] == [0, 1]
    assert record["history"][0]["total"] == 0.123457  # 6 significant digits
    assert record["predictions"]["u_pred"][0] == 0.123457
