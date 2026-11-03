"""Run the public phase-one task on a small physical circuit and data root."""

import json

from lib.runner import train_and_evaluate

from runtime_lib.dtypes import resolve_config_dtypes


def test_phase_one_artifacts_and_paired_noise_controls(tmp_path):
    data_root = tmp_path / "data"
    hub = data_root / "error-qcbm/hub"
    hub.mkdir(parents=True)
    snapshot = {
        "platform": "qpu:belenos",
        "http_status": 200,
        "queried_at_utc": "2026-10-17T00:00:00+00:00",
        "calibration_timestamp": None,
        "processor": {
            "perfs": {
                "Transmittance (%)": 50,
                "HOM (%)": 72.9,
                "g2 (%)": 1.1,
                "Clock (MHz)": 4.94,
            }
        },
    }
    (hub / "fixture.json").write_text(json.dumps(snapshot), encoding="utf-8")
    cfg = {
        "task": "phase1",
        "device": "cpu",
        "dtype": "float64",
        "data_root": str(data_root),
        "model": {"input_state": [1, 0, 1, 0, 0, 0]},
        "validation": {
            "seeds": [0, 1],
            "profiles": ["ideal", "paper_loss", "belenos_loss", "belenos_partial"],
            "shots": dict.fromkeys(
                ["ideal", "paper_loss", "belenos_loss", "belenos_partial"], 10000
            ),
            "native_shots": 100,
            "theory_seeds": [0, 1],
            "torch_threads": 1,
            "probability_atol": 1e-8,
            "snapshot_file": "fixture.json",
            "sampling_seed_offset": 100,
        },
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_and_evaluate(resolve_config_dtypes(cfg), run_dir)
    report = json.loads((run_dir / "phase1_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["trainable_parameters"] == 30
    assert len(report["records"]) == 8
    assert report["qpu_jobs_submitted"] == 0
    assert (data_root / "error-qcbm/gaussian/target.npz").is_file()
    assert (run_dir / "initializations.npz").is_file()
    for record in report["records"]:
        assert record["observed_shots"] == record["initial_shots"]
        assert record["sample_engine"] == "merlin_native_QuantumLayer_shots"
        assert record["theory_conditioned_max_probability_error"] < 1e-8
