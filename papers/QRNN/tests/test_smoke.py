from __future__ import annotations

import json
import os
from pathlib import Path

from common import PROJECT_DIR

from runtime_lib import run_from_project


def test_runtime_smoke(monkeypatch, tmp_path):
    original_cwd = Path.cwd()
    try:
        # Avoid network dependency (kagglehub) by generating a tiny local dataset
        # under the shared data root for this test.
        data_root = tmp_path / "data_root"
        paper_data = data_root / "QRNN"
        paper_data.mkdir(parents=True, exist_ok=True)

        rows = []
        for idx in range(50):
            rows.append(
                {
                    "timestamp": f"2020-01-01 {idx:02d}:00:00",
                    "temperature": 10.0 + idx * 0.1,
                    "humidity": 0.4 + idx * 0.001,
                }
            )
        csv_path = paper_data / "synthetic_weather.csv"
        csv_path.write_text(
            "timestamp,temperature,humidity\n"
            + "\n".join(
                f"{r['timestamp']},{r['temperature']},{r['humidity']}" for r in rows
            )
            + "\n",
            encoding="utf-8",
        )

        override_cfg = {
            "description": "Test-only override config (synthetic dataset)",
            "dataset": {
                "path": "synthetic_weather.csv",
                "kaggle_dataset": None,
                "preprocess": None,
                "target_column": "temperature",
                "feature_columns": ["temperature", "humidity"],
                "time_column": "timestamp",
            },
        }
        override_path = tmp_path / "override.json"
        override_path.write_text(json.dumps(override_cfg), encoding="utf-8")

        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--config",
                str(override_path),
                "--data-root",
                str(data_root),
                "--epochs",
                "1",
                "--outdir",
                str(tmp_path),
                "--batch-size",
                "4",
                "--sequence-length",
                "4",
            ],
        )
    finally:
        os.chdir(original_cwd)

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.csv").exists()
