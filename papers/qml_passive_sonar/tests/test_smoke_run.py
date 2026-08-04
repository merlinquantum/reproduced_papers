"""End-to-end smoke test of the runner with the defaults config."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_implementation_smoke_run(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    paper_dir = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "implementation.py"),
        "--paper",
        "qml_passive_sonar",
        "--outdir",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=paper_dir)
    assert result.returncode == 0, result.stderr
    runs = list(tmp_path.glob("run_*"))
    assert runs, "no run directory was created"
    snapshot = runs[0] / "config_snapshot.json"
    assert snapshot.exists()
    summary = runs[0] / "summary.json"
    assert summary.exists()
    payload = json.loads(summary.read_text())
    assert payload["status"] == "ok"
