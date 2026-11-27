"""Exercise the real CLI parser and runtime without repeated library imports."""

import json
from pathlib import Path

import pytest

from implementation import main

ROOT = next(
    path
    for path in Path(__file__).resolve().parents
    if (path / "implementation.py").is_file()
)


def test_discovery_and_help(capsys):
    assert main(["--list-papers"]) == 0
    assert "error-qcbm" in capsys.readouterr().out
    assert main(["--paper", "error-qcbm", "--help"]) == 0
    assert "--task" in capsys.readouterr().out


def test_overlay_selects_training_with_missing_settings(tmp_path, monkeypatch):
    overlay = tmp_path / "train.json"
    overlay.write_text(
        json.dumps({"description": "Verify training boundary", "task": "train"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(ROOT)
    with pytest.raises(ValueError, match="Training requires explicit"):
        main(
            [
                "--paper",
                "error-qcbm",
                "--config",
                str(overlay),
                "--outdir",
                str(tmp_path / "runs"),
            ]
        )
    snapshot = next((tmp_path / "runs").glob("run_*/config_snapshot.json"))
    assert json.loads(snapshot.read_text(encoding="utf-8"))["task"] == "train"
    assert not list((tmp_path / "runs").glob("run_*/environment.json"))
