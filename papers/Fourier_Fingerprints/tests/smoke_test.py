from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_python_files_parse() -> None:
    """Check that the main Fourier Fingerprints Python modules parse."""
    targets = [
        PROJECT_ROOT / "implementation.py",
        PROJECT_ROOT / "lib" / "fourier_1D.py",
        PROJECT_ROOT / "lib" / "fourier_2D.py",
        PROJECT_ROOT / "lib" / "runner.py",
    ]

    for file_path in targets:
        assert file_path.is_file(), f"Missing module: {file_path}"
        source = file_path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(file_path))


def test_configs_smoke_required_keys_exist() -> None:
    """Check that every Fourier Fingerprints config has the required schema."""
    config_paths = sorted((PROJECT_ROOT / "configs").glob("*.json"))
    assert config_paths, "No configuration files found."

    for config_path in config_paths:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["outdir"]
        assert cfg["graph_name"]
        assert cfg["circuits"]
        assert cfg["encoding"] in {"linear", "exponential", "balanced"}
        assert cfg["dimension"] in {1, 2}


def test_runner_dispatch_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Check that the runner dispatches configurations to the right dimension."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "lib"))
    import runner

    calls: list[tuple[str, dict[str, object]]] = []

    def record_1d(**kwargs: object) -> None:
        calls.append(("1d", kwargs))

    def record_2d(**kwargs: object) -> None:
        calls.append(("2d", kwargs))

    monkeypatch.setattr(runner, "main_1d", record_1d)
    monkeypatch.setattr(runner, "main_2d", record_2d)

    base_config = {
        "circuits": ["circuit_0"],
        "encoding": "linear",
        "graph_name": "smoke",
    }
    runner._run_experiment({**base_config, "dimension": 1}, tmp_path / "run_1d")
    runner._run_experiment({**base_config, "dimension": 2}, tmp_path / "run_2d")

    assert [dimension for dimension, _ in calls] == ["1d", "2d"]
    assert calls[0][1]["facteur_echelle"] == "linear"
    assert calls[1][1]["name"] == "smoke"
    assert calls[1][1]["rundir"] == tmp_path / "run_2d"
