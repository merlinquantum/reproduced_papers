from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_python_files_parse() -> None:
    """Smoke test that key Python modules are syntactically valid."""
    targets = [
        PROJECT_ROOT / "implementation.py",
        PROJECT_ROOT / "lib" / "hidden_manifold.py",
        PROJECT_ROOT / "lib" / "imports.py",
        PROJECT_ROOT / "lib" / "kernels.py",
        PROJECT_ROOT / "lib" / "metrics.py",
        PROJECT_ROOT / "lib" / "ploting.py",
        PROJECT_ROOT / "lib" / "runner.py",
    ]

    for file_path in targets:
        assert file_path.is_file(), f"Missing module: {file_path}"
        source = file_path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(file_path))


def test_defaults_config_smoke_required_keys_exist() -> None:
    """Smoke test that the default config is readable and structurally valid."""
    config_path = PROJECT_ROOT / "configs" / "defaults.json"
    assert config_path.is_file(), f"Missing config file: {config_path}"

    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    assert cfg["seed"] is not None
    assert cfg["outdir"]
    assert cfg["dataset"]["name"]

    graphs = cfg["graphs"]
    assert graphs["number_of_points"] > 0
    assert graphs["min"] < graphs["max"]

    experiments = cfg["experiments"]
    assert isinstance(experiments, list) and len(experiments) > 0
