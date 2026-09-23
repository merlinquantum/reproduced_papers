"""Shared-runtime wiring test: CLI overrides must reach the paper runner."""

from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

import runtime_lib.runtime as runtime_module
from runtime_lib import run_from_project


def test_runtime_passes_cli_overrides_to_the_runner(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}

    def fake_import_callable(name: str):
        assert name == "lib.runner.train_and_evaluate"

        def _runner(cfg, run_dir: Path):
            recorded["cfg"] = cfg
            recorded["run_dir"] = run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "done.txt").write_text("ok", encoding="utf-8")

        return _runner

    monkeypatch.setattr(runtime_module, "import_callable", fake_import_callable)
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--experiment", "reservoir_instance_sweep",
                "--n-instances", "3",
                "--selection-split", "test",
                "--horizons", "1",
                "--outdir", str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)

    assert recorded["run_dir"] == run_dir
    assert (run_dir / "done.txt").exists()
    cfg = recorded["cfg"]
    assert cfg["experiment"] == "reservoir_instance_sweep"
    assert cfg["instance_sweep"]["n_instances"] == 3
    assert cfg["evaluation"]["selection_split"] == "test"
    assert cfg["evaluation"]["horizons"] == [1]
    # Overrides must not clobber the paper-accurate settings they do not touch.
    assert cfg["quantum_reservoir"]["n_qubits_total"] == 10
    assert cfg["evaluation"]["n_out_of_sample"] == 245
