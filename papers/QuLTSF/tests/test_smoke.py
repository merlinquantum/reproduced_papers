from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

import runtime_lib.runtime as runtime_module
from runtime_lib import run_from_project


def test_shared_runtime_dispatches_to_project_runner(monkeypatch, tmp_path):
    recorded: dict[str, Path] = {}

    def fake_import_callable(name: str):
        assert name == "lib.runner.train_and_evaluate"

        def _runner(cfg, run_dir: Path):
            recorded["cfg"] = cfg
            recorded["run_dir"] = run_dir

        return _runner

    monkeypatch.setattr(runtime_module, "import_callable", fake_import_callable)
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            ["--epochs", "1", "--outdir", str(tmp_path)],
        )
    finally:
        os.chdir(original_cwd)

    assert recorded.get("run_dir") == run_dir
    assert recorded["cfg"]["training"]["epochs"] == 1
    assert (run_dir / "config_snapshot.json").exists()
