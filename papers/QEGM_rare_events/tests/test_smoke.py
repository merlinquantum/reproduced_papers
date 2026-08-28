from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

import runtime_lib.runtime as runtime_module
from runtime_lib import run_from_project


def test_placeholder(monkeypatch, tmp_path):
    recorded: dict[str, Path] = {}

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
            ["--epochs", "1", "--outdir", str(tmp_path)],
        )
    finally:
        os.chdir(original_cwd)

    assert recorded.get("run_dir") == run_dir
    assert (run_dir / "done.txt").exists()
