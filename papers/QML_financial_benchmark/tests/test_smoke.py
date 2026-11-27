from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

import runtime_lib.runtime as runtime_module
from runtime_lib import run_from_project


def test_runtime_applies_paper_flags(monkeypatch, tmp_path):
    recorded: dict = {}

    def fake_import_callable(name: str):
        assert name == "lib.runner.train_and_evaluate"

        def _runner(cfg, run_dir: Path):
            recorded["cfg"] = cfg
            run_dir.mkdir(parents=True, exist_ok=True)

        return _runner

    monkeypatch.setattr(runtime_module, "import_callable", fake_import_callable)
    original_cwd = Path.cwd()
    try:
        run_from_project(
            PROJECT_DIR,
            [
                "--ticker",
                "GARAN.IS",
                "--regime",
                "F3",
                "--split-mode",
                "within_regime",
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)

    cfg = recorded["cfg"]
    assert cfg["directional"]["dataset"]["ticker"] == "GARAN.IS"
    assert cfg["trading"]["dataset"]["regime"] == "F3"
    assert cfg["trading"]["dataset"]["split_mode"] == "within_regime"
