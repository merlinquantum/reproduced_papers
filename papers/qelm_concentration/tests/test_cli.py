from __future__ import annotations

import os
from pathlib import Path

from common import PROJECT_DIR

import runtime_lib.runtime as runtime_module
from runtime_lib import run_from_project


def test_cli_flags_reach_the_runner(monkeypatch, tmp_path):
    recorded: dict = {}

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
            [
                "--config",
                "configs/pareto_search_n8.json",
                "--n-qubits",
                "10",
                "--n-trials",
                "7",
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)

    assert recorded["run_dir"] == run_dir
    cfg = recorded["cfg"]
    assert cfg["task"] == "pareto_search"
    assert cfg["reservoir"]["n_qubits"] == 10
    assert cfg["pareto_search"]["n_trials"] == 7
    assert Path(cfg["data_root"]) == PROJECT_DIR.parent.parent / "data"
