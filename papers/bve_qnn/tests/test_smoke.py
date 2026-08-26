from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from common import PROJECT_DIR

from runtime_lib import run_from_project

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def test_runtime_smoke(tmp_path):
    original_cwd = Path.cwd()
    try:
        run_dir = run_from_project(
            PROJECT_DIR,
            [
                "--total-steps",
                "2",
                "--batch-size",
                "32",
                "--outdir",
                str(tmp_path),
            ],
        )
    finally:
        os.chdir(original_cwd)

    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "exp1_merlin_results.npz").exists()
    assert "psi_pred_training" in np.load(run_dir / "exp1_merlin_results.npz").files
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "done.txt").exists()


def test_checkpoint_path_resolution_supports_direct_and_model_paths(tmp_path):
    from lib.runner import _resolve_checkpoint_path

    direct_checkpoint = tmp_path / "direct.pt"
    direct_checkpoint.write_bytes(b"checkpoint")

    original_cwd = Path.cwd()
    models_checkpoint = PROJECT_DIR / "models" / "local_test_checkpoint.pt"
    try:
        os.chdir(PROJECT_DIR)
        models_checkpoint.write_bytes(b"checkpoint")

        assert (
            _resolve_checkpoint_path({"model": {"checkpoint": str(direct_checkpoint)}})
            == direct_checkpoint
        )
        assert (
            _resolve_checkpoint_path({"model": {"checkpoint": models_checkpoint.name}})
            == Path("models") / models_checkpoint.name
        )
    finally:
        os.chdir(original_cwd)
        if models_checkpoint.exists():
            models_checkpoint.unlink()
