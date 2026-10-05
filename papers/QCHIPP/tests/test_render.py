"""Result rendering resolves run directories independently of the machine."""

import sys
from pathlib import Path, PurePosixPath

from common import PROJECT_DIR

sys.path.insert(0, str(PROJECT_DIR / "utils"))

from render_results import resolve_run_dir  # noqa: E402


def test_relative_run_dir_resolves_inside_sweep(tmp_path):
    resolved = resolve_run_dir(tmp_path, "binding_logistic_49/run_20261002-060737")
    assert resolved == tmp_path / "binding_logistic_49" / "run_20261002-060737"


def test_legacy_absolute_container_path_resolves_inside_sweep(tmp_path):
    legacy = str(
        PurePosixPath(
            "/reproduced_papers/papers/QCHIPP/outdir/comparison100/"
            "binding_gate_source_49/run_20261002-060326"
        )
    )
    resolved = resolve_run_dir(tmp_path, legacy)
    assert resolved == tmp_path / "binding_gate_source_49" / "run_20261002-060326"
    assert Path(legacy).parts[-2:] == resolved.parts[-2:]
