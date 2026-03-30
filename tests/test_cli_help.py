from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "implementation.py").exists():
            return parent
    raise RuntimeError("Could not locate repo root containing implementation.py")


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    repo_root = _repo_root()
    cmd = [sys.executable, str(repo_root / "implementation.py"), *args]
    return subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_alone_shows_general_message() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "Tip: use --paper NAME --help" in result.stdout


def test_help_with_valid_paper_shows_paper_usage() -> None:
    result = _run_cli("--paper", "QORC", "--help")
    assert result.returncode == 0
    assert "Options for paper QORC" in result.stdout
    assert "usage: implementation.py --paper QORC [options]" in result.stdout
    assert "--list-papers" not in result.stdout


def test_help_with_invalid_paper_errors() -> None:
    result = _run_cli("--paper", "xxxx", "--help")
    assert result.returncode == 2
    assert "Error: unknown paper 'xxxx'" in result.stderr
    assert result.stdout.strip() == ""


def test_list_papers_includes_qorc() -> None:
    result = _run_cli("--list-papers")
    assert result.returncode == 0
    assert "papers/QORC" in result.stdout
