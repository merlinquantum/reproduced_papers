from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIRNAME = "data"
ENV_DATA_ROOT = "DATA_DIR"


def find_repo_root() -> Path:
    """Best-effort repo root detection (implementation.py or .git marker)."""
    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for path in candidates:
        if (path / "implementation.py").exists() or (path / ".git").exists():
            return path
    return Path(__file__).resolve().parent


def resolve_data_root(
    preferred_root: str | Path | None = None, project_dir: Path | None = None
) -> Path:
    """Resolve the shared data root.

    Precedence:
    1) Explicit preferred_root (CLI/config)
    2) DATA_DIR environment variable
    3) <repo_root>/data (or relative to project_dir when provided)
    """

    if preferred_root:
        root = Path(preferred_root).expanduser()
    elif ENV_DATA_ROOT in os.environ:
        root = Path(os.environ[ENV_DATA_ROOT]).expanduser()
    else:
        root = find_repo_root() / DEFAULT_DATA_DIRNAME

    if not root.is_absolute():
        base = project_dir.resolve() if project_dir else find_repo_root()
        root = (base / root).resolve()

    return root


def paper_data_dir(
    paper_name: str | None = None,
    data_root: str | Path | None = None,
    ensure_exists: bool = True,
) -> Path:
    """Return the data directory for a paper (defaults to CWD name).

    Creates the directory by default so callers can immediately write/download.
    """

    root = resolve_data_root(data_root)
    name = paper_name or Path.cwd().name
    path = (root / name).resolve()
    if ensure_exists:
        path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "ENV_DATA_ROOT",
    "DEFAULT_DATA_DIRNAME",
    "find_repo_root",
    "resolve_data_root",
    "paper_data_dir",
]
