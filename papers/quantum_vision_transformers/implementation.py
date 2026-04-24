#!/usr/bin/env python3
"""Compatibility entry point for the QVT paper folder.

The shared repo-level runner is the primary entrypoint now. This shim keeps the
paper-local invocation working while the migration settles.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_lib import run_from_project


def main(argv: list[str] | None = None) -> int:
    project_dir = Path(__file__).resolve().parent
    run_from_project(project_dir, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
