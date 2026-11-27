#!/usr/bin/env python3
"""Thin wrapper that delegates to the repository-wide runner."""

from __future__ import annotations

import sys
from pathlib import Path

from runtime_lib import run_from_project


def main(argv: list[str] | None = None) -> int:
    run_from_project(Path(__file__).resolve().parent, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
