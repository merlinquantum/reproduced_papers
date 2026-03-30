#!/usr/bin/env python3
"""Convenience wrapper around the shared time-series metrics plotter.

Prefer the shared version:
    python -m papers.shared.time_series.plot_metrics <run_dir>

This wrapper exists for backwards compatibility:
    python papers/QRNN/utils/plot_metrics.py <run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> int:
    _ensure_repo_root_on_syspath()
    from papers.shared.time_series.plot_metrics import main as shared_main

    return int(shared_main())


if __name__ == "__main__":
    raise SystemExit(main())
