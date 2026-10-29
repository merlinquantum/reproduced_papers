#!/usr/bin/env python3
"""Print a sweep's execution ledger (markdown) from ``sweep_status.json``.

Usage::

    .venv/bin/python papers/QK_LSTM/utils/sweep_ledger.py <SWEEP_DIR>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_PREFIX = str(Path(__file__).resolve().parents[3]) + "/"


def main(argv: list[str]) -> int:
    sweep_dir = Path(argv[0])
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    commit = status["code"]["commit"][:7] + (
        " (dirty)" if status["code"]["dirty"] else ""
    )
    print(
        "| Candidate | Seed / rep | Attempt | Status | Exact command | Run log | Run status | Config snapshot | Metric artifact | Code commit | Notes |"
    )
    print("|---|---|---:|---|---|---|---|---|---|---|---|")
    for run in status["runs"]:
        command = run["command"].replace(REPO_PREFIX, "")
        notes = json.dumps(run["error"]) if run.get("error") else "—"
        print(
            f"| {run['candidate']} | {run['seed']} / {run['repetition']} | {run['attempt']} | {run['status']} | `{command}` | "
            f"`{run['run_log']}` | `{run['run_status']}` | `{run['config_path']}` | `{run['metrics_path']}` | {commit} | {notes} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
