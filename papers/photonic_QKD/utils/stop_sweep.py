"""Gracefully stop a running sweep (its runs keep their last epoch checkpoint).

Sends SIGTERM to the coordinator PID recorded in ``coordinator.pid``. The
coordinator forwards it to its runs, waits for them to finish writing their
FAILED status, and closes the sweep as PARTIAL. Resume later with
``utils/run_sweep.py --resume <sweep dir>``.

Run inside the same WSL distribution as the sweep (from the repository root)::

    python papers/photonic_QKD/utils/stop_sweep.py --latest
    python papers/photonic_QKD/utils/stop_sweep.py <sweep dir>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

SWEEPS_DIR = Path(__file__).resolve().parents[1] / "outdir" / "sweeps"


def latest_running_sweep() -> Path:
    """Most recently started sweep whose status is still RUNNING."""
    candidates = sorted(
        SWEEPS_DIR.glob("*/sweep_status.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for status_path in candidates:
        if json.loads(status_path.read_text())["status"] == "RUNNING":
            return status_path.parent
    raise SystemExit(f"No RUNNING sweep under {SWEEPS_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dir", nargs="?", type=Path)
    parser.add_argument(
        "--latest", action="store_true", help="Stop the most recent RUNNING sweep"
    )
    args = parser.parse_args()
    if args.latest == (args.sweep_dir is not None):
        parser.error("give either a sweep directory or --latest")
    sweep_dir = latest_running_sweep() if args.latest else args.sweep_dir.resolve()
    pid = int((sweep_dir / "coordinator.pid").read_text())
    os.kill(pid, signal.SIGTERM)
    print(
        f"Sent SIGTERM to coordinator {pid} of {sweep_dir.name}; runs stop after their current step."
    )
    print(
        f"Resume later with: python papers/photonic_QKD/utils/run_sweep.py --resume {sweep_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
