"""Sweep coordinator: run configs x seeds through the shared runtime, in parallel.

Each planned run is one ``(config, seed)`` pair, executed as
``python implementation.py --paper distributed_secure_kernels --config <config>
--seed <seed> --outdir <sweep_dir>/<candidate>_seed<seed>`` in its own process
(its own ``run.log``). Only this coordinator writes ``sweep.log`` and
``sweep_status.json``. After the runs end, the summary is regenerated from the
runs' ``metrics.json`` files by :mod:`utils.summarize_sweep`.

The sweeps of this reproduction are seed repetitions (shot noise, oversampling),
not hyperparameter selections: ``selected_candidates`` stays empty.

Usage (from the repository root)::

    .venv/bin/python papers/distributed_secure_kernels/utils/run_sweep.py \
        --sweep-id table1_parkinsons --seeds 0 1 2 3 4 --workers 5 \
        --configs configs/accuracy_parkinsons_original.json configs/accuracy_parkinsons_reference.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parents[1]
sys.path[:0] = [str(PROJECT_DIR.parent), str(REPO_DIR)]

from distributed_secure_kernels.lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)
from distributed_secure_kernels.utils.summarize_sweep import summarize  # noqa: E402

logger = logging.getLogger("sweep")


def _run_entry(sweep_dir: Path, config: str, seed: int, repetition: int) -> dict:
    candidate = Path(config).stem
    return {
        "run_id": f"{candidate}_seed{seed}",
        "candidate": {"config": config},
        "seed": seed,
        "repetition": repetition,
        "status": "PLANNED",
        "run_dir": None,
        "run_log": None,
        "run_status": None,
        "config_path": None,
        "metrics_path": None,
        "attempt": 0,
        "command": None,
    }


def _refresh(entry: dict, sweep_dir: Path) -> None:
    """Read a worker's run directory and copy its status and progress."""

    base = sweep_dir / entry["run_id"]
    runs = sorted(base.glob("run_*")) if base.exists() else []
    if not runs:
        return
    run_dir = runs[-1]
    rel = os.path.relpath(run_dir, sweep_dir)
    entry.update(
        run_dir=rel,
        run_log=f"{rel}/run.log",
        run_status=f"{rel}/run_status.json",
        config_path=f"{rel}/config_snapshot.json",
        metrics_path=f"{rel}/metrics.json",
    )
    status_path = run_dir / "run_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            return
        if status.get("progress"):
            entry["progress"] = status["progress"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    sweep_dir = PROJECT_DIR / "outdir" / f"sweep_{args.sweep_id}_{stamp}"
    sweep_dir.mkdir(parents=True)
    configure_logging(sweep_dir / "sweep.log")
    entries = [
        _run_entry(sweep_dir, config, seed, repetition)
        for config in args.configs
        for repetition, seed in enumerate(args.seeds)
    ]
    status = {
        "schema_version": 1,
        "sweep_id": args.sweep_id,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_DIR),
        "expected_runs": len(entries),
        "runs": entries,
        "summary_path": None,
        "selected_candidates": [],
        "error": None,
    }
    write_json(sweep_dir / "sweep_status.json", status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | workers=%d | configs=%s | seeds=%s",
        args.sweep_id,
        len(entries),
        args.workers,
        args.configs,
        args.seeds,
    )
    # Share the CPU between workers instead of oversubscribing BLAS threads.
    threads = str(max(1, (os.cpu_count() or 1) // max(1, args.workers)))
    env = {
        **os.environ,
        "OMP_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
    }
    pending, running = list(entries), {}
    try:
        while pending or running:
            while pending and len(running) < args.workers:
                entry = pending.pop(0)
                outdir = sweep_dir / entry["run_id"]
                command = [
                    sys.executable,
                    str(REPO_DIR / "implementation.py"),
                    "--paper",
                    PROJECT_DIR.name,
                    "--config",
                    entry["candidate"]["config"],
                    "--seed",
                    str(entry["seed"]),
                    "--outdir",
                    str(outdir),
                    # Identity recorded in sweep_status.json, copied into run_status.json.
                    "--sweep-id",
                    args.sweep_id,
                    "--run-id",
                    entry["run_id"],
                    "--repetition",
                    str(entry["repetition"]),
                    "--candidate-config",
                    entry["candidate"]["config"],
                ]
                entry.update(
                    status="RUNNING",
                    attempt=entry["attempt"] + 1,
                    command=" ".join(command[1:]),
                )
                running[entry["run_id"]] = (
                    subprocess.Popen(
                        command,
                        cwd=REPO_DIR,
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    ),
                    entry,
                )
                logger.info(
                    "CANDIDATE_STARTED | run_id=%s | candidate=%s | seed=%d",
                    entry["run_id"],
                    entry["candidate"],
                    entry["seed"],
                )
                write_json(sweep_dir / "sweep_status.json", status)
            time.sleep(5)
            for run_id, (process, entry) in list(running.items()):
                _refresh(entry, sweep_dir)
                code = process.poll()
                if code is None:
                    continue
                del running[run_id]
                # Keep the process exit status as evidence, whatever the outcome.
                entry["exit_code"] = code
                run_status = {}
                if entry["run_status"]:
                    run_status = json.loads(
                        (sweep_dir / entry["run_status"]).read_text()
                    )
                metrics_ok = (
                    entry["metrics_path"]
                    and (sweep_dir / entry["metrics_path"]).exists()
                )
                if code == 0 and run_status.get("status") == "COMPLETED" and metrics_ok:
                    entry["status"] = "COMPLETED"
                    entry.pop("progress", None)
                    logger.info(
                        "CANDIDATE_COMPLETED | run_id=%s | exit_code=%d | metrics=%s",
                        run_id,
                        code,
                        entry["metrics_path"],
                    )
                else:
                    entry["status"] = "FAILED"
                    entry["error"] = {
                        "exit_code": code,
                        "run_status": run_status.get("status"),
                        "error": run_status.get("error"),
                    }
                    logger.error(
                        "CANDIDATE_FAILED | run_id=%s | exit_code=%s | error=%s",
                        run_id,
                        code,
                        run_status.get("error"),
                    )
            write_json(sweep_dir / "sweep_status.json", status)
            if status["status"] == "RUNNING":
                logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))
        summary_path = summarize(sweep_dir)
        status["summary_path"] = os.path.relpath(summary_path, sweep_dir)
        completed = sum(entry["status"] == "COMPLETED" for entry in entries)
        status["status"] = (
            "COMPLETED"
            if completed == len(entries)
            else ("PARTIAL" if completed else "FAILED")
        )
        status["completed_at"] = utc_now()
        write_json(sweep_dir / "sweep_status.json", status)
        event = {
            "COMPLETED": "SWEEP_COMPLETED",
            "PARTIAL": "SWEEP_PARTIAL",
            "FAILED": "SWEEP_FAILED",
        }[status["status"]]
        logger.info(
            "%s | summary=%s | completed=%d/%d | selected=%s",
            event,
            status["summary_path"],
            completed,
            len(entries),
            [],
        )
    except Exception as exc:
        for process, _ in running.values():
            process.terminate()
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(sweep_dir / "sweep_status.json", status)
        logger.exception("SWEEP_FAILED | sweep_id=%s", args.sweep_id)
        raise
    return 0 if status["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
