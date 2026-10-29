#!/usr/bin/env python3
"""Sweep coordinator for the QK-LSTM reproduction.

Runs every (candidate, seed) pair of a plan as an isolated worker process of the
shared runtime (``implementation.py --paper QK_LSTM``). Each worker writes its
own ``run.log``; only this coordinator writes ``sweep.log`` and
``sweep_status.json`` in the sweep directory. After the last worker, the summary
table is computed from the workers' ``metrics.json`` files only.

Plan file (JSON)::

    {
      "sweep_name": "paper_hp",
      "seeds": [0, 1, 2, 3, 4],
      "candidates": [
        {"label": "qb_quantum", "config": "configs/aqi_qklstm_count209_original.json",
         "args": ["--lr", "0.001"]}
      ],
      "selection": {"metric": "test.observed_targets.rmse" | "validation.observed_targets.rmse" | null,
                    "direction": "minimize", "tie_tolerance_relative": 0.01}
    }

Usage::

    .venv/bin/python papers/QK_LSTM/utils/sweep.py --plan papers/QK_LSTM/configs/sweeps/<plan>.json --max-parallel 2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(PROJECT_DIR), str(REPO_ROOT)]

from lib.experiment_logging import (  # noqa: E402
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)

logger = logging.getLogger("qk_lstm.sweep")
SUMMARY_METRICS = ("rmse", "mae", "mape", "r2")


def _configure_sweep_logging(sweep_dir: Path) -> None:
    handler = logging.FileHandler(sweep_dir / "sweep.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)sZ [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S"
        )
    )
    logging.Formatter.converter = time.gmtime
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(handler.formatter)
    logger.handlers[:] = [handler, stream]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _relative(path: Path, base: Path) -> str:
    return str(Path(path).resolve().relative_to(base.resolve())) if path else None


def _metric_from_path(metrics: dict, dotted: str) -> float:
    split, scoring, name = dotted.split(".")
    return float(metrics["splits"][split][scoring][name])


def _find_run_dir(worker_outdir: Path) -> Path | None:
    runs = sorted(worker_outdir.glob("run_*"))
    return runs[-1] if runs else None


def _epochs_completed(run_dir: Path | None) -> int:
    # Progress only (never used for values or rankings).
    if run_dir is None or not (run_dir / "run.log").exists():
        return 0
    return (
        (run_dir / "run.log").read_text(encoding="utf-8").count("TRAIN_EPOCH_COMPLETED")
    )


def build_summary(status: dict, sweep_dir: Path, selection: dict) -> dict:
    """Aggregate test/validation metrics per candidate from metrics.json files."""
    by_candidate: dict[str, list[dict]] = {}
    for run in status["runs"]:
        by_candidate.setdefault(run["candidate"], [])
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads(
            (sweep_dir / run["metrics_path"]).read_text(encoding="utf-8")
        )
        by_candidate[run["candidate"]].append({"seed": run["seed"], "metrics": metrics})

    expected_per_candidate = len(status["seeds"])
    rows = []
    for candidate, entries in by_candidate.items():
        row = {
            "candidate": candidate,
            "successful_runs": len(entries),
            "expected_runs": expected_per_candidate,
            "seeds": [entry["seed"] for entry in entries],
            "trainable_parameters": sorted(
                {entry["metrics"]["trainable_parameters"] for entry in entries}
            ),
            "status": "COMPLETE"
            if len(entries) == expected_per_candidate
            else "PARTIAL",
        }
        for split in ("test", "validation"):
            for metric in SUMMARY_METRICS:
                values = []
                for entry in entries:
                    split_metrics = entry["metrics"]["splits"].get(split)
                    if split_metrics is not None:
                        values.append(float(split_metrics["observed_targets"][metric]))
                if values:
                    if not all(math.isfinite(v) for v in values):
                        row["status"] = "INVALID"
                    row[f"{split}.{metric}.mean"] = statistics.fmean(values)
                    row[f"{split}.{metric}.std"] = (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    )
                    row[f"{split}.{metric}.values"] = values
        rows.append(row)

    summary = {
        "sweep_id": status["sweep_id"],
        "rows": rows,
        "selection": selection,
        "selected_candidates": [],
    }
    metric = selection.get("metric")
    complete = all(row["status"] == "COMPLETE" for row in rows) and len(rows) == len(
        status["candidates"]
    )
    summary["coverage"] = "COMPLETE SWEEP" if complete else "PARTIAL SWEEP"
    if metric:
        key = f"{metric.split('.')[0]}.{metric.split('.')[-1]}.mean"
        eligible = [row for row in rows if row["status"] == "COMPLETE" and key in row]
        if eligible and complete:
            sign = 1.0 if selection["direction"] == "minimize" else -1.0
            best = min(sign * row[key] for row in eligible)
            tolerance = abs(best) * float(selection["tie_tolerance_relative"])
            summary["selected_candidates"] = [
                row["candidate"]
                for row in eligible
                if sign * row[key] - best <= tolerance
            ]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument(
        "--sweeps-root", type=Path, default=PROJECT_DIR / "outdir" / "sweeps"
    )
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    sweep_id = f"{plan['sweep_name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    sweep_dir = (args.sweeps_root / sweep_id).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=False)
    _configure_sweep_logging(sweep_dir)
    (sweep_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    seeds = list(plan["seeds"])
    candidates = plan["candidates"]
    status = {
        "schema_version": 1,
        "sweep_id": sweep_id,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "expected_runs": len(candidates) * len(seeds),
        "candidates": [c["label"] for c in candidates],
        "seeds": seeds,
        "runs": [],
        "summary_path": None,
        "selected_candidates": [],
        "error": None,
    }
    queue = []
    for candidate in candidates:
        for repetition, seed in enumerate(seeds):
            worker_outdir = sweep_dir / candidate["label"] / f"seed{seed}"
            command = [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(REPO_ROOT / "implementation.py"),
                "--paper",
                "QK_LSTM",
                "--config",
                candidate["config"],
                "--seed",
                str(seed),
                "--outdir",
                str(worker_outdir),
                "--sweep-id",
                sweep_id,
                "--candidate",
                candidate["label"],
                "--repetition",
                str(repetition),
                *candidate.get("args", []),
            ]
            entry = {
                "run_id": None,
                "candidate": candidate["label"],
                "seed": seed,
                "repetition": repetition,
                "status": "PLANNED",
                "attempt": 1,
                "command": " ".join(command),
                "run_dir": None,
                "run_log": None,
                "run_status": None,
                "config_path": None,
                "metrics_path": None,
            }
            status["runs"].append(entry)
            queue.append((entry, command, worker_outdir))
    write_json(sweep_dir / "sweep_status.json", status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | max_parallel=%d",
        sweep_id,
        status["expected_runs"],
        args.max_parallel,
    )
    total_epochs = None

    try:
        running: list[tuple[dict, subprocess.Popen, Path, object]] = []
        pending = list(queue)
        while pending or running:
            while pending and len(running) < args.max_parallel:
                entry, command, worker_outdir = pending.pop(0)
                worker_outdir.mkdir(parents=True, exist_ok=True)
                stdout = (worker_outdir / "worker_stdout.txt").open(
                    "w", encoding="utf-8"
                )
                process = subprocess.Popen(
                    command, cwd=REPO_ROOT, stdout=stdout, stderr=subprocess.STDOUT
                )
                entry["status"] = "RUNNING"
                running.append((entry, process, worker_outdir, stdout))
                logger.info(
                    "CANDIDATE_STARTED | candidate=%s | seed=%s | repetition=%s",
                    entry["candidate"],
                    entry["seed"],
                    entry["repetition"],
                )
                write_json(sweep_dir / "sweep_status.json", status)
                logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))
            time.sleep(10)
            changed = False
            for item in list(running):
                entry, process, worker_outdir, stdout = item
                run_dir = _find_run_dir(worker_outdir)
                if run_dir is not None and total_epochs is None:
                    snapshot = run_dir / "config_snapshot.json"
                    if snapshot.exists():
                        total_epochs = int(
                            json.loads(snapshot.read_text())["training"]["epochs"]
                        )
                if total_epochs and entry["candidate"] and run_dir is not None:
                    current = _epochs_completed(run_dir)
                    if current != entry.get("progress", {}).get("current"):
                        entry["progress"] = {
                            "current": current,
                            "total": total_epochs,
                            "unit": "epoch",
                        }
                        changed = True
                if process.poll() is None:
                    continue
                stdout.close()
                running.remove(item)
                changed = True
                run_status = (
                    json.loads((run_dir / "run_status.json").read_text())
                    if run_dir and (run_dir / "run_status.json").exists()
                    else None
                )
                if run_dir is not None:
                    entry.update(
                        run_id=run_dir.name,
                        run_dir=_relative(run_dir, sweep_dir),
                        run_log=_relative(run_dir / "run.log", sweep_dir),
                        run_status=_relative(run_dir / "run_status.json", sweep_dir),
                        config_path=_relative(
                            run_dir / "config_snapshot.json", sweep_dir
                        ),
                        metrics_path=_relative(run_dir / "metrics.json", sweep_dir),
                    )
                if (
                    process.returncode == 0
                    and run_status
                    and run_status["status"] == "COMPLETED"
                ):
                    entry["status"] = "COMPLETED"
                    entry.pop("progress", None)
                    logger.info(
                        "CANDIDATE_COMPLETED | run_id=%s | candidate=%s | seed=%s | metrics=%s",
                        entry["run_id"],
                        entry["candidate"],
                        entry["seed"],
                        entry["metrics_path"],
                    )
                else:
                    entry["status"] = "FAILED"
                    entry["error"] = {
                        "returncode": process.returncode,
                        "run_status": None
                        if run_status is None
                        else run_status.get("error"),
                    }
                    logger.info(
                        "CANDIDATE_FAILED | run_id=%s | candidate=%s | seed=%s | returncode=%s",
                        entry["run_id"],
                        entry["candidate"],
                        entry["seed"],
                        process.returncode,
                    )
            if changed:
                write_json(sweep_dir / "sweep_status.json", status)
                logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))

        summary = build_summary(status, sweep_dir, plan.get("selection", {}))
        summary_path = sweep_dir / "summary.json"
        write_json(summary_path, summary)
        status["summary_path"] = summary_path.name
        status["selected_candidates"] = summary["selected_candidates"]
        status["completed_at"] = utc_now()
        all_completed = all(run["status"] == "COMPLETED" for run in status["runs"])
        any_completed = any(run["status"] == "COMPLETED" for run in status["runs"])
        status["status"] = (
            "COMPLETED" if all_completed else ("PARTIAL" if any_completed else "FAILED")
        )
        if status["status"] == "FAILED":
            status["error"] = {
                "type": "AllCandidatesFailed",
                "message": "No candidate run completed",
            }
        write_json(sweep_dir / "sweep_status.json", status)
        event = {
            "COMPLETED": "SWEEP_COMPLETED",
            "PARTIAL": "SWEEP_PARTIAL",
            "FAILED": "SWEEP_FAILED",
        }[status["status"]]
        if event == "SWEEP_FAILED":
            try:
                raise RuntimeError(status["error"]["message"])
            except RuntimeError:
                logger.exception("SWEEP_FAILED | sweep_id=%s", sweep_id)
            return 1
        logger.info(
            "%s | summary=%s | coverage=%s | selected=%s",
            event,
            summary_path.name,
            summary["coverage"],
            summary["selected_candidates"],
        )
        return 0
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(sweep_dir / "sweep_status.json", status)
        logger.exception("SWEEP_FAILED | sweep_id=%s", sweep_id)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
