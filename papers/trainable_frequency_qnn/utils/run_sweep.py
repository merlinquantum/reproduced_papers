#!/usr/bin/env python3
"""Run a predeclared sweep of (candidate, seed) runs through the shared runtime.

A sweep spec (JSON, usually under ``configs/sweeps/``) lists candidates, each an
experiment config plus optional dotted-path overrides, the seeds, the parallel
worker count and the predeclared selection rule. Every planned run is one
``implementation.py`` subprocess with its own ``--outdir``, so workers never
share a log file. This coordinator alone writes ``sweep.log`` and
``sweep_status.json`` (experiment-logging contract) and, at the end,
``summary.json`` / ``summary.csv`` built from the runs' ``metrics.json``.

Each planned run gets at most two attempts (initial + one automatic retry).

Example
-------
From the repository root::

    python papers/trainable_frequency_qnn/utils/run_sweep.py \\
        --spec configs/sweeps/e2_fig2b.json --sweep-dir outdir/sweeps/e2_fig2b \\
        -- --device cpu

Relative ``--spec`` and ``--sweep-dir`` paths resolve against the paper folder.
Arguments after ``--`` are appended to every run command.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PAPER_DIR.parents[1]
for import_root in (REPOSITORY_ROOT, PAPER_DIR):
    sys.path.insert(0, str(import_root))

from lib.experiment_logging import git_state, sweep_progress, utc_now, write_json  # noqa: E402

logger = logging.getLogger("sweep")
SCHEMA_VERSION = 1
MAX_ATTEMPTS = 2
PROGRESS_SECONDS = 15
ITERATION_PATTERN = re.compile(r"TRAIN_ITERATION_COMPLETED \| iteration=(\d+)/(\d+)")


def configure_sweep_logging(sweep_dir: Path) -> None:
    """Attach ``sweep.log`` and stderr handlers to the coordinator logger only."""
    formatter = logging.Formatter(
        "%(asctime)sZ [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (
        logging.FileHandler(sweep_dir / "sweep.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def set_by_path(target: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``target["a"]["b"] = value`` for ``dotted = "a.b"``."""
    keys = dotted.split(".")
    node = target
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def get_by_path(source: dict[str, Any], dotted: str) -> Any:
    """Return ``source["a"]["b"]`` for ``dotted = "a.b"``."""
    node: Any = source
    for key in dotted.split("."):
        node = node[key]
    return node


def write_candidate_config(
    spec: dict[str, Any], candidate: dict[str, Any], sweep_dir: Path
) -> Path:
    """Write the candidate's config (base config + overrides + sweep identity)."""
    config = json.loads((PAPER_DIR / candidate["config"]).read_text(encoding="utf-8"))
    for dotted, value in candidate.get("overrides", {}).items():
        set_by_path(config, dotted, value)
    config["sweep"] = {
        "sweep_id": spec["sweep_id"],
        "candidate": candidate["name"],
        "candidate_config": candidate["config"],
        "candidate_overrides": candidate.get("overrides", {}),
    }
    path = sweep_dir / "candidate_configs" / f"{candidate['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


class Sweep:
    """Coordinator state: the planned runs and their status file."""

    def __init__(
        self, spec: dict[str, Any], sweep_dir: Path, extra_args: list[str]
    ) -> None:
        self.spec = spec
        self.sweep_dir = sweep_dir
        self.extra_args = extra_args
        self.lock = threading.Lock()
        self.status_path = sweep_dir / "sweep_status.json"
        workers = int(spec["max_workers"])
        self.child_threads = max(1, (os.cpu_count() or 1) // workers)
        self.config_paths = {
            c["name"]: write_candidate_config(spec, c, sweep_dir)
            for c in spec["candidates"]
        }
        self.status: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "sweep_id": spec["sweep_id"],
            "status": "RUNNING",
            "started_at": utc_now(),
            "completed_at": None,
            "code": git_state(REPOSITORY_ROOT),
            "expected_runs": len(spec["candidates"]) * len(spec["seeds"]),
            "runs": [],
            "summary_path": None,
            "selected_candidates": [],
            "error": None,
        }
        for candidate in spec["candidates"]:
            for repetition, seed in enumerate(spec["seeds"]):
                self.status["runs"].append(
                    {
                        "run_id": None,
                        "candidate": candidate["name"],
                        "seed": seed,
                        "repetition": repetition,
                        "status": "PLANNED",
                        "attempts": 0,
                        "run_dir": None,
                        "run_log": None,
                        "run_status": None,
                        "config_path": None,
                        "metrics_path": None,
                    }
                )

    def save(self) -> None:
        with self.lock:
            write_json(self.status_path, self.status)
            if self.status["status"] == "RUNNING":
                logger.info("SWEEP_PROGRESS | %s", sweep_progress(self.status))

    def _relative(self, path: Path) -> str:
        return os.path.relpath(path, self.sweep_dir)

    def _command(self, entry: dict[str, Any], outdir: Path) -> list[str]:
        repetition_config = (
            self.sweep_dir
            / "candidate_configs"
            / f"{entry['candidate']}__rep{entry['repetition']}.json"
        )
        config = json.loads(
            self.config_paths[entry["candidate"]].read_text(encoding="utf-8")
        )
        config["sweep"]["repetition"] = entry["repetition"]
        repetition_config.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        return [
            sys.executable,
            str(REPOSITORY_ROOT / "implementation.py"),
            "--paper",
            PAPER_DIR.name,
            "--config",
            str(repetition_config),
            "--seed",
            str(entry["seed"]),
            "--outdir",
            str(outdir),
            *self.extra_args,
        ]

    def run_one(self, entry: dict[str, Any]) -> None:
        """Run one planned run with at most ``MAX_ATTEMPTS`` attempts."""
        environment = dict(
            os.environ,
            OMP_NUM_THREADS=str(self.child_threads),
            MKL_NUM_THREADS=str(self.child_threads),
        )
        while entry["attempts"] < MAX_ATTEMPTS:
            entry["attempts"] += 1
            outdir = (
                self.sweep_dir
                / entry["candidate"]
                / f"seed_{entry['seed']}"
                / f"attempt_{entry['attempts']}"
            )
            outdir.mkdir(parents=True, exist_ok=True)
            command = self._command(entry, outdir)
            entry.update(status="RUNNING", progress=None)
            entry["command"] = " ".join(command)
            entry["_outdir"] = str(outdir)
            # Validators count one CANDIDATE_STARTED and one terminal event per
            # planned run, so a retry is logged under its own event names.
            logger.info(
                "%s | candidate=%s | seed=%s | attempt=%d | command=%s",
                "CANDIDATE_STARTED"
                if entry["attempts"] == 1
                else "CANDIDATE_RETRY_STARTED",
                entry["candidate"],
                entry["seed"],
                entry["attempts"],
                entry["command"],
            )
            self.save()
            with (outdir / "stdout.log").open("w", encoding="utf-8") as stdout:
                completed = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            run_dirs = sorted(outdir.glob("run_*"))
            run_status = None
            if run_dirs:
                run_dir = run_dirs[-1]
                entry.update(
                    run_id=run_dir.name,
                    run_dir=self._relative(run_dir),
                    run_log=self._relative(run_dir / "run.log"),
                    run_status=self._relative(run_dir / "run_status.json"),
                    config_path=self._relative(run_dir / "config_snapshot.json"),
                )
                status_file = run_dir / "run_status.json"
                if status_file.exists():
                    run_status = json.loads(status_file.read_text(encoding="utf-8"))
            ok = (
                completed.returncode == 0
                and run_status is not None
                and run_status["status"] == "COMPLETED"
            )
            if ok:
                entry.update(
                    status="COMPLETED",
                    metrics_path=self._relative(
                        self.sweep_dir / entry["run_dir"] / "metrics.json"
                    ),
                )
                entry.pop("progress", None)
                logger.info(
                    "CANDIDATE_COMPLETED | candidate=%s | seed=%s | run_id=%s | metrics=%s",
                    entry["candidate"],
                    entry["seed"],
                    entry["run_id"],
                    entry["metrics_path"],
                )
                self.save()
                return
            final_attempt = entry["attempts"] >= MAX_ATTEMPTS
            entry.update(
                status="FAILED" if final_attempt else "RUNNING", metrics_path=None
            )
            entry.pop("progress", None)
            logger.info(
                "%s | candidate=%s | seed=%s | attempt=%d | exit_code=%d | run_status=%s",
                "CANDIDATE_FAILED" if final_attempt else "CANDIDATE_ATTEMPT_FAILED",
                entry["candidate"],
                entry["seed"],
                entry["attempts"],
                completed.returncode,
                None if run_status is None else run_status["status"],
            )
            self.save()

    def update_progress(self) -> None:
        """Refresh ``progress`` of running entries from their run logs (progress only)."""
        changed = False
        for entry in self.status["runs"]:
            if entry["status"] != "RUNNING" or not entry.get("_outdir"):
                continue
            logs = sorted(Path(entry["_outdir"]).glob("run_*/run.log"))
            if not logs:
                continue
            matches = ITERATION_PATTERN.findall(
                logs[-1].read_text(encoding="utf-8", errors="replace")[-4000:]
            )
            if matches:
                current, total = map(int, matches[-1])
                entry["progress"] = {
                    "current": current,
                    "total": total,
                    "unit": "iteration",
                }
                changed = True
        if changed:
            self.save()

    def summarize(self) -> tuple[Path, list[str]]:
        """Build the full result table from metrics.json files and apply the selection rule."""
        selection = self.spec["selection"]
        rows = []
        for candidate in self.spec["candidates"]:
            name = candidate["name"]
            entries = [e for e in self.status["runs"] if e["candidate"] == name]
            values = []
            for entry in entries:
                if entry["status"] != "COMPLETED":
                    continue
                metrics = json.loads(
                    (self.sweep_dir / entry["metrics_path"]).read_text(encoding="utf-8")
                )
                value = float(get_by_path(metrics, selection["metric"]))
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite {selection['metric']} for {name} seed {entry['seed']}."
                    )
                values.append(value)
            complete = len(values) == len(entries)
            rows.append(
                {
                    "candidate": name,
                    "successful_runs": len(values),
                    "expected_runs": len(entries),
                    "mean": statistics.fmean(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else None,
                    "median": statistics.median(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "status": "COMPLETE" if complete else "PARTIAL",
                    "values": values,
                }
            )
        sign = 1 if selection["direction"] == "minimize" else -1
        ranked = [r for r in rows if r["mean"] is not None]
        selected: list[str] = []
        if ranked:
            best = min(sign * r["mean"] for r in ranked)
            tolerance = abs(best) * float(selection["tie_tolerance_rel"])
            selected = [
                r["candidate"] for r in ranked if sign * r["mean"] - best <= tolerance
            ]
        summary = {
            "sweep_id": self.spec["sweep_id"],
            "selection": selection,
            "coverage": "COMPLETE SWEEP"
            if all(r["status"] == "COMPLETE" for r in rows)
            else "PARTIAL SWEEP",
            "selected_candidates": selected,
            "selected_label": "selected candidate(s) under the declared rule"
            if all(r["status"] == "COMPLETE" for r in rows)
            else "lowest/highest observed aggregate among completed runs (partial)",
            "rows": rows,
        }
        summary_path = self.sweep_dir / "summary.json"
        write_json(summary_path, summary)
        with (self.sweep_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=[k for k in rows[0] if k != "values"]
            )
            writer.writeheader()
            writer.writerows(
                [{k: v for k, v in r.items() if k != "values"} for r in rows]
            )
        return summary_path, selected


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extra_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra_args = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--spec",
        required=True,
        type=Path,
        help="Sweep spec JSON (relative to the paper folder).",
    )
    parser.add_argument(
        "--sweep-dir",
        required=True,
        type=Path,
        help="Output folder (relative to the paper folder).",
    )
    args = parser.parse_args(argv)

    spec_path = args.spec if args.spec.is_absolute() else PAPER_DIR / args.spec
    sweep_dir = (
        args.sweep_dir if args.sweep_dir.is_absolute() else PAPER_DIR / args.sweep_dir
    )
    if (sweep_dir / "sweep_status.json").exists():
        raise SystemExit(
            f"{sweep_dir} already holds a sweep; choose a new --sweep-dir."
        )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    configure_sweep_logging(sweep_dir)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    (sweep_dir / "sweep_spec.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8"
    )

    sweep = Sweep(spec, sweep_dir, extra_args)
    sweep.save()
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | workers=%d | threads_per_run=%d | extra_args=%s",
        spec["sweep_id"],
        sweep.status["expected_runs"],
        spec["max_workers"],
        sweep.child_threads,
        extra_args,
    )
    stop_monitor = threading.Event()

    def monitor() -> None:
        while not stop_monitor.wait(PROGRESS_SECONDS):
            sweep.update_progress()

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=int(spec["max_workers"])) as executor:
            list(executor.map(sweep.run_one, sweep.status["runs"]))
        stop_monitor.set()
        for entry in sweep.status["runs"]:
            entry.pop("_outdir", None)
        summary_path, selected = sweep.summarize()
        coverage_complete = all(
            e["status"] == "COMPLETED" for e in sweep.status["runs"]
        )
        sweep.status.update(
            status="COMPLETED" if coverage_complete else "PARTIAL",
            completed_at=utc_now(),
            summary_path=os.path.relpath(summary_path, sweep_dir),
            selected_candidates=selected,
        )
        sweep.save()
        if coverage_complete:
            logger.info(
                "SWEEP_COMPLETED | summary=%s | selected=%s",
                summary_path.name,
                selected,
            )
            return 0
        logger.info(
            "SWEEP_PARTIAL | summary=%s | selected=%s", summary_path.name, selected
        )
        return 1
    except Exception as exc:
        stop_monitor.set()
        sweep.status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        sweep.save()
        logger.exception("SWEEP_FAILED | sweep_id=%s", spec["sweep_id"])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
