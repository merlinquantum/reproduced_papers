"""Sweep coordinator: run one ``implementation.py`` process per planned point.

Usage (from the repository root)::

    python3 papers/virtual_qnn/utils/run_sweep.py papers/virtual_qnn/configs/sweeps/<sweep>.json

The sweep spec (JSON) declares ``sweep_id``, ``base_config``, the candidate
grid (``axes`` for a Cartesian product and/or an explicit ``grid`` list of CLI
overrides), ``repetitions`` (one CLI-override dict per repetition),
``fixed_args``, ``max_workers`` and the predeclared ``plan`` (metric,
direction, aggregation, selection rule). Every planned candidate x repetition
is one run in its own output directory::

    outdir/sweeps/<sweep_id>/
        sweep.log, sweep_status.json, summary.json, summary.csv
        runs/<run_id>/run_<timestamp>/   (run.log, run_status.json, metrics.json, ...)
        runs/<run_id>/worker_attempt<k>.out

Each planned run gets at most two attempts (initial + one automatic retry).
Only this coordinator writes ``sweep.log`` and ``sweep_status.json``; workers
write their own ``run.log``. Retry attempts are logged as
``CANDIDATE_ATTEMPT_FAILED`` / ``CANDIDATE_RETRY_STARTED`` and kept in each
run entry's ``attempts`` list.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_DIR.parent))

from virtual_qnn.lib.experiment_logging import (  # noqa: E402
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)
from virtual_qnn.utils.summarize_sweep import summarize  # noqa: E402

logger = logging.getLogger("virtual_qnn.sweep")
MAX_ATTEMPTS = 2
PROGRESS_EVENT = "TRAIN_ITERATION_COMPLETED"


def paper_code_fingerprint() -> str:
    """SHA-256 over the paper's code, CLI schema, configs and requirements."""
    digest = hashlib.sha256()
    patterns = (
        "lib/*.py",
        "utils/*.py",
        "cli.json",
        "configs/*.json",
        "configs/sweeps/*.json",
        "requirements.txt",
    )
    files = sorted({path for pattern in patterns for path in PROJECT_DIR.glob(pattern)})
    for path in files:
        digest.update(str(path.relative_to(PROJECT_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def expand_candidates(spec: dict) -> list[dict]:
    """Return the ordered list of candidate CLI-override dicts."""
    candidates = [dict(point) for point in spec.get("grid", [])]
    axes = spec.get("axes", {})
    if axes:
        keys = list(axes)
        for values in itertools.product(*(axes[key] for key in keys)):
            candidates.append(dict(zip(keys, values)))
    if not candidates:
        raise ValueError("Sweep spec defines no candidates ('grid' or 'axes')")
    return candidates


def _cli_args(overrides: dict) -> list[str]:
    args: list[str] = []
    for flag, value in overrides.items():
        args += [flag, str(value)]
    return args


def _configure_logging(sweep_dir: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)sZ [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (
        logging.FileHandler(sweep_dir / "sweep.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


class Coordinator:
    """Plan, launch, track and summarize one sweep."""

    def __init__(self, spec: dict, spec_path: Path, sweep_root: Path) -> None:
        self.spec = spec
        self.spec_path = spec_path
        self.sweep_id = spec["sweep_id"]
        self.sweep_dir = sweep_root / self.sweep_id
        self.lock = threading.Lock()
        self.candidates = expand_candidates(spec)
        self.repetitions = spec["repetitions"]
        self.runs: list[dict] = []
        for c_index, candidate in enumerate(self.candidates):
            for r_index, repetition in enumerate(self.repetitions):
                run_id = f"{self.sweep_id}__c{c_index:03d}_r{r_index:02d}"
                self.runs.append(
                    {
                        "run_id": run_id,
                        "candidate": {"planned_overrides": candidate},
                        "grid_point": candidate,
                        "repetition_overrides": repetition,
                        "seed": int(repetition["--seed"]),
                        "repetition": r_index,
                        "status": "PLANNED",
                        "run_dir": None,
                        "run_log": None,
                        "run_status": None,
                        "config_path": None,
                        "metrics_path": None,
                        "attempts": [],
                    }
                )
        self.status = {
            "schema_version": 1,
            "sweep_id": self.sweep_id,
            "status": "RUNNING",
            "started_at": None,
            "completed_at": None,
            "code": {
                **git_state(REPO_ROOT),
                "paper_code_sha256": paper_code_fingerprint(),
            },
            "spec_path": str(spec_path.relative_to(REPO_ROOT))
            if spec_path.is_relative_to(REPO_ROOT)
            else str(spec_path),
            "spec": spec,
            "expected_runs": len(self.runs),
            "runs": self.runs,
            "summary_path": None,
            "selected_candidates": [],
            "error": None,
        }

    def _write_status(self, log_progress: bool = True) -> None:
        with self.lock:
            write_json(self.sweep_dir / "sweep_status.json", self.status)
            if log_progress and self.status["status"] == "RUNNING":
                logger.info("SWEEP_PROGRESS | %s", sweep_progress(self.status))

    def _command(self, run: dict, outdir: Path) -> list[str]:
        command = [
            sys.executable,
            str(REPO_ROOT / "implementation.py"),
            "--paper",
            "virtual_qnn",
            "--config",
            self.spec["base_config"],
            "--outdir",
            str(outdir),
            "--sweep-id",
            self.sweep_id,
            "--run-id",
            run["run_id"],
            "--repetition",
            str(run["repetition"]),
        ]
        command += _cli_args(self.spec.get("fixed_args", {}))
        command += _cli_args(run["grid_point"])
        command += _cli_args(run["repetition_overrides"])
        return command

    def _track_progress(
        self, run: dict, outdir: Path, process: subprocess.Popen
    ) -> None:
        total = self.spec.get("progress_total")
        while True:
            try:
                process.wait(timeout=15)
                return
            except subprocess.TimeoutExpired:
                pass
            logs = list(outdir.glob("run_*/run.log"))
            if total and logs:
                current = (
                    logs[0]
                    .read_text(encoding="utf-8", errors="replace")
                    .count(PROGRESS_EVENT)
                )
                with self.lock:
                    run["progress"] = {
                        "current": current,
                        "total": total,
                        "unit": "iteration",
                    }
                self._write_status()

    def _execute(self, run: dict) -> None:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            outdir = self.sweep_dir / "runs" / run["run_id"] / f"attempt{attempt}"
            outdir.mkdir(parents=True, exist_ok=True)
            command = self._command(run, outdir)
            with self.lock:
                run["status"] = "RUNNING"
            if attempt == 1:
                logger.info(
                    "CANDIDATE_STARTED | run_id=%s | candidate=%s | seed=%s",
                    run["run_id"],
                    run["grid_point"],
                    run["seed"],
                )
            else:
                logger.info(
                    "CANDIDATE_RETRY_STARTED | run_id=%s | attempt=%d",
                    run["run_id"],
                    attempt,
                )
            self._write_status()
            environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            started = time.time()
            with (outdir / "worker.out").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                self._track_progress(run, outdir, process)
            elapsed = time.time() - started
            run_dirs = sorted(outdir.glob("run_*"))
            record = {
                "attempt": attempt,
                "command": " ".join(command),
                "exit_code": process.returncode,
                "elapsed_s": round(elapsed, 1),
                "run_dir": str(run_dirs[0].relative_to(self.sweep_dir))
                if run_dirs
                else None,
            }
            outcome = self._check_run(run_dirs)
            record["outcome"] = outcome
            with self.lock:
                run["attempts"].append(record)
            if outcome == "COMPLETED":
                run_dir = run_dirs[0]
                run_status = json.loads((run_dir / "run_status.json").read_text())
                with self.lock:
                    run.update(
                        status="COMPLETED",
                        candidate=run_status["candidate"],
                        run_dir=str(run_dir.relative_to(self.sweep_dir)),
                        run_log=str((run_dir / "run.log").relative_to(self.sweep_dir)),
                        run_status=str(
                            (run_dir / "run_status.json").relative_to(self.sweep_dir)
                        ),
                        config_path=str(
                            (run_dir / run_status["config_path"]).relative_to(
                                self.sweep_dir
                            )
                        ),
                        metrics_path=str(
                            (run_dir / run_status["metrics_path"]).relative_to(
                                self.sweep_dir
                            )
                        ),
                    )
                    run.pop("progress", None)
                logger.info(
                    "CANDIDATE_COMPLETED | run_id=%s | metrics=%s | elapsed_s=%.1f",
                    run["run_id"],
                    run["metrics_path"],
                    elapsed,
                )
                self._write_status()
                return
            if attempt < MAX_ATTEMPTS:
                logger.warning(
                    "CANDIDATE_ATTEMPT_FAILED | run_id=%s | attempt=%d | outcome=%s | exit=%s",
                    run["run_id"],
                    attempt,
                    outcome,
                    process.returncode,
                )
        with self.lock:
            run["status"] = "FAILED"
            run.pop("progress", None)
        logger.error(
            "CANDIDATE_FAILED | run_id=%s | attempts=%d", run["run_id"], MAX_ATTEMPTS
        )
        self._write_status()

    @staticmethod
    def _check_run(run_dirs: list[Path]) -> str:
        if len(run_dirs) != 1:
            return f"INVALID (found {len(run_dirs)} run directories)"
        status_path = run_dirs[0] / "run_status.json"
        if not status_path.is_file():
            return "INVALID (no run_status.json)"
        run_status = json.loads(status_path.read_text())
        if run_status.get("status") != "COMPLETED":
            return f"FAILED (run status {run_status.get('status')})"
        try:
            json.loads((run_dirs[0] / run_status["metrics_path"]).read_text())
        except (OSError, ValueError) as exc:
            return f"INVALID (metrics unreadable: {exc})"
        return "COMPLETED"

    def run(self) -> int:
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        if (self.sweep_dir / "sweep_status.json").exists():
            raise FileExistsError(f"Sweep directory already used: {self.sweep_dir}")
        _configure_logging(self.sweep_dir)
        self.status["started_at"] = utc_now()
        self._write_status(log_progress=False)
        logger.info(
            "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | candidates=%d | repetitions=%d | workers=%d | commit=%s | dirty=%s | paper_code_sha256=%s",
            self.sweep_id,
            len(self.runs),
            len(self.candidates),
            len(self.repetitions),
            self.spec["max_workers"],
            self.status["code"]["commit"],
            self.status["code"]["dirty"],
            self.status["code"]["paper_code_sha256"],
        )
        try:
            with ThreadPoolExecutor(max_workers=int(self.spec["max_workers"])) as pool:
                list(pool.map(self._execute, self.runs))
            summary_path = summarize(self.sweep_dir / "sweep_status.json", self.status)
            self.status["summary_path"] = str(summary_path.relative_to(self.sweep_dir))
            completed = sum(run["status"] == "COMPLETED" for run in self.runs)
            fingerprint_end = paper_code_fingerprint()
            if fingerprint_end != self.status["code"]["paper_code_sha256"]:
                logger.warning(
                    "CODE_CHANGED_DURING_SWEEP | start=%s | end=%s",
                    self.status["code"]["paper_code_sha256"],
                    fingerprint_end,
                )
                self.status["code"]["paper_code_sha256_end"] = fingerprint_end
            self.status["completed_at"] = utc_now()
            if completed == len(self.runs):
                self.status["status"] = "COMPLETED"
                self._write_status(log_progress=False)
                logger.info(
                    "SWEEP_COMPLETED | summary=%s | selected=none (descriptive sweep) | completed=%d/%d",
                    self.status["summary_path"],
                    completed,
                    len(self.runs),
                )
                return 0
            self.status["status"] = "PARTIAL" if completed else "FAILED"
            if not completed:
                self.status["error"] = {
                    "type": "AllRunsFailed",
                    "message": "No planned run completed",
                }
                self._write_status(log_progress=False)
                try:
                    raise RuntimeError("No planned run completed")
                except RuntimeError:
                    logger.exception("SWEEP_FAILED | sweep_id=%s", self.sweep_id)
                return 1
            self._write_status(log_progress=False)
            logger.warning(
                "SWEEP_PARTIAL | summary=%s | completed=%d/%d",
                self.status["summary_path"],
                completed,
                len(self.runs),
            )
            return 1
        except Exception as exc:
            self.status.update(
                status="FAILED",
                completed_at=utc_now(),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            self._write_status(log_progress=False)
            logger.exception("SWEEP_FAILED | sweep_id=%s", self.sweep_id)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "spec", type=Path, help="Sweep spec JSON (configs/sweeps/<sweep>.json)"
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=PROJECT_DIR / "outdir" / "sweeps",
        help="Parent directory of sweep folders",
    )
    args = parser.parse_args(argv)
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    return Coordinator(spec, spec_path, args.sweep_root.resolve()).run()


if __name__ == "__main__":
    raise SystemExit(main())
