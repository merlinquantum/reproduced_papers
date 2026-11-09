"""Sweep coordinator: launch planned PQKD runs in parallel and summarise them.

A sweep spec (JSON) fixes, before launch, every candidate, the seeds, the
selection metric, its direction, the tie tolerance and the tie-break rule
(scientific-validity policy, "Sweep Integrity and Selection"). Each planned
(candidate, seed) run gets its own output directory, so parallel runs never
share a run directory or a log file. Only this coordinator writes
``sweep.log`` and ``sweep_status.json``.

Usage (from the repository root)::

    python papers/photonic_QKD/utils/run_sweep.py papers/photonic_QKD/sweeps/<spec>.json
    python papers/photonic_QKD/utils/run_sweep.py --summarize <sweep_dir>
    python papers/photonic_QKD/utils/run_sweep.py --resume <stopped sweep_dir>

Stopping: ``utils/stop_sweep.py`` sends SIGTERM to the PID in
``coordinator.pid``. The coordinator forwards SIGTERM to its runs (each ends
FAILED with its last epoch ``checkpoint.pt`` kept), drops runs that never
started from the run list (listed under ``not_started``), and closes the
sweep as PARTIAL with a summary, so it still passes sweep validation.
``--resume`` then starts a completion sweep for the unfinished
(candidate, seed) pairs, passing ``--resume-from`` to runs with a checkpoint.

Spec keys
---------
sweep_id : str
description : str
fixed_args : list[str]
    Arguments passed to every run after ``implementation.py --paper photonic_QKD``.
candidates : list[{"name": str, "args": list[str]}]
    In tie-break priority order when ``tie_break`` is ``"spec_order"``.
seeds : list[int]
max_parallel, threads_per_run : int
run_order : str, optional
    ``candidate_major`` (default) or ``seed_major``; seed-major interleaves
    candidates so memory-heavy ones do not all run at once.
teacher_sweep : str | null
    Completed teacher sweep directory; its seed-matched run is passed as
    ``--teacher-path``.
selection : {"metric", "direction", "split", "checkpoint", "aggregation",
             "tie_tolerance", "tie_break"}
report_metrics : list[str]
    Dotted ``metrics.json`` paths shown in the summary table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parents[1]
sys.path[:0] = [str(PAPER_DIR), str(REPO_ROOT)]

from lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)

logger = logging.getLogger("pqkd.sweep")
SCHEMA_VERSION = 1
POLL_SECONDS = 30
STOP_GRACE_SECONDS = 300


def _dotted(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _run_label(candidate: str, seed: int) -> str:
    return f"{candidate}__seed{seed}"


def _find_run_dir(outdir: Path) -> Path | None:
    runs = sorted(outdir.glob("run_*"))
    return runs[-1] if runs else None


def _teacher_runs(teacher_sweep: Path) -> dict[int, Path]:
    status = json.loads((teacher_sweep / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise RuntimeError(
            f"Teacher sweep {teacher_sweep} is {status['status']}, not COMPLETED"
        )
    return {
        int(run["seed"]): (teacher_sweep / run["run_dir"]).resolve()
        for run in status["runs"]
        if run["status"] == "COMPLETED"
    }


class SweepCoordinator:
    """Launch, track and summarise one sweep."""

    def __init__(self, spec: dict[str, Any], sweep_dir: Path) -> None:
        self.spec = spec
        self.sweep_dir = sweep_dir
        self.teachers = (
            _teacher_runs((REPO_ROOT / spec["teacher_sweep"]).resolve())
            if spec.get("teacher_sweep")
            else {}
        )
        self.runs: list[dict[str, Any]] = []
        order = spec.get("run_order", "candidate_major")
        if order not in ("candidate_major", "seed_major"):
            raise ValueError(
                f"run_order must be candidate_major or seed_major, got {order!r}"
            )
        pairs = (
            [
                (candidate, seed)
                for candidate in spec["candidates"]
                for seed in spec["seeds"]
            ]
            if order == "candidate_major"
            else [
                (candidate, seed)
                for seed in spec["seeds"]
                for candidate in spec["candidates"]
            ]
        )
        if spec.get("only_runs"):
            # Completion sweeps re-plan only the (candidate, seed) pairs an
            # interrupted sweep did not finish; the spec records which and why.
            wanted = {(name, int(seed)) for name, seed in spec["only_runs"]}
            known = {(candidate["name"], seed) for candidate, seed in pairs}
            if not wanted <= known:
                raise ValueError(
                    f"only_runs not in candidates x seeds: {sorted(wanted - known)}"
                )
            pairs = [
                (candidate, seed)
                for candidate, seed in pairs
                if (candidate["name"], seed) in wanted
            ]
        for candidate, seed in pairs:
            self.runs.append(
                {
                    "run_id": _run_label(candidate["name"], seed),
                    "candidate": candidate["name"],
                    "seed": seed,
                    "repetition": spec["seeds"].index(seed),
                    "status": "PLANNED",
                    "run_dir": None,
                    "run_log": None,
                    "run_status": None,
                    "config_path": None,
                    "metrics_path": None,
                    "command": self._command(candidate, seed),
                }
            )
        self.status: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "sweep_id": spec["sweep_id"],
            "status": "RUNNING",
            "started_at": utc_now(),
            "completed_at": None,
            "code": git_state(REPO_ROOT),
            "expected_runs": len(self.runs),
            "runs": self.runs,
            "summary_path": None,
            "selected_candidates": [],
            "selection": spec["selection"],
            "error": None,
        }

    def _outdir(self, label: str) -> Path:
        return self.sweep_dir / "runs" / label

    def _command(self, candidate: dict[str, Any], seed: int) -> list[str]:
        command = [
            sys.executable,
            "implementation.py",
            "--paper",
            "photonic_QKD",
            *self.spec["fixed_args"],
            *candidate["args"],
            "--seed",
            str(seed),
            "--outdir",
            str(self._outdir(_run_label(candidate["name"], seed))),
            "--sweep-id",
            self.spec["sweep_id"],
            "--run-label",
            _run_label(candidate["name"], seed),
            "--candidate-name",
            candidate["name"],
            "--repetition",
            str(self.spec["seeds"].index(seed)),
        ]
        if self.spec.get("teacher_sweep"):
            if seed not in self.teachers:
                raise RuntimeError(f"No completed teacher for seed {seed}")
            command += ["--teacher-path", str(self.teachers[seed])]
        resume_from = self.spec.get("resume_from", {}).get(
            _run_label(candidate["name"], seed)
        )
        if resume_from:
            command += ["--resume-from", resume_from]
        return command

    def _write_status(self) -> None:
        write_json(self.sweep_dir / "sweep_status.json", self.status)
        if self.status["status"] == "RUNNING":
            logger.info("SWEEP_PROGRESS | %s", sweep_progress(self.status))

    def _refresh_paths(self, run: dict[str, Any]) -> Path | None:
        run_dir = _find_run_dir(self._outdir(run["run_id"]))
        if run_dir is None:
            return None
        relative = os.path.relpath(run_dir, self.sweep_dir)
        run.update(
            run_dir=relative,
            run_log=f"{relative}/run.log",
            run_status=f"{relative}/run_status.json",
            config_path=f"{relative}/config_snapshot.json",
            metrics_path=f"{relative}/metrics.json",
        )
        return run_dir

    def _update_progress(self, run: dict[str, Any]) -> None:
        run_dir = self._refresh_paths(run)
        if run_dir is None or not (run_dir / "config_snapshot.json").exists():
            return
        epochs = json.loads((run_dir / "config_snapshot.json").read_text())["training"][
            "epochs"
        ]
        log_path = run_dir / "run.log"
        done = (
            log_path.read_text(encoding="utf-8").count("TRAIN_EPOCH_COMPLETED")
            if log_path.exists()
            else 0
        )
        run["progress"] = {"current": done, "total": int(epochs), "unit": "epoch"}

    def _finish(self, run: dict[str, Any], return_code: int) -> None:
        run_dir = self._refresh_paths(run)
        run_status = (
            json.loads((run_dir / "run_status.json").read_text())
            if run_dir is not None and (run_dir / "run_status.json").exists()
            else None
        )
        completed = (
            return_code == 0
            and run_status is not None
            and run_status["status"] == "COMPLETED"
            and (run_dir / "metrics.json").exists()
        )
        run["status"] = "COMPLETED" if completed else "FAILED"
        run["return_code"] = return_code
        run.pop("progress", None)
        if completed:
            logger.info(
                "CANDIDATE_COMPLETED | run_id=%s | metrics=%s",
                run["run_id"],
                run["metrics_path"],
            )
        else:
            logger.error(
                "CANDIDATE_FAILED | run_id=%s | return_code=%s | run_status=%s | console=%s",
                run["run_id"],
                return_code,
                None if run_status is None else run_status.get("status"),
                os.path.relpath(
                    self._outdir(run["run_id"]) / "console.log", self.sweep_dir
                ),
            )

    def _request_stop(self, signal_number: int, _frame: Any) -> None:
        if not self.stop_requested:
            logger.warning("SWEEP_STOP_REQUESTED | signal=%d", signal_number)
        self.stop_requested = True

    def _stop(
        self, active: dict[str, tuple[subprocess.Popen, Any]], pending: list
    ) -> None:
        """Stop running candidates (each keeps its checkpoint) and close the sweep as PARTIAL."""
        for process, _console in active.values():
            process.send_signal(signal.SIGTERM)
        for run in self.runs:
            if run["run_id"] not in active:
                continue
            process, console = active.pop(run["run_id"])
            try:
                return_code = process.wait(timeout=STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
            console.close()
            self._finish(run, return_code)
            if run["status"] == "FAILED":
                run["failure_reason"] = (
                    "stopped by user (SIGTERM); resume with run_sweep.py --resume"
                )
        never_started = [run["run_id"] for run in pending]
        self.runs[:] = [run for run in self.runs if run["run_id"] not in never_started]
        self.status.update(not_started=never_started, stopped=True)
        logger.warning("SWEEP_STOPPED | not_started=%s", never_started)

    def run(self) -> None:
        (self.sweep_dir / "coordinator.pid").write_text(
            str(os.getpid()), encoding="utf-8"
        )
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)
        logger.info(
            "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | max_parallel=%d | threads_per_run=%d | commit=%s | dirty=%s",
            self.spec["sweep_id"],
            len(self.runs),
            self.spec["max_parallel"],
            self.spec["threads_per_run"],
            self.status["code"]["commit"],
            self.status["code"]["dirty"],
        )
        self._write_status()
        environment = dict(os.environ)
        threads = str(self.spec["threads_per_run"])
        environment.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
        pending = list(self.runs)
        active: dict[str, tuple[subprocess.Popen, Any]] = {}
        try:
            while (pending or active) and not self.stop_requested:
                while (
                    pending
                    and len(active) < int(self.spec["max_parallel"])
                    and not self.stop_requested
                ):
                    run = pending.pop(0)
                    outdir = self._outdir(run["run_id"])
                    outdir.mkdir(parents=True, exist_ok=True)
                    console = (outdir / "console.log").open("w", encoding="utf-8")
                    process = subprocess.Popen(
                        run["command"],
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=console,
                        stderr=subprocess.STDOUT,
                    )
                    active[run["run_id"]] = (process, console)
                    run["status"] = "RUNNING"
                    logger.info(
                        "CANDIDATE_STARTED | run_id=%s | candidate=%s | seed=%d",
                        run["run_id"],
                        run["candidate"],
                        run["seed"],
                    )
                    self._write_status()
                    time.sleep(
                        2
                    )  # keep runtime timestamps of parallel launches distinct
                for _ in range(POLL_SECONDS):
                    if self.stop_requested:
                        break
                    time.sleep(1)
                for run in self.runs:
                    if run["run_id"] not in active:
                        continue
                    process, console = active[run["run_id"]]
                    return_code = process.poll()
                    if return_code is None:
                        self._update_progress(run)
                        continue
                    console.close()
                    del active[run["run_id"]]
                    self._finish(run, return_code)
                self._write_status()
            if self.stop_requested:
                self._stop(active, pending)
        except BaseException as exc:
            for process, console in active.values():
                process.terminate()
                console.close()
            self.status.update(
                status="FAILED",
                completed_at=utc_now(),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            write_json(self.sweep_dir / "sweep_status.json", self.status)
            logger.exception("SWEEP_FAILED | sweep_id=%s", self.spec["sweep_id"])
            raise
        summarize(self.sweep_dir, self.spec, self.status, stopped=self.stop_requested)


def _aggregate(values: list[float]) -> tuple[float, float | None]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def summarize(
    sweep_dir: Path, spec: dict[str, Any], status: dict[str, Any], stopped: bool = False
) -> None:
    """Build the result table from ``metrics.json`` files and apply the declared selection rule.

    A user-stopped sweep is always PARTIAL (never FAILED), even with no
    completed run, because nothing failed.
    """
    selection = spec["selection"]
    metric = selection["metric"]
    report_metrics = spec.get("report_metrics", [metric])
    rows = []
    invalid: list[str] = []
    for candidate in spec["candidates"]:
        name = candidate["name"]
        runs = [run for run in status["runs"] if run["candidate"] == name]
        values: dict[str, list[float]] = {key: [] for key in report_metrics}
        seeds_ok = []
        for run in runs:
            if run["status"] != "COMPLETED":
                continue
            metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
            row_values = {key: _dotted(metrics, key) for key in report_metrics}
            primary = _dotted(metrics, metric)
            if not isinstance(primary, (int, float)) or not math.isfinite(primary):
                invalid.append(run["run_id"])
                run["status"] = "FAILED"
                run["failure_reason"] = (
                    f"selection metric {metric} missing or non-finite"
                )
                continue
            seeds_ok.append(run["seed"])
            for key, value in row_values.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values[key].append(float(value))
        expected = len(runs)
        row = {
            "candidate": name,
            "successful_runs": len(seeds_ok),
            "expected_runs": expected,
            "seeds": seeds_ok,
            "status": "COMPLETE" if len(seeds_ok) == expected else "PARTIAL",
        }
        for key in report_metrics:
            if values[key]:
                mean, std = _aggregate(values[key])
                row[key] = {
                    "mean": mean,
                    "std": std,
                    "n": len(values[key]),
                    "values": values[key],
                }
            else:
                row[key] = None
        rows.append(row)

    complete = all(row["status"] == "COMPLETE" for row in rows) and not invalid
    eligible = [row for row in rows if row.get(metric) is not None]
    selected: list[str] = []
    if eligible:
        sign = 1 if selection["direction"] == "maximize" else -1
        best = max(sign * row[metric]["mean"] for row in eligible)
        tied = [
            row["candidate"]
            for row in eligible
            if best - sign * row[metric]["mean"] <= float(selection["tie_tolerance"])
        ]
        selected = tied[:1] if selection.get("tie_break") == "spec_order" else tied
    else:
        tied = []
    summary = {
        "sweep_id": spec["sweep_id"],
        "coverage": "COMPLETE SWEEP" if complete else "PARTIAL SWEEP",
        "selection": selection,
        "tied_within_tolerance": tied,
        "selected_candidates": selected,
        "selection_scope": "complete sweep"
        if complete
        else "highest/lowest observed aggregate among completed runs (partial)",
        "invalid_runs": invalid,
        "rows": rows,
        "generated_at": utc_now(),
        "generator": "papers/photonic_QKD/utils/run_sweep.py --summarize",
    }
    write_json(sweep_dir / "summary.json", summary)
    lines = [
        f"# {spec['sweep_id']} — {summary['coverage']}",
        "",
        f"Selection: {selection}",
        "",
    ]
    header = ["candidate", "runs", "status", *report_metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        cells = [
            row["candidate"],
            f"{row['successful_runs']}/{row['expected_runs']}",
            row["status"],
        ]
        for key in report_metrics:
            entry = row[key]
            if entry is None:
                cells.append("—")
            elif entry["std"] is None:
                cells.append(f"{entry['mean']:.4f} (n=1)")
            else:
                cells.append(
                    f"{entry['mean']:.4f} ± {entry['std']:.4f} (n={entry['n']})"
                )
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"Tied within tolerance: {tied}", f"Selected: {selected}"]
    (sweep_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failed = [run for run in status["runs"] if run["status"] == "FAILED"]
    terminal = (
        "COMPLETED"
        if complete and not stopped
        else (
            "FAILED"
            if not stopped and not any(r["successful_runs"] for r in rows)
            else "PARTIAL"
        )
    )
    status.update(
        status=terminal,
        completed_at=utc_now(),
        summary_path="summary.json",
        selected_candidates=selected,
    )
    write_json(sweep_dir / "sweep_status.json", status)
    event = {
        "COMPLETED": "SWEEP_COMPLETED",
        "PARTIAL": "SWEEP_PARTIAL",
        "FAILED": "SWEEP_FAILED",
    }[terminal]
    log = logger.info if terminal == "COMPLETED" else logger.error
    log(
        "%s | summary=summary.json | selected=%s | failed=%s",
        event,
        selected,
        [run["run_id"] for run in failed],
    )


def resume_spec(stopped_dir: Path) -> dict[str, Any]:
    """Spec of a completion sweep for the unfinished pairs of ``stopped_dir``.

    Works for a sweep closed by a graceful stop and for one whose coordinator
    was killed (its status frozen at RUNNING). Every pair not COMPLETED is
    re-planned; when its latest run directory holds ``checkpoint.pt`` the run
    continues from it via ``--resume-from``.
    """
    spec = json.loads((stopped_dir / "spec.json").read_text())
    status = json.loads((stopped_dir / "sweep_status.json").read_text())
    completed = {
        (run["candidate"], int(run["seed"]))
        for run in status["runs"]
        if run["status"] == "COMPLETED"
    }
    planned = spec.get("only_runs") or [
        [candidate["name"], seed]
        for candidate in spec["candidates"]
        for seed in spec["seeds"]
    ]
    remaining = [
        [name, int(seed)]
        for name, seed in planned
        if (name, int(seed)) not in completed
    ]
    if not remaining:
        raise SystemExit(f"{stopped_dir} has no unfinished runs")
    resume_from = {}
    for name, seed in remaining:
        label = _run_label(name, seed)
        checkpoints = sorted((stopped_dir / "runs" / label).glob("run_*/checkpoint.pt"))
        if checkpoints:
            resume_from[label] = str(checkpoints[-1].parent.resolve())
    base_id = re.sub(r"_resume\d+$", "", spec["sweep_id"])
    attempt = int(spec.get("resume_index", 0)) + 1
    resumed = dict(spec)
    resumed.update(
        sweep_id=f"{base_id}_resume{attempt}",
        resume_index=attempt,
        resumes=os.path.relpath(stopped_dir.resolve(), REPO_ROOT.resolve())
        if stopped_dir.resolve().is_relative_to(REPO_ROOT.resolve())
        else str(stopped_dir.resolve()),
        only_runs=remaining,
        resume_from=resume_from,
        description=f"Resumption {attempt} of {stopped_dir.name}: {len(remaining)} unfinished runs, "
        f"{len(resume_from)} continue from checkpoints. " + spec.get("description", ""),
    )
    return resumed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("spec", nargs="?", type=Path, help="Sweep spec JSON")
    parser.add_argument(
        "--summarize", type=Path, help="Re-summarise an existing sweep directory"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Start a completion sweep for a stopped or killed sweep",
    )
    args = parser.parse_args(argv)
    if args.summarize:
        sweep_dir = args.summarize.resolve()
        spec = json.loads((sweep_dir / "spec.json").read_text())
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        configure_logging(sweep_dir / "summarize.log")
        summarize(sweep_dir, spec, status)
        return 0
    if args.resume:
        spec = resume_spec(args.resume.resolve())
    elif args.spec is None:
        parser.error("a spec path, --summarize or --resume is required")
    else:
        spec = json.loads(args.spec.read_text())
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = PAPER_DIR / "outdir" / "sweeps" / f"{spec['sweep_id']}_{stamp}"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    configure_logging(sweep_dir / "sweep.log")
    print(sweep_dir, flush=True)
    SweepCoordinator(spec, sweep_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
