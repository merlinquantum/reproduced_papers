"""Sweep coordinator: plan expansion, parallel runs, status files and the summary.

A sweep plan (JSON, see ``utils/sweep_plans/``) predeclares the grid, the seeds and the
selection rule, so the winner is never chosen after looking at results. The
coordinator writes ``sweep.log`` and ``sweep_status.json`` itself; every run writes its
own ``run.log`` through :mod:`lib.runner`. Rankings are computed from the runs'
``metrics.json`` files only, never from logs.

Each planned run (candidate, seed) gets at most two attempts. A failed first attempt is
recorded in ``ledger.json`` with its cause before the single retry.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from runtime_lib.config import deep_update

from .experiment_logging import (
    configure_logging,
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_ATTEMPTS = 2
REQUIRED_PLAN_KEYS = (
    "sweep_id",
    "purpose",
    "paper",
    "base_config",
    "fixed_overrides",
    "seeds",
    "workers",
    "threads_per_worker",
    "selection",
)
REQUIRED_SELECTION_KEYS = (
    "metric",
    "split",
    "checkpoint",
    "direction",
    "aggregation",
    "tie_tolerance",
)


def load_plan(path: Path) -> dict[str, Any]:
    """Load and validate a sweep plan.

    Raises
    ------
    KeyError
        If a required plan or selection key is missing.
    ValueError
        If the direction, aggregation or checkpoint rule is not supported.
    """
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in REQUIRED_PLAN_KEYS:
        if key not in plan:
            raise KeyError(f"Missing required sweep plan key {key}")
    if ("axes" in plan) == ("candidates" in plan):
        raise KeyError("A sweep plan needs exactly one of 'axes' or 'candidates'")
    for key in REQUIRED_SELECTION_KEYS:
        if key not in plan["selection"]:
            raise KeyError(f"Missing required sweep plan key selection.{key}")
    selection = plan["selection"]
    if selection["direction"] not in ("maximize", "minimize"):
        raise ValueError(f"Unknown selection.direction {selection['direction']!r}")
    if selection["aggregation"] != "mean":
        raise ValueError("Only the 'mean' aggregation over seeds is implemented")
    if selection["checkpoint"] != "last":
        raise ValueError("Only the 'last' checkpoint rule is implemented")
    return plan


def _nested(dotted_overrides: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for dotted, value in dotted_overrides.items():
        cursor = nested
        *parents, leaf = dotted.split(".")
        for part in parents:
            cursor = cursor.setdefault(part, {})
        cursor[leaf] = value
    return nested


def candidate_name(axis_values: dict[str, Any]) -> str:
    """Stable short name such as ``layers2_lr0.1_init_std1.0``."""
    return "_".join(
        f"{key.split('.')[-1]}{value}" for key, value in axis_values.items()
    )


def expand_runs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the plan into every planned (candidate, seed) run, in a fixed order."""
    if "candidates" in plan:
        # Named one-at-a-time candidates (diagnosis plans), not a full grid.
        named = [(item["name"], item["overrides"]) for item in plan["candidates"]]
    else:
        axes = plan["axes"]
        named = [
            (candidate_name(values), values)
            for values in (
                dict(zip(axes, combination))
                for combination in itertools.product(*axes.values())
            )
        ]
    runs = []
    for name, axis_values in named:
        for repetition, seed in enumerate(plan["seeds"]):
            runs.append(
                {
                    "candidate_name": name,
                    "name": f"{name}_seed{seed}",
                    "axis_values": axis_values,
                    "seed": int(seed),
                    "repetition": repetition,
                }
            )
    return runs


def summarize(
    plan: dict[str, Any], status: dict[str, Any], sweep_dir: Path
) -> dict[str, Any]:
    """Aggregate the completed runs' structured metrics under the predeclared rule.

    Parameters
    ----------
    plan : dict
        Validated sweep plan.
    status : dict
        ``sweep_status.json`` content; run paths are relative to ``sweep_dir``.
    sweep_dir : pathlib.Path
        Directory the paths in ``status`` are relative to.

    Returns
    -------
    dict
        Per-candidate table, ``coverage`` (``COMPLETE SWEEP`` or ``PARTIAL SWEEP``) and
        the selected or tied candidates. A partial sweep reports the highest or lowest
        observed aggregate and never an unqualified winner.
    """
    selection = plan["selection"]
    metric = selection["metric"]
    expected_per_candidate = len(plan["seeds"])
    grouped: dict[str, dict[str, Any]] = {}
    for run in status["runs"]:
        name = run["candidate_name"]
        entry = grouped.setdefault(
            name,
            {"axis_values": run["axis_values"], "values": {}, "test_values": []},
        )
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads(
            (sweep_dir / run["metrics_path"]).read_text(encoding="utf-8")
        )
        final = metrics["history"][-1]
        value = final[metric]
        if not math.isfinite(value):
            continue
        entry["values"][run["seed"]] = value
        if "test_accuracy" in final:
            entry["test_values"].append(final["test_accuracy"])

    rows = []
    for name, entry in grouped.items():
        values = list(entry["values"].values())
        rows.append(
            {
                "candidate": name,
                "axis_values": entry["axis_values"],
                "successful_runs": len(values),
                "expected_runs": expected_per_candidate,
                "aggregate": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "per_seed": entry["values"],
                "test_mean_not_used_for_selection": (
                    statistics.fmean(entry["test_values"])
                    if entry["test_values"]
                    else None
                ),
                "status": "COMPLETE"
                if len(values) == expected_per_candidate
                else "PARTIAL",
            }
        )
    ranked = [row for row in rows if row["aggregate"] is not None]
    sign = 1 if selection["direction"] == "maximize" else -1
    ranked.sort(key=lambda row: -sign * row["aggregate"])
    selected: list[str] = []
    margin_note = None
    if ranked:
        best = ranked[0]["aggregate"]
        tolerance = float(selection["tie_tolerance"])
        selected = [
            row["candidate"]
            for row in ranked
            if abs(row["aggregate"] - best) <= tolerance
        ]
        if len(ranked) > 1:
            gap = abs(best - ranked[1]["aggregate"])
            spreads = [row["std"] for row in ranked[:2] if row["std"] is not None]
            margin_note = {
                "gap_to_runner_up": gap,
                "largest_std_of_top_two": max(spreads) if spreads else None,
                "exceeds_variability": bool(spreads) and gap > max(spreads),
            }
    expected_total = len(expand_runs(plan))
    observed_total = sum(row["successful_runs"] for row in rows)
    coverage = (
        "COMPLETE SWEEP"
        if observed_total == expected_total and len(rows) == len(ranked)
        else "PARTIAL SWEEP"
    )
    return {
        "sweep_id": plan["sweep_id"],
        "selection_rule": selection,
        "coverage": coverage,
        "expected_runs": expected_total,
        "successful_runs": observed_total,
        "candidates": sorted(rows, key=lambda row: row["candidate"]),
        "ranking": [row["candidate"] for row in ranked],
        "selected_candidates": selected,
        "selection_wording": (
            "selected candidate"
            if coverage == "COMPLETE SWEEP"
            else "observed aggregate only"
        ),
        "margin": margin_note,
    }


class _Coordinator:
    def __init__(self, plan: dict[str, Any], sweep_dir: Path, repo_root: Path) -> None:
        self.plan = plan
        self.sweep_dir = sweep_dir
        self.repo_root = repo_root
        self.lock = threading.Lock()
        self.ledger: list[dict[str, Any]] = []
        self.planned = expand_runs(plan)
        self.status: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "sweep_id": plan["sweep_id"],
            "status": "RUNNING",
            "started_at": utc_now(),
            "completed_at": None,
            "code": git_state(repo_root),
            "expected_runs": len(self.planned),
            "runs": [
                {
                    "run_id": f"{plan['sweep_id']}-{run['name']}-a1",
                    "candidate": run["axis_values"],
                    "candidate_name": run["candidate_name"],
                    "axis_values": run["axis_values"],
                    "seed": run["seed"],
                    "repetition": run["repetition"],
                    "status": "PLANNED",
                    "attempts": 0,
                    "run_dir": None,
                    "run_log": None,
                    "run_status": None,
                    "config_path": None,
                    "metrics_path": None,
                }
                for run in self.planned
            ],
            "summary_path": None,
            "selected_candidates": [],
            "error": None,
        }

    def write_status(self) -> None:
        write_json(self.sweep_dir / "sweep_status.json", self.status)
        write_json(self.sweep_dir / "ledger.json", {"attempts": self.ledger})

    def _command_and_config(
        self, run: dict[str, Any], attempt: int
    ) -> tuple[list[str], Path, Path]:
        plan = self.plan
        base = json.loads(
            (self.repo_root / "papers" / plan["paper"] / plan["base_config"]).read_text(
                encoding="utf-8"
            )
        )
        run_id = f"{plan['sweep_id']}-{run['name']}-a{attempt}"
        overlay = deep_update(
            copy.deepcopy(base), copy.deepcopy(plan["fixed_overrides"])
        )
        overlay = deep_update(overlay, _nested(run["axis_values"]))
        overlay.update(
            run_id=run_id,
            sweep_id=plan["sweep_id"],
            repetition=run["repetition"],
            candidate_overrides=run["axis_values"],
        )
        config_path = self.sweep_dir / "configs" / f"{run['name']}_a{attempt}.json"
        write_json(config_path, overlay)
        outdir = self.sweep_dir / "runs" / f"{run['name']}_a{attempt}"
        command = [
            sys.executable,
            str(self.repo_root / "implementation.py"),
            "--paper", plan["paper"],
            "--config", str(config_path),
            "--seed", str(run["seed"]),
            "--outdir", str(outdir),
        ]  # fmt: skip
        return command, config_path, outdir

    def _execute(self, index: int) -> None:
        run = self.planned[index]
        entry = self.status["runs"][index]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            command, config_path, outdir = self._command_and_config(run, attempt)
            run_id = f"{self.plan['sweep_id']}-{run['name']}-a{attempt}"
            with self.lock:
                entry.update(run_id=run_id, status="RUNNING", attempts=attempt)
                logger.info("CANDIDATE_STARTED | run_id=%s | candidate=%s | seed=%d | attempt=%d",
                            run_id, run["candidate_name"], run["seed"], attempt)  # fmt: skip
                self.write_status()
            outdir.mkdir(parents=True, exist_ok=True)
            environment = {
                **os.environ,
                "OMP_NUM_THREADS": str(self.plan["threads_per_worker"]),
                "MKL_NUM_THREADS": str(self.plan["threads_per_worker"]),
            }
            with (outdir / "stdout_stderr.txt").open("w", encoding="utf-8") as output:
                completed = subprocess.run(
                    command, cwd=self.repo_root, env=environment, stdout=output,
                    stderr=subprocess.STDOUT, check=False,
                )  # fmt: skip
            failure = self._failure_cause(completed.returncode, outdir)
            with self.lock:
                self.ledger.append(
                    {
                        "run_id": run_id, "candidate": run["candidate_name"],
                        "seed": run["seed"], "attempt": attempt, "command": command,
                        "exit_code": completed.returncode,
                        "outcome": "COMPLETED" if failure is None else "FAILED",
                        "cause": failure,
                        "corrective_change": None if failure is None else "none; automatic single retry",
                    }
                )  # fmt: skip
                if failure is None:
                    run_dir = next(iter(sorted(outdir.glob("run_*"))))
                    run_status = json.loads(
                        (run_dir / "run_status.json").read_text(encoding="utf-8")
                    )
                    entry.update(
                        status="COMPLETED",
                        candidate=run_status["candidate"],
                        run_dir=self._relative(run_dir),
                        run_log=self._relative(run_dir / "run.log"),
                        run_status=self._relative(run_dir / "run_status.json"),
                        config_path=self._relative(run_dir / "config_snapshot.json"),
                        metrics_path=self._relative(run_dir / "metrics.json"),
                    )  # fmt: skip
                    logger.info(
                        "CANDIDATE_COMPLETED | run_id=%s | metrics=%s",
                        run_id,
                        entry["metrics_path"],
                    )
                    self._log_progress()
                    self.write_status()
                    return
                logger.error(
                    "CANDIDATE_FAILED | run_id=%s | attempt=%d | cause=%s",
                    run_id,
                    attempt,
                    failure,
                )
                if attempt == MAX_ATTEMPTS:
                    entry["status"] = "FAILED"
                    self._log_progress()
                self.write_status()

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.sweep_dir)).replace(os.sep, "/")

    def _failure_cause(self, returncode: int, outdir: Path) -> str | None:
        if returncode != 0:
            tail = (
                (outdir / "stdout_stderr.txt")
                .read_text(encoding="utf-8")
                .strip()
                .splitlines()[-1:]
            )
            return f"exit code {returncode}: {tail[0] if tail else 'no output'}"
        run_dirs = sorted(outdir.glob("run_*"))
        if len(run_dirs) != 1:
            return f"expected one run directory, found {len(run_dirs)}"
        status_file = run_dirs[0] / "run_status.json"
        if (
            not status_file.is_file()
            or json.loads(status_file.read_text())["status"] != "COMPLETED"
        ):
            return "run_status.json missing or not COMPLETED"
        if not (run_dirs[0] / "metrics.json").is_file():
            return "metrics.json missing"
        return None

    def _log_progress(self) -> None:
        logger.info("SWEEP_PROGRESS | %s", sweep_progress(self.status))

    def run(self) -> dict[str, Any]:
        logger.info("SWEEP_STARTED | sweep_id=%s | expected_runs=%d | workers=%d",
                    self.plan["sweep_id"], len(self.planned), self.plan["workers"])  # fmt: skip
        self.write_status()
        try:
            with ThreadPoolExecutor(max_workers=int(self.plan["workers"])) as pool:
                for future in [
                    pool.submit(self._execute, i) for i in range(len(self.planned))
                ]:
                    future.result()
            summary = summarize(self.plan, self.status, self.sweep_dir)
            write_json(self.sweep_dir / "summary.json", summary)
            self.status.update(
                summary_path="summary.json",
                selected_candidates=summary["selected_candidates"],
                completed_at=utc_now(),
                status="COMPLETED"
                if summary["coverage"] == "COMPLETE SWEEP"
                else "PARTIAL",
            )
            self.write_status()
            terminal = (
                "SWEEP_COMPLETED"
                if self.status["status"] == "COMPLETED"
                else "SWEEP_PARTIAL"
            )
            logger.info(
                "%s | summary=summary.json | selected=%s",
                terminal,
                summary["selected_candidates"],
            )
        except Exception as exc:
            self.status.update(
                status="FAILED", completed_at=utc_now(),
                error={"type": type(exc).__name__, "message": str(exc)},
            )  # fmt: skip
            self.write_status()
            logger.exception("SWEEP_FAILED | sweep_id=%s", self.plan["sweep_id"])
            raise
        return summary


def run_sweep(plan_path: Path, sweep_dir: Path, repo_root: Path) -> dict[str, Any]:
    """Run a sweep plan to completion and return its summary.

    Parameters
    ----------
    plan_path : pathlib.Path
        Sweep plan JSON.
    sweep_dir : pathlib.Path
        Directory receiving ``sweep.log``, ``sweep_status.json``, ``ledger.json``,
        ``summary.json`` and one ``runs/<run>/`` folder per attempt.
    repo_root : pathlib.Path
        Repository root holding ``implementation.py``.

    Returns
    -------
    dict
        The summary written to ``summary.json``.
    """
    plan = load_plan(plan_path)
    sweep_dir = Path(sweep_dir)
    configure_logging(sweep_dir / "sweep.log")
    return _Coordinator(plan, sweep_dir, Path(repo_root)).run()
