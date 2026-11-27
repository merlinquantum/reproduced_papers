"""Run a QNN_STOCK sweep: every (candidate config, seed) pair as one isolated run.

The coordinator is the only writer of ``sweep.log`` and ``sweep_status.json``.
Each run is a separate ``implementation.py`` process with its own output
folder, so parallel runs never share a log handler or a timestamped run
directory.

Usage (from the repository root)::

    python papers/QNN_STOCK/utils/run_sweep.py \
        --plan papers/QNN_STOCK/configs/sweeps/phase3_s1_core.json --workers 16
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_DIR))

from lib.experiment_logging import (  # noqa: E402
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)
from lib.runner import SCHEMA_VERSION  # noqa: E402

logger = logging.getLogger("qnn_stock.sweep")


def _configure_coordinator_logging(sweep_dir: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%SZ"
    )
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(sweep_dir / "sweep.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.handlers = [file_handler, stream_handler]
    logger.propagate = False


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    """Return every finite numeric leaf of ``metrics.json`` keyed by its path."""

    flat: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            flat.update(_flatten_numbers(child, f"{prefix}/{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flat[prefix] = float(value)
    return flat


def expand_candidates(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the candidate records of a plan.

    ``candidates`` lists named configs. ``grid`` (optional) adds one candidate
    per (base config, override) pair: ``{"configs": [...], "overrides":
    {"label": {...}}}`` gives the candidate ID ``<config>__<label>``.
    """

    records = [{"config": name} for name in plan.get("candidates", [])]
    grid = plan.get("grid")
    if grid:
        for base_config in grid["configs"]:
            for label, overrides in grid["overrides"].items():
                records.append(
                    {
                        "config": f"{base_config}__{label}",
                        "base_config": base_config,
                        "overrides": overrides,
                    }
                )
    ids = [r["config"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate IDs in the sweep plan")
    return records


class Sweep:
    """Coordinator state for one sweep directory."""

    def __init__(self, plan: dict[str, Any], sweep_dir: Path, python: str) -> None:
        self.plan = plan
        self.sweep_dir = sweep_dir
        self.python = python
        self.lock = threading.Lock()
        runs = []
        for candidate in expand_candidates(plan):
            for seed in plan["seeds"]:
                run_id = f"{candidate['config']}_s{seed}"
                run_parent = Path("runs") / run_id
                runs.append(
                    {
                        "run_id": run_id,
                        "candidate": candidate,
                        "seed": seed,
                        "repetition": plan["repetition"],
                        "status": "PLANNED",
                        "attempt": 0,
                        "command": None,
                        "run_dir": None,
                        "run_log": None,
                        "run_status": None,
                        "config_path": None,
                        "metrics_path": None,
                        "stdout": (run_parent / "stdout.log").as_posix(),
                        "error": None,
                    }
                )
        self.status: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "sweep_id": plan["sweep_id"],
            "status": "RUNNING",
            "started_at": utc_now(),
            "completed_at": None,
            "code": git_state(REPO_ROOT),
            "expected_runs": len(runs),
            "runs": runs,
            "summary_path": None,
            "selected_candidates": [],
            "selection_rule": plan["evaluation"]["selection_rule"],
            "plan": plan,
            "error": None,
        }

    def write_status(self) -> None:
        write_json(self.sweep_dir / "sweep_status.json", self.status)
        if self.status["status"] == "RUNNING":
            logger.info("SWEEP_PROGRESS | %s", sweep_progress(self.status))

    def run_one(self, entry: dict[str, Any]) -> None:
        candidate = entry["candidate"]["config"]
        seed = entry["seed"]
        run_parent = self.sweep_dir / "runs" / entry["run_id"]
        run_parent.mkdir(parents=True, exist_ok=True)
        base_config = entry["candidate"].get("base_config", candidate)
        named_config = json.loads(
            (PAPER_DIR / "configs" / f"{base_config}.json").read_text(encoding="utf-8")
        )
        named_config = _deep_merge(
            named_config, entry["candidate"].get("overrides", {})
        )
        run_config = _deep_merge(
            named_config,
            {
                "seed": seed,
                "outdir": str(run_parent.resolve()),
                "sweep_id": self.plan["sweep_id"],
                "run_id": entry["run_id"],
                "candidate": entry["candidate"],
                "repetition": entry["repetition"],
            },
        )
        config_path = run_parent / "sweep_config.json"
        write_json(config_path, run_config)
        command = [
            self.python,
            str(REPO_ROOT / "implementation.py"),
            "--paper",
            "QNN_STOCK",
            "--config",
            str(config_path.resolve()),
        ]
        with self.lock:
            entry["status"] = "RUNNING"
            entry["attempt"] += 1
            entry["command"] = " ".join(command)
            logger.info(
                "CANDIDATE_STARTED | run_id=%s | candidate=%s | seed=%d",
                entry["run_id"],
                candidate,
                seed,
            )
            self.write_status()

        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env["OPENBLAS_NUM_THREADS"] = "1"
        with (run_parent / "stdout.log").open("w", encoding="utf-8") as stdout:
            returncode = subprocess.call(
                command, cwd=REPO_ROOT, stdout=stdout, stderr=subprocess.STDOUT, env=env
            )

        run_dirs = sorted(run_parent.glob("run_*"))
        error = None
        if returncode != 0:
            error = f"exit code {returncode}"
        elif len(run_dirs) != 1:
            error = f"expected one run directory, found {len(run_dirs)}"
        else:
            run_status = json.loads((run_dirs[0] / "run_status.json").read_text())
            metrics = json.loads((run_dirs[0] / "metrics.json").read_text())
            if run_status.get("status") != "COMPLETED":
                error = f"run status {run_status.get('status')}"
            elif not all(math.isfinite(v) for v in _flatten_numbers(metrics).values()):
                error = "non-finite metric"

        with self.lock:
            if run_dirs:
                run_dir = run_dirs[-1].relative_to(self.sweep_dir)
                entry["run_dir"] = run_dir.as_posix()
                entry["run_log"] = (run_dir / "run.log").as_posix()
                entry["run_status"] = (run_dir / "run_status.json").as_posix()
                entry["config_path"] = (run_dir / "config_snapshot.json").as_posix()
                entry["metrics_path"] = (run_dir / "metrics.json").as_posix()
            if error is None:
                entry["status"] = "COMPLETED"
                logger.info(
                    "CANDIDATE_COMPLETED | run_id=%s | metrics=%s",
                    entry["run_id"],
                    entry["metrics_path"],
                )
            else:
                entry["status"] = "FAILED"
                entry["error"] = error
                logger.error(
                    "CANDIDATE_FAILED | run_id=%s | error=%s | stdout=%s",
                    entry["run_id"],
                    error,
                    entry["stdout"],
                )
            self.write_status()

    def write_summary(self) -> Path:
        rows = []
        for entry in self.status["runs"]:
            if entry["status"] != "COMPLETED":
                continue
            metrics = json.loads((self.sweep_dir / entry["metrics_path"]).read_text())
            for path, value in _flatten_numbers(metrics).items():
                rows.append(
                    {
                        "run_id": entry["run_id"],
                        "candidate": entry["candidate"]["config"],
                        "seed": entry["seed"],
                        "metric_path": path,
                        "value": value,
                    }
                )
        summary_path = self.sweep_dir / "summary.json"
        write_json(
            summary_path,
            {
                "sweep_id": self.plan["sweep_id"],
                "completed_runs": sum(
                    entry["status"] == "COMPLETED" for entry in self.status["runs"]
                ),
                "expected_runs": self.status["expected_runs"],
                "rows": rows,
            },
        )
        return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--sweep-root", type=Path, default=PAPER_DIR / "outdir" / "sweeps"
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = (args.sweep_root / f"{plan['sweep_id']}_{stamp}").resolve()
    sweep_dir.mkdir(parents=True)
    _configure_coordinator_logging(sweep_dir)

    sweep = Sweep(plan, sweep_dir, args.python)
    write_json(sweep_dir / "sweep_status.json", sweep.status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | workers=%d | dir=%s",
        plan["sweep_id"],
        sweep.status["expected_runs"],
        args.workers,
        sweep_dir,
    )
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(sweep.run_one, sweep.status["runs"]))
        summary_path = sweep.write_summary()
        sweep.status["summary_path"] = summary_path.relative_to(sweep_dir).as_posix()
        failed = [
            e["run_id"] for e in sweep.status["runs"] if e["status"] != "COMPLETED"
        ]
        sweep.status["status"] = "PARTIAL" if failed else "COMPLETED"
        sweep.status["completed_at"] = utc_now()
        write_json(sweep_dir / "sweep_status.json", sweep.status)
        if failed:
            logger.warning(
                "SWEEP_PARTIAL | summary=%s | failed=%s",
                sweep.status["summary_path"],
                ",".join(failed),
            )
            return 1
        logger.info(
            "SWEEP_COMPLETED | summary=%s | selected=none (evaluation matrix)",
            sweep.status["summary_path"],
        )
        return 0
    except Exception as exc:
        sweep.status["status"] = "FAILED"
        sweep.status["completed_at"] = utc_now()
        sweep.status["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(sweep_dir / "sweep_status.json", sweep.status)
        logger.error("SWEEP_FAILED | %s\n%s", exc, traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
