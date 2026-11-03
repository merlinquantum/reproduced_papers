"""Sweep coordinator: run one config over candidates x seeds and write sweep evidence.

Each planned run is ``python implementation.py --paper distributed_qrc --config
<config> --seed <seed> [candidate flags]`` in its own output folder under the
sweep directory. The coordinator alone writes ``sweep.log`` and
``sweep_status.json``; every worker writes its own ``run.log``. Aggregates come
from the workers' ``metrics.json`` files, never from logs.

A candidate is a JSON object of paper CLI flags, e.g.
``{"observable": "z0", "update": "sequential"}``. With no ``--candidates`` the
sweep has the single candidate ``{}`` (the config as is), which is how seed
repetitions of E1-E4 are run.

Selection (optional, declared before the run): ``--select-cells`` lists
``reservoir|readout`` cells; each candidate's score is the mean over seeds and
selected cells of ``--select-metric`` (default ``eval.rmse``), minimized.
Candidates within ``--tie-tolerance`` (relative) of the best are tied; the tie is
resolved in favour of the first candidate listed (the default interpretation).

Usage (from the repository root)::

    python papers/distributed_qrc/utils/run_sweep.py --sweep-id E1 \\
        --config configs/e1_srsr_original.json --seeds 0 1 2
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

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parents[1]
# The paper is imported as the package ``distributed_qrc``; its lib/ imports runtime_lib
# from the repository root.
sys.path[:0] = [str(PAPER_DIR.parent), str(REPO_ROOT)]

from distributed_qrc.lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)
from distributed_qrc.lib.runner import lib_sha256  # noqa: E402

logger = logging.getLogger("distributed_qrc.sweep")
POLL_SECONDS = 20


def candidate_label(candidate: dict) -> str:
    if not candidate:
        return "default"
    return "_".join(f"{k}-{v}" for k, v in sorted(candidate.items()))


def candidate_flags(candidate: dict) -> list[str]:
    flags = []
    for key, value in candidate.items():
        flags += [f"--{key.replace('_', '-')}", str(value)]
    return flags


def find_run_dir(run_root: Path) -> Path | None:
    runs = sorted(run_root.glob("run_*"))
    return runs[-1] if runs else None


def relative(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def aggregate(entries: list[dict], sweep_dir: Path) -> dict:
    """Per candidate and cell: mean and std over completed seeds, from metrics.json."""

    table: dict[str, dict] = {}
    for entry in entries:
        if entry["status"] != "COMPLETED":
            continue
        metrics = json.loads((sweep_dir / entry["metrics_path"]).read_text())
        label = candidate_label(entry["candidate"])
        for cell in metrics["cells"]:
            key = f"{cell['configuration']}|{cell['reservoir_type']}|{cell['readout_type']}"
            slot = table.setdefault(label, {}).setdefault(key, {"seeds": []})
            slot["seeds"].append(entry["seed"])
            for split in ("eval", "test", "test_mw"):
                for name, value in cell["metrics"][split].items():
                    if not np.isfinite(value):
                        raise ValueError(
                            f"Non-finite {split}.{name} in {entry['run_dir']}"
                        )
                    slot.setdefault(f"{split}.{name}", []).append(value)
    summary: dict[str, dict] = {}
    for label, cells in table.items():
        summary[label] = {}
        for key, slot in cells.items():
            row = {"n_seeds": len(slot["seeds"]), "seeds": sorted(slot["seeds"])}
            for name, values in slot.items():
                if name == "seeds":
                    continue
                arr = np.asarray(values, dtype=float)
                row[name] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                }
            summary[label][key] = row
    return summary


def select(
    summary: dict,
    candidates: list[dict],
    cells: list[str],
    metric: str,
    tolerance: float,
) -> dict:
    scores = {}
    for candidate in candidates:
        label = candidate_label(candidate)
        rows = summary.get(label, {})
        values = [
            row[metric]["mean"]
            for key, row in rows.items()
            if any(key.endswith("|" + cell) for cell in cells)
        ]
        scores[label] = float(np.mean(values)) if values else float("nan")
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    best = min(finite.values())
    tied = [k for k, v in finite.items() if v <= best * (1 + tolerance)]
    ordered = [candidate_label(c) for c in candidates]
    chosen = min(tied, key=ordered.index)
    return {
        "metric": metric,
        "cells": cells,
        "scores": scores,
        "tied": tied,
        "selected": chosen,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument(
        "--config", required=True, help="Config path relative to the paper folder"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--candidates", default="[{}]", help="JSON list of flag dicts")
    parser.add_argument("--sweep-dir", default=None, help="Default: outdir/sweep_<id>")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--select-cells", default=None, help="Comma-separated reservoir|readout cells"
    )
    parser.add_argument("--select-metric", default="eval.rmse")
    parser.add_argument("--tie-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    candidates = json.loads(args.candidates)
    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else PAPER_DIR / "outdir" / f"sweep_{args.sweep_id}"
    )
    if not sweep_dir.is_absolute():
        sweep_dir = PAPER_DIR / sweep_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(sweep_dir / "sweep.log")

    planned = [(c, s) for c in candidates for s in args.seeds]
    entries = []
    for candidate, seed in planned:
        root = sweep_dir / f"{candidate_label(candidate)}_seed{seed}"
        entries.append(
            {
                "worker_root": relative(root, sweep_dir),
                "run_id": None,
                "candidate": candidate,
                "seed": seed,
                "repetition": seed,
                "status": "PLANNED",
                "run_dir": relative(root, sweep_dir),
                "run_log": None,
                "run_status": None,
                "config_path": None,
                "metrics_path": None,
            }
        )
    status = {
        "schema_version": 1,
        "sweep_id": args.sweep_id,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": {**git_state(PAPER_DIR), "lib_sha256": lib_sha256()},
        "expected_runs": len(planned),
        "config": args.config,
        "device": args.device,
        "runs": entries,
        "summary_path": None,
        "selected_candidates": [],
        "error": None,
    }
    status_path = sweep_dir / "sweep_status.json"
    write_json(status_path, status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | config=%s",
        args.sweep_id,
        len(planned),
        args.config,
    )

    try:
        for entry in entries:
            root = sweep_dir / entry["worker_root"]
            root.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(REPO_ROOT / "implementation.py"),
                "--paper",
                "distributed_qrc",
                "--config",
                args.config,
                "--seed",
                str(entry["seed"]),
                "--outdir",
                str(root),
                *candidate_flags(entry["candidate"]),
            ]
            if args.device:
                command += ["--device", args.device]
            entry.update(status="RUNNING", command=" ".join(command[1:]))
            write_json(status_path, status)
            logger.info(
                "CANDIDATE_STARTED | run=%s | candidate=%s | seed=%d",
                entry["worker_root"],
                entry["candidate"],
                entry["seed"],
            )
            env = {
                **os.environ,
                "DISTRIBUTED_QRC_SWEEP_ID": args.sweep_id,
                "DISTRIBUTED_QRC_SWEEP_CANDIDATE": json.dumps(entry["candidate"]),
            }
            with (root / "worker_stdout.txt").open("w") as stdout:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                while process.poll() is None:
                    time.sleep(POLL_SECONDS)
                    run_dir = find_run_dir(root)
                    if run_dir and (run_dir / "run_status.json").exists():
                        progress = json.loads(
                            (run_dir / "run_status.json").read_text()
                        ).get("progress")
                        if progress and progress != entry.get("progress"):
                            entry["progress"] = progress
                            write_json(status_path, status)
                            logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))
            run_dir = find_run_dir(root)
            run_status = (
                json.loads((run_dir / "run_status.json").read_text()) if run_dir else {}
            )
            if run_dir:
                entry.update(
                    run_id=run_dir.name,
                    run_dir=relative(run_dir, sweep_dir),
                    run_log=relative(run_dir / "run.log", sweep_dir),
                    run_status=relative(run_dir / "run_status.json", sweep_dir),
                    config_path=relative(run_dir / "config_snapshot.json", sweep_dir),
                )
            if process.returncode == 0 and run_status.get("status") == "COMPLETED":
                entry.update(
                    status="COMPLETED",
                    metrics_path=relative(
                        run_dir / run_status["metrics_path"], sweep_dir
                    ),
                )
                entry.pop("progress", None)
                logger.info(
                    "CANDIDATE_COMPLETED | run=%s | metrics=%s",
                    entry["run_dir"],
                    entry["metrics_path"],
                )
            else:
                entry.update(status="FAILED")
                logger.error(
                    "CANDIDATE_FAILED | run=%s | returncode=%s | run_status=%s",
                    entry["run_dir"],
                    process.returncode,
                    run_status.get("status"),
                )
            write_json(status_path, status)
            logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))

        summary = aggregate(entries, sweep_dir)
        completed = sum(e["status"] == "COMPLETED" for e in entries)
        record = {
            "sweep_id": args.sweep_id,
            "config": args.config,
            "completed_runs": completed,
            "expected_runs": len(planned),
            "cells": summary,
        }
        if args.select_cells:
            record["selection"] = select(
                summary,
                candidates,
                args.select_cells.split(","),
                args.select_metric,
                args.tie_tolerance,
            )
            status["selected_candidates"] = (
                record["selection"]["tied"] if completed == len(planned) else []
            )
        summary_path = sweep_dir / "summary.json"
        write_json(summary_path, record)
        status["summary_path"] = relative(summary_path, sweep_dir)
        status["completed_at"] = utc_now()
        if completed == len(planned):
            status["status"] = "COMPLETED"
            write_json(status_path, status)
            logger.info(
                "SWEEP_COMPLETED | summary=%s | selected=%s",
                status["summary_path"],
                status["selected_candidates"],
            )
        else:
            status["status"] = "PARTIAL"
            write_json(status_path, status)
            logger.info(
                "SWEEP_PARTIAL | summary=%s | completed=%d/%d",
                status["summary_path"],
                completed,
                len(planned),
            )
        return 0 if completed == len(planned) else 1
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("SWEEP_FAILED | sweep_id=%s", args.sweep_id)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
