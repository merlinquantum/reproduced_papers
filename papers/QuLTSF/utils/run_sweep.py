"""Run a planned QuLTSF sweep and record it.

A sweep plan (``configs/sweeps/<name>.json``) lists entries, each a config
overlay with the lookbacks, horizons and seeds to cross. Every planned run is
executed once, as its own ``implementation.py`` process; ``--workers N``
runs N such processes at a time on the same device. Each run keeps its
own ``run.log``, ``run_status.json``, ``config_snapshot.json`` and
``metrics.json``. Only this coordinator writes ``sweep.log`` and
``sweep_status.json``.

Layout::

    <sweep_dir>/
        sweep.log, sweep_status.json, sweep_summary.json, sweep_summary.csv
        <candidate>_seed<seed>/run_YYYYMMDD-HHMMSS/...

A planned run that fails is recorded as FAILED and the sweep ends PARTIAL.
The coordinator refuses to reuse a sweep folder: a rerun of failed runs is a
new sweep in a new folder, recorded in LOG.md.

Usage (from the repository root)::

    python papers/QuLTSF/utils/run_sweep.py --plan configs/sweeps/table1.json \
        --sweep-dir outdir/sweep_table1 --device cuda:0 --workers 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)
from utils.summarise_sweep import summarise_sweep  # noqa: E402

logger = logging.getLogger("qultsf.sweep")

POLL_SECONDS = 10


def expand_plan(plan: dict) -> list[dict]:
    """Return every planned run of a sweep plan, in execution order.

    Runs are ordered seed-major, so the first seed's complete table exists
    before later seeds start.

    Parameters
    ----------
    plan : dict
        Parsed sweep plan with ``entries``; each entry has ``name``,
        ``config``, ``seeds``, ``sequence_lengths``, ``prediction_horizons``
        and optional ``extra_args`` (more CLI flags, e.g. ``--model-name``).
        An optional ``run_seeds`` subset of ``seeds`` runs only those seeds
        while keeping each run's repetition index from the full ``seeds``
        list, so a follow-up sweep of leftover runs stays comparable.

    Returns
    -------
    list[dict]
        Planned runs with ``run_id``, ``candidate``, ``seed``, ``repetition``
        and the CLI ``args``.

    Raises
    ------
    ValueError
        If two planned runs share an ID.
    """

    def seeds_to_run(entry: dict) -> list[int]:
        subset = entry.get("run_seeds", entry["seeds"])
        unknown = sorted(set(subset) - set(entry["seeds"]))
        if unknown:
            raise ValueError(f"{entry['name']}: run_seeds {unknown} not in seeds")
        return subset

    all_seeds = sorted(
        {seed for entry in plan["entries"] for seed in seeds_to_run(entry)}
    )
    runs = []
    for seed in all_seeds:
        for entry in plan["entries"]:
            if seed not in seeds_to_run(entry):
                continue
            for seq_len, pred_len in itertools.product(
                entry["sequence_lengths"], entry["prediction_horizons"]
            ):
                candidate = f"{entry['name']}_L{seq_len}_T{pred_len}"
                runs.append(
                    {
                        "run_id": f"{candidate}_seed{seed}",
                        "candidate": candidate,
                        "seed": seed,
                        "repetition": entry["seeds"].index(seed),
                        "args": [
                            "--config",
                            entry["config"],
                            "--seed",
                            str(seed),
                            "--sequence-length",
                            str(seq_len),
                            "--prediction-horizon",
                            str(pred_len),
                            *entry.get("extra_args", []),
                        ],
                    }
                )
    ids = [run["run_id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("The sweep plan expands to duplicate runs.")
    return runs


def _ledger_entry(
    sweep_dir: Path, run: dict, run_dir: Path | None, status: str
) -> dict:
    def rel(name: str) -> str | None:
        if run_dir is None:
            return None
        return (run_dir / name).relative_to(sweep_dir).as_posix()

    return {
        "run_id": run["run_id"],
        "candidate": run["candidate"],
        "seed": run["seed"],
        "repetition": run["repetition"],
        "status": status,
        "command": " ".join(run["command"]) if "command" in run else None,
        "run_dir": None
        if run_dir is None
        else run_dir.relative_to(sweep_dir).as_posix(),
        "run_log": rel("run.log"),
        "run_status": rel("run_status.json"),
        "config_path": rel("config_snapshot.json"),
        "metrics_path": rel("metrics.json"),
    }


def _run_dirs(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("run_*") if path.is_dir())


def _terminal_status(run_dir: Path | None, exit_code: int) -> str:
    if run_dir is None or not (run_dir / "run_status.json").is_file():
        return "FAILED"
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))[
        "status"
    ]
    if (
        status == "COMPLETED"
        and exit_code == 0
        and (run_dir / "metrics.json").is_file()
    ):
        return "COMPLETED"
    return "FAILED"


def run_sweep(
    plan_path: Path, sweep_dir: Path, device: str, python: str, workers: int
) -> int:
    """Execute a sweep and return a process exit code (0 only when complete).

    Parameters
    ----------
    plan_path : pathlib.Path
        Plan path relative to ``papers/QuLTSF``.
    sweep_dir : pathlib.Path
        New, absolute sweep folder.
    device : str
        Device passed to every run (``--device``).
    python : str
        Interpreter used for the runs.
    workers : int
        Number of runs executed at the same time (separate processes sharing
        the device; each run still writes only its own logs).

    Returns
    -------
    int
        0 for a complete sweep, 1 for a partial one.

    Raises
    ------
    FileExistsError
        If ``sweep_dir`` already holds a sweep.
    """
    plan = json.loads((PROJECT_DIR / plan_path).read_text(encoding="utf-8"))
    runs = expand_plan(plan)
    status_path = sweep_dir / "sweep_status.json"
    if status_path.exists():
        raise FileExistsError(f"{sweep_dir} already holds a sweep; use a new folder.")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(sweep_dir / "sweep.log")

    status = {
        "schema_version": 1,
        "sweep_id": plan["sweep_id"],
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "plan_path": plan_path.as_posix(),
        "device": device,
        "expected_runs": len(runs),
        "runs": [_ledger_entry(sweep_dir, run, None, "PLANNED") for run in runs],
        "summary_path": None,
        "selected_candidates": [],
        "error": None,
    }
    write_json(status_path, status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | device=%s | plan=%s",
        plan["sweep_id"],
        len(runs),
        device,
        plan_path.as_posix(),
    )

    # Each worker gets an equal share of the CPU threads so parallel runs do
    # not oversubscribe the cores (BLAS/OpenMP default to every core).
    threads = str(max(1, (os.cpu_count() or 1) // workers))
    child_env = {**os.environ, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
    max_epochs = plan.get("max_epochs")
    pending = list(range(len(runs)))
    active: dict[int, tuple[subprocess.Popen, object]] = {}

    def start(index: int) -> None:
        run = runs[index]
        run_folder = sweep_dir / run["run_id"]
        run_folder.mkdir(parents=True)
        run["command"] = [
            python,
            "implementation.py",
            "--paper",
            "QuLTSF",
            *run["args"],
            "--device",
            device,
            "--outdir",
            str(run_folder.resolve()),
            "--sweep-id",
            plan["sweep_id"],
            "--sweep-candidate",
            run["candidate"],
            "--repetition",
            str(run["repetition"]),
        ]
        status["runs"][index] = _ledger_entry(sweep_dir, run, None, "RUNNING")
        write_json(status_path, status)
        logger.info(
            "CANDIDATE_STARTED | run_id=%s | candidate=%s | seed=%d | threads=%s | command=%s",
            run["run_id"],
            run["candidate"],
            run["seed"],
            threads,
            " ".join(run["command"][1:]),
        )
        console = (run_folder / "console.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            run["command"],
            cwd=REPO_ROOT,
            stdout=console,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
        active[index] = (process, console)

    def finish(index: int, process: subprocess.Popen) -> None:
        run = runs[index]
        run_folder = sweep_dir / run["run_id"]
        run_dirs = _run_dirs(run_folder)
        run_dir = run_dirs[-1] if run_dirs else None
        result = _terminal_status(run_dir, process.returncode)
        status["runs"][index] = _ledger_entry(sweep_dir, run, run_dir, result)
        write_json(status_path, status)
        if result == "COMPLETED":
            logger.info(
                "CANDIDATE_COMPLETED | run_id=%s | metrics=%s",
                run["run_id"],
                status["runs"][index]["metrics_path"],
            )
        else:
            logger.info(
                "CANDIDATE_FAILED | run_id=%s | exit_code=%s | console=%s",
                run["run_id"],
                process.returncode,
                (run_folder / "console.log").relative_to(sweep_dir).as_posix(),
            )

    def record_progress(index: int) -> None:
        run = runs[index]
        run_dirs = _run_dirs(sweep_dir / run["run_id"])
        if not run_dirs:
            return
        entry = _ledger_entry(sweep_dir, run, run_dirs[-1], "RUNNING")
        log_path = run_dirs[-1] / "run.log"
        if max_epochs and log_path.is_file():
            epochs = log_path.read_text(encoding="utf-8", errors="replace").count(
                "TRAIN_EPOCH_COMPLETED"
            )
            # Upper bound: early stopping usually ends a run sooner.
            entry["progress"] = {
                "current": epochs,
                "total": max_epochs,
                "unit": "epochs",
            }
        status["runs"][index] = entry

    try:
        while pending or active:
            while pending and len(active) < workers:
                start(pending.pop(0))
            time.sleep(POLL_SECONDS)
            for index, (process, console) in list(active.items()):
                if process.poll() is None:
                    record_progress(index)
                    continue
                console.close()
                del active[index]
                finish(index, process)
            write_json(status_path, status)
            logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))

        summary_path = summarise_sweep(sweep_dir)
        complete = all(entry["status"] == "COMPLETED" for entry in status["runs"])
        status.update(
            summary_path=summary_path.name,
            status="COMPLETED" if complete else "PARTIAL",
            completed_at=utc_now(),
        )
        write_json(status_path, status)
        if complete:
            logger.info(
                "SWEEP_COMPLETED | summary=%s | selected=none", summary_path.name
            )
            return 0
        logger.info(
            "SWEEP_PARTIAL | summary=%s | failed=%s",
            summary_path.name,
            [
                entry["run_id"]
                for entry in status["runs"]
                if entry["status"] != "COMPLETED"
            ],
        )
        return 1
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("SWEEP_FAILED | sweep_id=%s", plan["sweep_id"])
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--plan", type=Path, required=True, help="Plan path relative to papers/QuLTSF"
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="New sweep folder, relative to papers/QuLTSF unless absolute",
    )
    parser.add_argument(
        "--device", required=True, help="Device for every run, e.g. cuda:0"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Runs executed in parallel (default 1)"
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    sweep_dir = (
        args.sweep_dir if args.sweep_dir.is_absolute() else PROJECT_DIR / args.sweep_dir
    )
    return run_sweep(args.plan, sweep_dir, args.device, sys.executable, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
