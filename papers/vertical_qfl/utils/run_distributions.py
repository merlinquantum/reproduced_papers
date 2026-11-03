"""Run one experiment over several random feature distributions as a logged sweep.

With ``--grid`` it additionally crosses the permutation seeds with the Cartesian product of
hyperparameter values (a selection sweep, rule declared on the command line).

The paper repeats every experiment over 5 random pixel-to-client assignments
(permutation seeds 42-46 in the authors' code). This coordinator launches one
``implementation.py`` run per permutation seed, at most ``--workers`` at a time,
and is the only writer of ``sweep.log`` and ``sweep_status.json``. It makes no
selection: the summary reports every run and the mean and standard deviation.

Example
-------
    python utils/run_distributions.py --config configs/e1_8clients_original.json \
        --permutation-seeds 42 43 44 45 46 --workers 3 --threads 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)

logger = logging.getLogger("run_distributions")
SCHEMA_VERSION = 1
POLL_SECONDS = 20


def _epochs_done(run_dir: Path | None) -> int:
    """Count logged epochs, for the progress bar only (never for scientific values)."""
    if run_dir is None or not (run_dir / "run.log").exists():
        return 0
    text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    return text.count("TRAIN_EPOCH_COMPLETED")


def _latest_run_dir(candidate_dir: Path) -> Path | None:
    runs = sorted(candidate_dir.glob("run_*"))
    return runs[-1] if runs else None


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = config
    *parents, leaf = dotted_key.split(".")
    for parent in parents:
        node = node[parent]
    if leaf not in node:
        raise KeyError(f"Grid key {dotted_key} is not in the base config")
    node[leaf] = value


def _overrides(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in candidate.items()
        if k not in ("experiment", "permutation_seed")
    }


def _summarise_candidates(
    rows: list[dict[str, Any]],
    runs_per_candidate: int,
    select_metric: str | None,
    tie_tolerance: float,
    tie_break: list[str],
) -> dict[str, Any]:
    """Per-candidate means and, when requested, the predeclared selection."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(json.dumps(row["overrides"], sort_keys=True), []).append(row)
    candidates = []
    for key, members in groups.items():
        done = [m for m in members if m["status"] == "COMPLETED"]
        entry: dict[str, Any] = {
            "overrides": json.loads(key),
            "successful_runs": len(done),
            "expected_runs": len(members),
        }
        for metric in ("final_validation_accuracy", "final_test_accuracy"):
            values = [m[metric] for m in done if metric in m]
            if values:
                if any(not math.isfinite(v) for v in values):
                    raise ValueError(f"Non-finite {metric} in sweep results")
                entry[metric] = {
                    "mean": statistics.fmean(values),
                    "std_ddof1": statistics.stdev(values) if len(values) > 1 else None,
                }
        candidates.append(entry)
    result: dict[str, Any] = {"candidates": candidates, "selected": None, "tied": []}
    complete = all(c["successful_runs"] == c["expected_runs"] for c in candidates)
    if select_metric and complete:
        best = max(c[select_metric]["mean"] for c in candidates)
        tied = [
            c for c in candidates if c[select_metric]["mean"] >= best - tie_tolerance
        ]
        tied.sort(key=lambda c: tuple(c["overrides"][k] for k in tie_break))
        result["tied"] = [c["overrides"] for c in tied]
        result["selected"] = tied[0]["overrides"]
        result["rule"] = {
            "metric": select_metric,
            "direction": "maximize",
            "aggregation": "mean over permutation seeds",
            "tie_tolerance": tie_tolerance,
            "tie_break_ascending": tie_break,
        }
    elif select_metric:
        result["note"] = "partial sweep: no selection made"
    return result


def _write_summary(
    sweep_dir: Path,
    runs: list[dict[str, Any]],
    experiment: str,
    select_metric: str | None = None,
    tie_tolerance: float = 0.0,
    tie_break: list[str] | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    """Aggregate the completed runs' structured metrics into ``summary.json``.

    Returns the summary path and the selection result (``None`` without a grid).
    """
    rows = []
    for run in runs:
        if run["status"] != "COMPLETED":
            rows.append(
                {
                    "permutation_seed": run["candidate"]["permutation_seed"],
                    "seed": run["seed"],
                    "overrides": _overrides(run["candidate"]),
                    "status": run["status"],
                }
            )
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        rows.append(
            {
                "permutation_seed": run["candidate"]["permutation_seed"],
                "seed": run["seed"],
                "overrides": _overrides(run["candidate"]),
                "status": "COMPLETED",
                **(
                    {"final_validation_accuracy": metrics["final_validation_accuracy"]}
                    if "final_validation_accuracy" in metrics
                    else {}
                ),
                "final_test_accuracy": metrics["final_test_accuracy"],
                "final_test_loss": metrics["final_test_loss"],
                "best_epoch_test_accuracy_selected_on_test": metrics[
                    "best_epoch_test_accuracy_selected_on_test"
                ],
                "best_epoch": metrics["best_epoch"],
                "wall_clock_seconds": metrics["wall_clock_seconds"],
            }
        )
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    aggregate: dict[str, Any] = {
        "successful_runs": len(completed),
        "expected_runs": len(rows),
    }
    for key in (
        "final_test_accuracy",
        "best_epoch_test_accuracy_selected_on_test",
        "final_test_loss",
    ):
        values = [row[key] for row in completed]
        if any(not math.isfinite(v) for v in values):
            raise ValueError(f"Non-finite {key} in sweep results")
        aggregate[key] = {
            "mean": statistics.fmean(values) if values else None,
            "std_ddof1": statistics.stdev(values) if len(values) > 1 else None,
        }
    summary = {
        "experiment": experiment,
        "selection": "none (descriptive aggregate over feature distributions)",
        "runs": rows,
        "aggregate": aggregate,
    }
    selection = None
    if any(row["overrides"] for row in rows):
        per_candidate = len({r["permutation_seed"] for r in rows})
        selection = _summarise_candidates(
            rows, per_candidate, select_metric, tie_tolerance, tie_break or []
        )
        summary["selection"] = selection
        summary.pop(
            "aggregate"
        )  # an overall mean across different candidates is meaningless
    path = sweep_dir / "summary.json"
    write_json(path, summary)
    return path, selection


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Experiment config, relative to the paper folder or absolute",
    )
    parser.add_argument("--permutation-seeds", required=True, type=int, nargs="+")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Initialisation / batch-order seed shared by all runs",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Several training seeds, each crossed with every permutation seed",
    )
    parser.add_argument(
        "--candidates-file",
        type=Path,
        help="JSON list of override objects {dotted.key: value}; alternative to --grid",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--threads", type=int, default=3, help="Torch threads per worker"
    )
    parser.add_argument("--base-outdir", type=Path, default=PROJECT_DIR / "outdir")
    parser.add_argument("--tag", default="", help="Optional suffix for the sweep ID")
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="DOTTED.KEY=V1,V2",
        help="Hyperparameter candidates (numeric values); repeatable, crossed as a product",
    )
    parser.add_argument(
        "--select-metric", help="Metric to maximise, e.g. final_validation_accuracy"
    )
    parser.add_argument("--tie-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--tie-break",
        default="",
        help="Comma-separated override keys; ties go to the smallest values in this order",
    )
    args = parser.parse_args()

    base_config = json.loads(
        (
            args.config if args.config.is_absolute() else PROJECT_DIR / args.config
        ).read_text()
    )
    experiment = base_config.get("experiment", args.config.stem)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    sweep_id = f"{experiment}_{stamp}{('_' + args.tag) if args.tag else ''}"
    sweep_dir = (args.base_outdir / f"sweep_{sweep_id}").resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(sweep_dir / "sweep.log")

    grid_keys = [item.split("=", 1)[0] for item in args.grid]
    grid_values = [json.loads("[" + item.split("=", 1)[1] + "]") for item in args.grid]
    if args.candidates_file and args.grid:
        raise ValueError("Use either --grid or --candidates-file, not both")
    if args.candidates_file:
        path = args.candidates_file
        candidate_overrides = json.loads(
            (path if path.is_absolute() else PROJECT_DIR / path).read_text()
        )
    else:
        candidate_overrides = [
            dict(zip(grid_keys, combo)) for combo in itertools.product(*grid_values)
        ] or [{}]
    training_seeds = args.seeds or [args.seed]
    has_candidates = bool(args.grid or args.candidates_file)
    expected = (
        len(args.permutation_seeds) * len(candidate_overrides) * len(training_seeds)
    )
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sweep_id": sweep_id,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "expected_runs": expected,
        "runs": [],
        "summary_path": None,
        "selected_candidates": [],
        "error": None,
    }
    write_json(sweep_dir / "sweep_status.json", status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | config=%s | workers=%d",
        sweep_id,
        expected,
        args.config,
        args.workers,
    )

    # One job per (candidate, training seed, permutation seed); the repetition index
    # numbers the (training seed, permutation seed) pairs of a candidate.
    jobs = [
        (candidate_index, overrides, repetition, permutation_seed, seed)
        for candidate_index, overrides in enumerate(candidate_overrides)
        for repetition, (seed, permutation_seed) in enumerate(
            itertools.product(training_seeds, args.permutation_seeds)
        )
    ]
    pending = [
        (job_index, repetition, permutation_seed, candidate_index, overrides, seed)
        for job_index, (
            candidate_index,
            overrides,
            repetition,
            permutation_seed,
            seed,
        ) in enumerate(jobs)
    ]
    running: dict[int, dict[str, Any]] = {}
    try:
        while pending or running:
            while pending and len(running) < args.workers:
                (
                    job_index,
                    repetition,
                    permutation_seed,
                    candidate_index,
                    overrides,
                    seed,
                ) = pending.pop(0)
                label = (
                    f"c{candidate_index}_perm{permutation_seed}"
                    if has_candidates
                    else f"perm{permutation_seed}"
                )
                if len(training_seeds) > 1:
                    label += f"_s{seed}"
                candidate_dir = sweep_dir / f"cand_{label}"
                candidate_dir.mkdir(exist_ok=True)
                run_id = f"{sweep_id}_{label}"
                # Overlay: the resolved experiment config plus the sweep identity of this run.
                overlay = json.loads(json.dumps(base_config))
                overlay.update(
                    sweep_id=sweep_id,
                    repetition=repetition,
                    run_id=run_id,
                    seed=seed,
                )
                overlay["dataset"]["permutation_seed"] = permutation_seed
                for dotted_key, value in overrides.items():
                    _set_dotted(overlay, dotted_key, value)
                overlay["candidate_overrides"] = {
                    key.split(".")[-1]: value for key, value in overrides.items()
                }
                overlay["outdir"] = str(candidate_dir)
                overlay_path = candidate_dir / "launch_config.json"
                overlay_path.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
                command = [
                    sys.executable,
                    str(REPO_ROOT / "implementation.py"),
                    "--paper",
                    PROJECT_DIR.name,
                    "--config",
                    str(overlay_path),
                ]
                env = {
                    **os.environ,
                    "OMP_NUM_THREADS": str(args.threads),
                    "MKL_NUM_THREADS": str(args.threads),
                }
                stdout = (candidate_dir / "stdout.txt").open("w")
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                )
                entry = {
                    "run_id": run_id,
                    "candidate": {
                        "experiment": experiment,
                        "permutation_seed": permutation_seed,
                        **overlay["candidate_overrides"],
                    },
                    "seed": seed,
                    "repetition": repetition,
                    "status": "RUNNING",
                    "run_dir": None,
                    "run_log": None,
                    "run_status": None,
                    "config_path": None,
                    "metrics_path": None,
                    "progress": {
                        "current": 0,
                        "total": overlay["training"]["epochs"],
                        "unit": "epochs",
                    },
                }
                status["runs"].append(entry)
                running[job_index] = {
                    "process": process,
                    "entry": entry,
                    "dir": candidate_dir,
                    "stdout": stdout,
                }
                logger.info(
                    "CANDIDATE_STARTED | run_id=%s | candidate=%s | command=%s",
                    run_id,
                    entry["candidate"],
                    " ".join(command),
                )
                write_json(sweep_dir / "sweep_status.json", status)
            time.sleep(POLL_SECONDS)
            for job_index in list(running):
                job = running[job_index]
                entry = job["entry"]
                run_dir = _latest_run_dir(job["dir"])
                code = job["process"].poll()
                if code is None:
                    entry["progress"]["current"] = _epochs_done(run_dir)
                    continue
                job["stdout"].close()
                del running[job_index]
                entry.pop("progress", None)
                completed = False
                if run_dir is not None:
                    relative = run_dir.relative_to(sweep_dir)
                    entry.update(
                        run_dir=str(relative),
                        run_log=str(relative / "run.log"),
                        run_status=str(relative / "run_status.json"),
                        config_path=str(relative / "config_snapshot.json"),
                        metrics_path=str(relative / "metrics.json"),
                    )
                    run_status_path = run_dir / "run_status.json"
                    if run_status_path.exists():
                        completed = (
                            code == 0
                            and json.loads(run_status_path.read_text())["status"]
                            == "COMPLETED"
                        )
                if completed:
                    entry["status"] = "COMPLETED"
                    logger.info(
                        "CANDIDATE_COMPLETED | run_id=%s | metrics=%s",
                        entry["run_id"],
                        entry["metrics_path"],
                    )
                else:
                    entry["status"] = "FAILED"
                    logger.error(
                        "CANDIDATE_FAILED | run_id=%s | exit_code=%s | see %s",
                        entry["run_id"],
                        code,
                        job["dir"] / "stdout.txt",
                    )
                    if run_dir is None:
                        entry.update(
                            run_dir="",
                            run_log="",
                            run_status="",
                            config_path="",
                            metrics_path="",
                        )
            write_json(sweep_dir / "sweep_status.json", status)
            logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))

        summary_path, selection = _write_summary(
            sweep_dir,
            status["runs"],
            experiment,
            args.select_metric,
            args.tie_tolerance,
            [key.split(".")[-1] for key in args.tie_break.split(",") if key],
        )
        status["summary_path"] = str(summary_path.relative_to(sweep_dir))
        if selection and selection["selected"] is not None:
            status["selected_candidates"] = [selection["selected"]]
        failed = [run for run in status["runs"] if run["status"] != "COMPLETED"]
        status["status"] = (
            "COMPLETED"
            if not failed
            else ("PARTIAL" if len(failed) < expected else "FAILED")
        )
        status["completed_at"] = utc_now()
        if status["status"] == "FAILED":
            status["error"] = {"type": "AllRunsFailed", "message": "no run completed"}
        write_json(sweep_dir / "sweep_status.json", status)
        event = {
            "COMPLETED": "SWEEP_COMPLETED",
            "PARTIAL": "SWEEP_PARTIAL",
            "FAILED": "SWEEP_FAILED",
        }[status["status"]]
        if status["status"] == "FAILED":
            try:
                raise RuntimeError("no run completed")
            except RuntimeError:
                logger.exception("%s | summary=%s", event, status["summary_path"])
        else:
            logger.info(
                "%s | summary=%s | selected=%s",
                event,
                status["summary_path"],
                status["selected_candidates"] or "none",
            )
        return 0 if status["status"] == "COMPLETED" else 1
    except BaseException as exc:
        for job in running.values():
            job["process"].terminate()
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        for run in status["runs"]:
            if run["status"] == "RUNNING":
                run["status"] = "FAILED"
                run.pop("progress", None)
                for key in (
                    "run_dir",
                    "run_log",
                    "run_status",
                    "config_path",
                    "metrics_path",
                ):
                    run[key] = run[key] or ""
        write_json(sweep_dir / "sweep_status.json", status)
        logger.exception("SWEEP_FAILED | sweep_id=%s", sweep_id)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
