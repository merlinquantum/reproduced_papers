"""Seed-repetition sweep coordinator for one experiment config.

Launches one ``implementation.py`` run per seed as a separate process (each
with ``--num-threads 1``; the small circuits gain nothing from intra-op
threads), at most ``--max-parallel`` at a time. Each worker writes its own
run directory under ``<sweep_dir>/seed<k>/``; only this coordinator writes
``sweep.log`` and ``sweep_status.json``. After all workers end, the summary is
computed from the workers' ``metrics.json`` files (``summary.json``,
``summary.csv``).

The selection rule lives inside each run (see the task modules); the sweep
aggregates repetitions and does not select a candidate, so
``selected_candidates`` is empty.

A sweep can hold several variants (e.g. tickers or regimes), each run for
every seed under one ``--max-parallel`` cap; ``summary.json`` then has one
section per variant.

Usage (from the repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/sweep.py \\
        --config configs/e1_original.json --name e1_kchol --seeds 0 1 2 3 4 \\
        --max-parallel 5 -- --ticker KCHOL.IS

    .venv/bin/python papers/QML_financial_benchmark/utils/sweep.py --name e5_stage1 \\
        --seeds 0 --max-parallel 4 \\
        --variant F1 configs/e5_original.json "--regime F1" \\
        --variant F2 configs/e5_original.json "--regime F2"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

import pandas as pd  # noqa: E402
from lib.experiment_logging import git_state, sweep_progress, utc_now, write_json  # noqa: E402
from lib.summary import aggregate, collect_test_metrics, selected_candidates  # noqa: E402

logger = logging.getLogger("sweep")
POLL_SECONDS = 20


def configure_sweep_logging(log_path: Path) -> None:
    """Coordinator-only file and console logging, in UTC."""
    handlers = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ]
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def find_run_dir(seed_dir: Path) -> Path | None:
    """The single ``run_*`` directory a worker created, if it exists yet."""
    candidates = sorted(seed_dir.glob("run_*"))
    if len(candidates) > 1:
        raise RuntimeError(
            f"Expected one run directory in {seed_dir}, found {len(candidates)}"
        )
    return candidates[0] if candidates else None


def refresh_entry(entry: dict, sweep_dir: Path) -> None:
    """Fill the run paths and progress of one sweep entry from its worker's files."""
    run_dir = find_run_dir(sweep_dir / entry["_seed_dir"])
    if run_dir is None:
        return
    relative = run_dir.relative_to(sweep_dir).as_posix()
    entry.update(
        run_id=run_dir.name,
        run_dir=relative,
        run_log=f"{relative}/run.log",
        run_status=f"{relative}/run_status.json",
        config_path=f"{relative}/config_snapshot.json",
        metrics_path=f"{relative}/metrics.json",
    )
    status_file = run_dir / "run_status.json"
    if status_file.is_file():
        progress = json.loads(status_file.read_text()).get("progress")
        if progress:
            entry["progress"] = progress


def public(status: dict) -> dict:
    """Sweep status without coordinator-private keys."""
    return {
        **status,
        "runs": [
            {key: value for key, value in run.items() if not key.startswith("_")}
            for run in status["runs"]
        ],
    }


def parse_variants(args: argparse.Namespace) -> list[dict]:
    """Variants as ``{label, config, extra}``; without ``--variant``, one from ``--config`` and the trailing flags."""
    if args.variant:
        return [
            {"label": label, "config": config, "extra": shlex.split(extra)}
            for label, config, extra in args.variant
        ]
    if not args.config:
        raise SystemExit(
            "Give --config (with optional trailing flags) or at least one --variant"
        )
    return [
        {
            "label": "main",
            "config": args.config,
            "extra": [arg for arg in args.run_args if arg != "--"],
        }
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", help="Experiment config, relative to the paper folder"
    )
    parser.add_argument(
        "--variant",
        nargs=3,
        action="append",
        metavar=("LABEL", "CONFIG", "EXTRA_ARGS"),
        help="One planned variant: label, config and a quoted string of extra run flags. Repeatable.",
    )
    parser.add_argument(
        "--name", required=True, help="Short sweep name, used in the sweep ID"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-parallel", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "run_args",
        nargs=argparse.REMAINDER,
        help="Extra run flags after '--' (single-variant form)",
    )
    args = parser.parse_args(argv)
    variants = parse_variants(args)

    sweep_id = f"{args.name}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    sweep_dir = PROJECT_DIR / "outdir" / "sweeps" / sweep_id
    sweep_dir.mkdir(parents=True)
    configure_sweep_logging(sweep_dir / "sweep.log")
    runs = []
    for variant in variants:
        candidate = " ".join([variant["config"], *variant["extra"]])
        for repetition, seed in enumerate(args.seeds):
            runs.append(
                {
                    "run_id": None,
                    "candidate": candidate,
                    "seed": seed,
                    "repetition": repetition,
                    "status": "PLANNED",
                    "run_dir": None,
                    "run_log": None,
                    "run_status": None,
                    "config_path": None,
                    "metrics_path": None,
                    "_variant": variant,
                    "_seed_dir": f"{variant['label']}/seed{seed}",
                }
            )
    status = {
        "schema_version": 1,
        "sweep_id": sweep_id,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "expected_runs": len(runs),
        "runs": runs,
        "summary_path": None,
        "selected_candidates": [],
        "error": None,
    }
    write_json(sweep_dir / "sweep_status.json", public(status))
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | variants=%s | seeds=%s",
        sweep_id,
        len(runs),
        ",".join(variant["label"] for variant in variants),
        args.seeds,
    )
    try:
        pending = list(runs)
        running: dict[int, tuple[subprocess.Popen, dict]] = {}
        while pending or running:
            while pending and len(running) < args.max_parallel:
                entry = pending.pop(0)
                variant = entry["_variant"]
                seed_dir = sweep_dir / entry["_seed_dir"]
                seed_dir.mkdir(parents=True)
                command = [
                    args.python,
                    "implementation.py",
                    "--paper",
                    "QML_financial_benchmark",
                    "--config",
                    variant["config"],
                    "--seed",
                    str(entry["seed"]),
                    "--outdir",
                    str(seed_dir),
                    "--num-threads",
                    "1",
                    "--sweep-id",
                    sweep_id,
                    "--candidate",
                    entry["candidate"],
                    "--repetition",
                    str(entry["repetition"]),
                    *variant["extra"],
                ]
                console = (seed_dir / "console.log").open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command, cwd=REPO_ROOT, stdout=console, stderr=subprocess.STDOUT
                )
                entry.update(status="RUNNING", _command=shlex.join(command))
                running[process.pid] = (process, entry)
                logger.info(
                    "CANDIDATE_STARTED | variant=%s | seed=%d | repetition=%d | pid=%d | command=%s",
                    variant["label"],
                    entry["seed"],
                    entry["repetition"],
                    process.pid,
                    entry["_command"],
                )
                write_json(sweep_dir / "sweep_status.json", public(status))
                # Distinct run_<timestamp> names need launches at least a second apart.
                time.sleep(1.1)
            time.sleep(POLL_SECONDS)
            for pid, (process, entry) in list(running.items()):
                refresh_entry(entry, sweep_dir)
                if process.poll() is None:
                    continue
                del running[pid]
                refresh_entry(entry, sweep_dir)
                run_status_file = (
                    sweep_dir / entry["run_status"] if entry["run_status"] else None
                )
                worker_status = (
                    json.loads(run_status_file.read_text())["status"]
                    if run_status_file and run_status_file.is_file()
                    else None
                )
                if (
                    process.returncode == 0
                    and worker_status == "COMPLETED"
                    and (sweep_dir / entry["metrics_path"]).is_file()
                ):
                    entry["status"] = "COMPLETED"
                    entry.pop("progress", None)
                    logger.info(
                        "CANDIDATE_COMPLETED | run_id=%s | variant=%s | seed=%d | metrics=%s",
                        entry["run_id"],
                        entry["_variant"]["label"],
                        entry["seed"],
                        entry["metrics_path"],
                    )
                else:
                    entry["status"] = "FAILED"
                    logger.info(
                        "CANDIDATE_FAILED | run_id=%s | variant=%s | seed=%d | returncode=%s | worker_status=%s | console=%s/console.log",
                        entry["run_id"],
                        entry["_variant"]["label"],
                        entry["seed"],
                        process.returncode,
                        worker_status,
                        entry["_seed_dir"],
                    )
            write_json(sweep_dir / "sweep_status.json", public(status))
            logger.info("SWEEP_PROGRESS | %s", sweep_progress(public(status)))

        summaries, tables = {}, []
        for variant in variants:
            completed = {
                entry["seed"]: sweep_dir / entry["metrics_path"]
                for entry in runs
                if entry["_variant"] is variant and entry["status"] == "COMPLETED"
            }
            if not completed:
                continue
            long_table = collect_test_metrics(completed)
            summary_table = aggregate(long_table).assign(variant=variant["label"])
            tables.append(summary_table)
            summaries[variant["label"]] = {
                "candidate": " ".join([variant["config"], *variant["extra"]]),
                "seeds_completed": sorted(completed),
                "seeds_planned": args.seeds,
                "coverage": "COMPLETE"
                if len(completed) == len(args.seeds)
                else "PARTIAL",
                "test_metrics": summary_table.to_dict(orient="records"),
                "selected_candidates_per_seed": selected_candidates(
                    long_table
                ).to_dict(),
            }
        n_completed = sum(entry["status"] == "COMPLETED" for entry in runs)
        if summaries:
            pd.concat(tables).to_csv(sweep_dir / "summary.csv", index=False)
            write_json(
                sweep_dir / "summary.json",
                {"sweep_id": sweep_id, "variants": summaries},
            )
            status["summary_path"] = "summary.json"
        status["completed_at"] = utc_now()
        if n_completed == len(runs):
            status["status"] = "COMPLETED"
            write_json(sweep_dir / "sweep_status.json", public(status))
            logger.info(
                "SWEEP_COMPLETED | summary=summary.json | selected=none (repetition sweep)"
            )
        elif n_completed:
            status["status"] = "PARTIAL"
            write_json(sweep_dir / "sweep_status.json", public(status))
            logger.info(
                "SWEEP_PARTIAL | summary=summary.json | completed=%d/%d",
                n_completed,
                len(runs),
            )
        else:
            raise RuntimeError("No sweep run completed")
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(sweep_dir / "sweep_status.json", public(status))
        logger.exception("SWEEP_FAILED | sweep_id=%s", sweep_id)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
