"""Run a fixed-budget task x model x seed comparison and summarise it.

Every planned run is recorded in ``sweep_status.json`` whether it completes or
fails, and ``summary.json`` aggregates test metrics as mean and sample SD.
Run directories are stored relative to the sweep directory so the sweep can be
moved or rendered on another machine.

Example (from the repository root)::

    python papers/QCHIPP/utils/run_comparison.py --epochs 100 --seeds 49 50 51 \
        --models merlin_reference classical_reference \
        --output papers/QCHIPP/outdir/comparison_reference
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for import_root in (REPO_ROOT, PROJECT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from lib.artifacts import git_state, utc_now, write_json  # noqa: E402

from runtime_lib import run_from_project  # noqa: E402

LOGGER = logging.getLogger("qchipp.comparison")

# Config file stem per model; the binding logistic control is defaults.json.
CONFIG_STEMS = {
    "gate_source": "original",
    "gate_corrected": "corrected",
    "merlin": "merlin",
    "merlin_reference": "merlin_reference",
    "classical": "classical",
    "classical_reference": "classical_reference",
    "logistic": "logistic",
    "logistic_onehot": "logistic_onehot",
    "rf": "rf",
    "rf_onehot": "rf_onehot",
}
DEFAULT_MODELS = [
    "gate_source",
    "gate_corrected",
    "merlin",
    "classical",
    "rf",
    "logistic",
]
SUMMARY_METRICS = ("f1", "auroc", "accuracy", "average_precision")


def config_for(task: str, model: str) -> str:
    """Return the config path (relative to the paper) for one task/model pair."""
    if task == "binding" and model == "logistic":
        return "configs/defaults.json"
    return f"configs/{task}_{CONFIG_STEMS[model]}.json"


def check_completed_run(run_dir: Path, task: str, model: str, seed: int) -> dict:
    """Load a run's metrics and confirm the run completed as planned.

    Raises
    ------
    ValueError
        If the run status is not COMPLETED, the metrics belong to another
        candidate, or the held-out metrics are not finite.
    """
    status = json.loads((run_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{run_dir} finished with status {status['status']}")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    if (metrics["task"], metrics["model"], metrics["seed"]) != (task, model, seed):
        raise ValueError(f"{run_dir} metrics do not match the planned candidate")
    if not (
        np.isfinite(metrics["test"]["f1"]) and np.isfinite(metrics["test"]["auroc"])
    ):
        raise ValueError(f"{run_dir} has nonfinite held-out metrics")
    return metrics


def summarise(collected: list[dict], tasks, models, seeds) -> list[dict]:
    """Aggregate completed runs per task/model as mean and sample SD."""
    rows = []
    for task, model in itertools.product(tasks, models):
        group = [m for m in collected if m["task"] == task and m["model"] == model]
        row = {
            "task": task,
            "model": model,
            "completed_runs": len(group),
            "expected_runs": len(seeds),
            "status": "COMPLETE" if len(group) == len(seeds) else "PARTIAL",
            "seeds": [m["seed"] for m in group],
            "parameters": group[0]["parameters"] if group else None,
            "mean_wall_seconds": (
                float(np.mean([m["wall_seconds"] for m in group])) if group else None
            ),
        }
        for metric in SUMMARY_METRICS:
            values = [m["test"][metric] for m in group]
            row[metric] = (
                {
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0,
                }
                if values
                else None
            )
        rows.append(row)
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[49, 50, 51])
    parser.add_argument(
        "--models", choices=list(CONFIG_STEMS), nargs="+", default=DEFAULT_MODELS
    )
    parser.add_argument(
        "--tasks",
        choices=["binding", "immunogenic"],
        nargs="+",
        default=["binding", "immunogenic"],
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Shared data root; defaults to the runtime convention (DATA_DIR, then data/).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for option in ("seeds", "models", "tasks"):
        values = getattr(args, option)
        if len(set(values)) != len(values):
            parser.error(f"--{option} values must be unique")
    return args


def main(argv=None) -> Path:
    """Execute every planned candidate and write the sweep ledger and summary.

    Returns
    -------
    Path
        The sweep output directory.
    """
    args = parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "sweep.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

    planned = list(itertools.product(args.tasks, args.models, args.seeds))
    write_json(
        output / "plan.json",
        {
            "epochs": args.epochs,
            "seeds": args.seeds,
            "tasks": args.tasks,
            "models": args.models,
            "expected_runs": len(planned),
            "selection": "last fixed-budget checkpoint; test metrics are not used for selection",
            "aggregation": "mean and sample SD over seeds",
        },
    )
    status = {
        "schema_version": 2,
        "sweep_id": output.name,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(PROJECT_DIR),
        "expected_runs": len(planned),
        "runs": [],
        "error": None,
    }
    write_json(output / "sweep_status.json", status)
    LOGGER.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d", output.name, len(planned)
    )

    original_cwd = Path.cwd()
    collected = []
    try:
        for task, model, seed in planned:
            candidate_dir = output / f"{task}_{model}_{seed}"
            item = {
                "run_id": candidate_dir.name,
                "candidate": {"task": task, "model": model},
                "seed": seed,
                "repetition": args.seeds.index(seed),
                "status": "RUNNING",
                "run_dir": None,
            }
            status["runs"].append(item)
            write_json(output / "sweep_status.json", status)
            LOGGER.info("CANDIDATE_STARTED | run_id=%s", item["run_id"])
            argv_run = [
                "--config", config_for(task, model),
                "--epochs", str(args.epochs),
                "--seed", str(seed),
                "--outdir", str(candidate_dir),
                "--sweep-id", output.name,
                "--run-id", item["run_id"],
                "--repetition", str(item["repetition"]),
            ]  # fmt: skip
            if args.data_root is not None:
                argv_run += ["--data-root", str(args.data_root.resolve())]
            try:
                run_dir = run_from_project(PROJECT_DIR, argv_run)
                metrics = check_completed_run(run_dir, task, model, seed)
                relative_run_dir = run_dir.resolve().relative_to(output).as_posix()
                item.update(status="COMPLETED", run_dir=relative_run_dir)
                collected.append({"run_dir": relative_run_dir, **metrics})
                LOGGER.info("CANDIDATE_COMPLETED | run_id=%s", item["run_id"])
            except Exception as error:
                item.update(
                    status="FAILED",
                    error={"type": type(error).__name__, "message": str(error)},
                )
                LOGGER.exception("CANDIDATE_FAILED | run_id=%s", item["run_id"])
            finally:
                os.chdir(original_cwd)
            write_json(output / "sweep_status.json", status)

        observed = {(m["task"], m["model"], m["seed"]) for m in collected}
        complete = observed == set(planned)
        write_json(
            output / "summary.json",
            {
                "scope": "partial release-split, fixed-budget, no winner selected",
                "status": "COMPLETE COMPARISON" if complete else "PARTIAL COMPARISON",
                "planned_count": len(planned),
                "completed_count": len(collected),
                "rows": summarise(collected, args.tasks, args.models, args.seeds),
                "runs": collected,
            },
        )
        status.update(
            status="COMPLETED" if complete else "PARTIAL", completed_at=utc_now()
        )
        write_json(output / "sweep_status.json", status)
        LOGGER.info("SWEEP_%s | summary=%s", status["status"], output / "summary.json")
    except Exception as error:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(error).__name__, "message": str(error)},
        )
        write_json(output / "sweep_status.json", status)
        LOGGER.exception("SWEEP_FAILED")
        raise
    finally:
        os.chdir(original_cwd)
        LOGGER.removeHandler(handler)
        handler.close()
    return output


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
