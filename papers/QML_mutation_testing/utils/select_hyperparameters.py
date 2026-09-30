#!/usr/bin/env python3
"""Small pre-declared grid over the hyperparameters the paper leaves unspecified.

The paper states only the software stack and the model family; it reports no
optimizer, learning rate, epoch count, or ansatz depth. This coordinator runs a
pre-declared grid and selects a candidate on *validation* accuracy, so that the
choices recorded in ``LOG.md`` are auditable rather than hand-picked.

Sweep plan (declared in ``LOG.md`` before launching):

- varied: ``model.params.n_layers`` and ``training.lr``;
- fixed: everything else in ``configs/defaults.json``;
- repetitions: one run per (candidate, seed) over three model-init seeds;
- selection metric: validation accuracy, maximised, final epoch, averaged over
  seeds;
- tie tolerance: 0.005 absolute, broken by fewer layers then by smaller lr.

Usage
-----
``python utils/select_hyperparameters.py [--outdir outdir/sweeps]``
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import uuid
from itertools import product
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
for _path in (PROJECT_DIR.parents[1], PROJECT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    utc_now,
    write_json,
)
from lib.runner import REPO_ROOT, SCHEMA_VERSION, train_and_evaluate  # noqa: E402

logger = logging.getLogger("sweep")

LAYER_GRID = (2, 3, 4, 6)
LR_GRID = (0.02, 0.05, 0.10)
SEEDS = (42, 43, 44)
SELECTION = {
    "name": "val_accuracy",
    "direction": "maximize",
    "split": "validation",
    "checkpoint": "final epoch",
    "aggregation": "mean over seeds",
    "tie_tolerance": 0.005,
    "tie_break": "fewer layers, then smaller lr",
}


def _candidates() -> list[dict]:
    """Return the pre-declared candidate list, in a stable order."""

    return [
        {"n_layers": layers, "lr": lr} for layers, lr in product(LAYER_GRID, LR_GRID)
    ]


def _candidate_key(candidate: dict) -> str:
    """Return a filesystem-safe identifier for a candidate."""

    return f"L{candidate['n_layers']}_lr{candidate['lr']:g}"


def _candidate_config(
    base: dict,
    candidate: dict,
    seed: int,
    *,
    run_id: str,
    sweep_id: str,
    repetition: int,
) -> dict:
    """Return a resolved config for one (candidate, seed) run."""

    cfg = copy.deepcopy(base)
    cfg["model"]["params"]["n_layers"] = candidate["n_layers"]
    cfg["training"]["lr"] = candidate["lr"]
    cfg["experiment"]["seeds"] = [seed]
    # Keeps run_status.json consistent with the sweep's own record.
    cfg["run_context"] = {
        "run_id": run_id,
        "sweep_id": sweep_id,
        "candidate": candidate,
        "repetition": repetition,
    }
    return cfg


def _run_candidate(cfg: dict, run_dir: Path) -> bool:
    """Execute one candidate run inside its own log file.

    Parameters
    ----------
    cfg : dict
        Resolved configuration for this run.
    run_dir : Path
        Directory receiving the run artifacts.

    Returns
    -------
    bool
        Whether the run completed successfully.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )

    # Give this run its own run.log while keeping the sweep log attached.
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().addHandler(handler)
    try:
        train_and_evaluate(cfg, run_dir)
        return True
    except Exception:  # noqa: BLE001 - recorded by the runner, reported by the sweep
        return False
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _summarise(runs: list[dict], sweep_dir: Path) -> tuple[list[dict], list[dict]]:
    """Aggregate candidate results from the structured per-run metrics.

    Rankings are computed from ``metrics.json`` files, never from log text.

    Returns
    -------
    tuple[list[dict], list[dict]]
        The per-candidate summary rows (sorted by the selection rule) and the
        rows tied with the best one within the declared tolerance.
    """

    by_candidate: dict[str, dict] = {}
    for run in runs:
        key = _candidate_key(run["candidate"])
        row = by_candidate.setdefault(
            key,
            {
                "candidate_key": key,
                "candidate": run["candidate"],
                "expected_runs": len(SEEDS),
                "successful_runs": 0,
                "val_accuracies": [],
                "test_accuracies": [],
                "trainable_parameters": None,
            },
        )
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads(
            (sweep_dir / run["metrics_path"]).read_text(encoding="utf-8")
        )
        row["successful_runs"] += 1
        row["val_accuracies"].append(metrics["aggregate"]["val_accuracy"]["mean"])
        row["test_accuracies"].append(metrics["aggregate"]["test_accuracy"]["mean"])
        row["trainable_parameters"] = metrics["parameter_budget"][
            "photonic_trainable_phases"
        ]

    rows = []
    for row in by_candidate.values():
        complete = row["successful_runs"] == row["expected_runs"]
        values = row["val_accuracies"]
        row["val_accuracy_mean"] = sum(values) / len(values) if values else float("nan")
        row["test_accuracy_mean"] = (
            sum(row["test_accuracies"]) / len(row["test_accuracies"])
            if row["test_accuracies"]
            else float("nan")
        )
        row["status"] = "COMPLETE" if complete else "INCOMPLETE"
        rows.append(row)

    comparable = [row for row in rows if row["status"] == "COMPLETE"]
    comparable.sort(
        key=lambda row: (
            -row["val_accuracy_mean"],
            row["candidate"]["n_layers"],
            row["candidate"]["lr"],
        )
    )
    tied: list[dict] = []
    if comparable:
        best = comparable[0]["val_accuracy_mean"]
        tied = [
            row
            for row in comparable
            if best - row["val_accuracy_mean"] <= SELECTION["tie_tolerance"]
        ]
    rows.sort(key=lambda row: (-row["val_accuracy_mean"], row["candidate_key"]))
    return rows, tied


def main() -> int:
    """Run the pre-declared grid and write the sweep artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="outdir/sweeps", type=Path)
    args = parser.parse_args()

    sweep_id = f"hparam-{uuid.uuid4().hex[:8]}"
    sweep_dir = (PROJECT_DIR / args.outdir / sweep_id).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(sweep_dir / "sweep.log")

    base = json.loads(
        (PROJECT_DIR / "configs" / "defaults.json").read_text(encoding="utf-8")
    )
    candidates = _candidates()
    expected_runs = len(candidates) * len(SEEDS)

    status = {
        "schema_version": SCHEMA_VERSION,
        "sweep_id": sweep_id,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "expected_runs": expected_runs,
        "runs": [],
        "summary_path": None,
        "selected_candidates": [],
        "selection": SELECTION,
        "error": None,
    }
    write_json(sweep_dir / "sweep_status.json", status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | layers=%s | lr=%s | seeds=%s",
        sweep_id,
        expected_runs,
        list(LAYER_GRID),
        list(LR_GRID),
        list(SEEDS),
    )

    for candidate in candidates:
        for repetition, seed in enumerate(SEEDS, start=1):
            run_id = f"{_candidate_key(candidate)}_s{seed}"
            run_dir = sweep_dir / run_id
            record = {
                "run_id": run_id,
                "candidate": candidate,
                "seed": seed,
                "repetition": repetition,
                "status": "RUNNING",
                "run_dir": run_id,
                "run_log": f"{run_id}/run.log",
                "run_status": f"{run_id}/run_status.json",
                "config_path": f"{run_id}/config_snapshot.json",
                "metrics_path": f"{run_id}/metrics.json",
            }
            status["runs"].append(record)
            write_json(sweep_dir / "sweep_status.json", status)
            logger.info(
                "CANDIDATE_STARTED | run_id=%s | candidate=%s | seed=%d",
                run_id,
                candidate,
                seed,
            )

            ok = _run_candidate(
                _candidate_config(
                    base,
                    candidate,
                    seed,
                    run_id=run_id,
                    sweep_id=sweep_id,
                    repetition=repetition,
                ),
                run_dir,
            )
            record["status"] = "COMPLETED" if ok else "FAILED"
            write_json(sweep_dir / "sweep_status.json", status)
            if ok:
                logger.info(
                    "CANDIDATE_COMPLETED | run_id=%s | metrics=%s",
                    run_id,
                    record["metrics_path"],
                )
            else:
                logger.error("CANDIDATE_FAILED | run_id=%s", run_id)

    rows, tied = _summarise(status["runs"], sweep_dir)
    summary_path = sweep_dir / "sweep_summary.json"
    write_json(
        summary_path,
        {
            "sweep_id": sweep_id,
            "selection": SELECTION,
            "candidates": rows,
            "tied_candidates": [row["candidate_key"] for row in tied],
        },
    )
    _write_summary_csv(sweep_dir / "sweep_summary.csv", rows)

    completed = sum(1 for run in status["runs"] if run["status"] == "COMPLETED")
    is_complete = completed == expected_runs
    status.update(
        status="COMPLETED" if is_complete else "PARTIAL",
        completed_at=utc_now(),
        summary_path=summary_path.name,
        selected_candidates=[row["candidate_key"] for row in tied[:1]],
    )
    write_json(sweep_dir / "sweep_status.json", status)

    event = "SWEEP_COMPLETED" if is_complete else "SWEEP_PARTIAL"
    logger.info(
        "%s | summary=%s | selected=%s | tied=%s | completed_runs=%d/%d",
        event,
        summary_path.name,
        status["selected_candidates"],
        [row["candidate_key"] for row in tied],
        completed,
        expected_runs,
    )
    print(sweep_dir)
    return 0 if is_complete else 1


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    """Write the candidate table as CSV for quick inspection."""

    header = (
        "candidate_key,n_layers,lr,trainable_parameters,successful_runs,"
        "expected_runs,val_accuracy_mean,test_accuracy_mean,status\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['candidate_key']},{row['candidate']['n_layers']},"
            f"{row['candidate']['lr']},{row['trainable_parameters']},"
            f"{row['successful_runs']},{row['expected_runs']},"
            f"{row['val_accuracy_mean']:.6f},{row['test_accuracy_mean']:.6f},"
            f"{row['status']}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
