"""Coordinator of the ``moons_assignment_rules`` sensitivity sweep (E8).

Usage (from the repository root or the paper folder)::

    python papers/adaptive_boson_sampling/utils/run_moons_assignment_sweep.py

The plan is recorded in ``LOG.md`` ("Sweep Plan: moons_assignment_rules"). Each
candidate is one ``rank_rule`` x ``noise`` pair, run once through the shared
runtime with the 9 moons dataset variants as in-run repetitions. The coordinator
alone writes ``outdir/sweep_moons_assignment_rules/{sweep.log,
sweep_status.json}``, then generates ``summary.csv`` and ``summary.json`` from
the candidates' ``metrics.json``. No candidate is selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
# The paper folder provides lib/; the repository root provides runtime_lib/.
sys.path[:0] = [str(PROJECT_DIR), str(REPO_ROOT)]

from lib.data import CENTROID_RANK_RULES  # noqa: E402
from lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    utc_now,
    write_json,
)

SWEEP_ID = "moons_assignment_rules"
NOISES = (0.1, 0.15, 0.2)
BASE_CONFIG = PROJECT_DIR / "configs" / "classify_moons_xorder_original.json"
PAPER = {"qubit": 0.65, "qutrit": 0.90}
MAX_ATTEMPTS = 2
logger = logging.getLogger("sweep")


def candidate_config(rule: str, noise: float) -> dict:
    config = json.loads(BASE_CONFIG.read_text())
    config = deepcopy(config)
    config["description"] = (
        f"Sweep {SWEEP_ID}: moons clusters ranked by centroid rule {rule!r}, "
        f"noise {noise}, 3 data seeds x 3 K-means seeds, paper arm."
    )
    config["validity"] = (
        "substitute-data (regenerated moons; cluster-to-outcome rule unpublished, "
        "sensitivity sweep)"
    )
    config["dataset"].update(
        {"assignment": "centroid_rank", "rank_rule": rule, "noise": noise}
    )
    config["evaluation"]["arms"] = ["paper"]
    config["sweep_id"] = SWEEP_ID
    config["candidate"] = {"rank_rule": rule, "noise": noise}
    config["repetition"] = 0
    return config


def launch(config_path: Path) -> tuple[Path | None, str]:
    command = [
        sys.executable, str(REPO_ROOT / "implementation.py"),
        "--paper", "adaptive_boson_sampling",
        "--config", str(config_path), "--seed", "42",
    ]  # fmt: skip
    completed = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"Artifacts in: (\S+)", output)
    if completed.returncode != 0 or match is None:
        return None, output[-4000:]
    return PROJECT_DIR / match.group(1), output


def summarize(sweep_dir: Path, runs: list[dict]) -> tuple[Path, int]:
    """Write summary.csv/json from candidate metrics; return path and row count."""
    rows = []
    for run in runs:
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads(Path(run["metrics_path"]).read_text())
        for label, result in metrics["results"].items():
            accuracies = np.array(
                [
                    inst["mean_test_accuracy"]
                    for inst in result["arms"]["paper"]["instances"]
                ]
            )
            if accuracies.size != 9 or not np.all(np.isfinite(accuracies)):
                raise ValueError(f"{run['run_id']} {label}: expected 9 finite variants")
            rows.append({
                "rank_rule": run["candidate"]["rank_rule"],
                "noise": run["candidate"]["noise"],
                "model": label,
                "n_variants": int(accuracies.size),
                "mean_test_accuracy": float(accuracies.mean()),
                "std_over_variants": float(accuracies.std(ddof=1)),
                "paper_accuracy": PAPER["qubit" if label.startswith("B2") else "qutrit"],
                "run_id": run["run_id"],
            })  # fmt: skip
    csv_path = sweep_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(sweep_dir / "summary.json", {"sweep_id": SWEEP_ID, "rows": rows})
    return csv_path, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir", type=Path, default=PROJECT_DIR / "outdir" / f"sweep_{SWEEP_ID}"
    )
    args = parser.parse_args()
    sweep_dir = args.sweep_dir.resolve()
    if sweep_dir.exists():
        raise FileExistsError(f"{sweep_dir} exists; move it before relaunching")
    configure_logging(sweep_dir / "sweep.log", console=True)
    candidates = [(rule, noise) for noise in NOISES for rule in CENTROID_RANK_RULES]
    status = {
        "schema_version": 1, "sweep_id": SWEEP_ID, "status": "RUNNING",
        "started_at": utc_now(), "completed_at": None,
        "code": git_state(REPO_ROOT), "expected_runs": len(candidates),
        "runs": [], "summary_path": None, "selected_candidates": [],
        "selection_rule": "none (sensitivity sweep; every candidate reported)",
        "error": None,
        "progress": {"current": 0, "total": len(candidates), "unit": "candidates"},
    }  # fmt: skip
    write_json(sweep_dir / "sweep_status.json", status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d", SWEEP_ID, len(candidates)
    )
    try:
        for index, (rule, noise) in enumerate(candidates, start=1):
            config_path = sweep_dir / "configs" / f"candidate_{index:02d}.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(candidate_config(rule, noise), indent=2))
            logger.info("CANDIDATE_STARTED | %d/%d | rank_rule=%s | noise=%s",
                        index, len(candidates), rule, noise)  # fmt: skip
            run_dir, attempts = None, []
            for attempt in range(1, MAX_ATTEMPTS + 1):
                run_dir, output = launch(config_path)
                attempts.append({"attempt": attempt, "run_dir": str(run_dir)})
                if run_dir is not None:
                    break
                logger.warning("CANDIDATE_ATTEMPT_FAILED | attempt=%d | tail=%s",
                               attempt, output[-500:])  # fmt: skip
            entry = {
                "candidate": {"rank_rule": rule, "noise": noise},
                "seed": 42, "repetition": 0, "attempts": attempts,
                "config_path": str(config_path),
            }  # fmt: skip
            if run_dir is None:
                entry.update({"run_id": f"candidate_{index:02d}", "status": "FAILED",
                              "run_dir": None, "run_log": None, "run_status": None,
                              "metrics_path": None})  # fmt: skip
                logger.error("CANDIDATE_FAILED | rank_rule=%s | noise=%s", rule, noise)
            else:
                entry.update({
                    "run_id": run_dir.name, "status": "COMPLETED",
                    "run_dir": str(run_dir), "run_log": str(run_dir / "run.log"),
                    "run_status": str(run_dir / "run_status.json"),
                    "config_path": str(run_dir / "config_snapshot.json"),
                    "metrics_path": str(run_dir / "metrics.json"),
                })  # fmt: skip
                logger.info("CANDIDATE_COMPLETED | rank_rule=%s | noise=%s | run=%s",
                            rule, noise, run_dir.name)  # fmt: skip
            status["runs"].append(entry)
            status["progress"]["current"] = index
            write_json(sweep_dir / "sweep_status.json", status)
        summary_path, n_rows = summarize(sweep_dir, status["runs"])
        completed = sum(run["status"] == "COMPLETED" for run in status["runs"])
        status["summary_path"] = str(summary_path)
        status["completed_at"] = utc_now()
        if completed == len(candidates):
            status["status"] = "COMPLETED"
            write_json(sweep_dir / "sweep_status.json", status)
            logger.info(
                "SWEEP_COMPLETED | runs=%d | summary_rows=%d", completed, n_rows
            )
        else:
            status["status"] = "PARTIAL"
            write_json(sweep_dir / "sweep_status.json", status)
            logger.info("SWEEP_PARTIAL | completed=%d/%d", completed, len(candidates))
    except Exception as exc:
        status.update({
            "status": "FAILED", "completed_at": utc_now(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        })  # fmt: skip
        write_json(sweep_dir / "sweep_status.json", status)
        logger.error("SWEEP_FAILED | %s\n%s", exc, traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
