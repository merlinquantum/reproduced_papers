"""Sweep coordinator: run every (candidate, seed) of a sweep spec, one process per run.

Usage (from the repository root, locally or as one remote launch)::

    python papers/photonic_MMD_generative/utils/run_sweep.py configs/sweeps/<sweep>.json \
        [--device cuda:0] [--data-root ../../data/<folder>]

Sweep spec (JSON, under ``configs/sweeps/``)::

    {"sweep_id": "...", "description": "...", "seeds": [0, 1, 2, 3, 4],
     "candidates": [{"label": "...", "config": "configs/<x>.json",
                     "overrides": {"kernel.sigma": 2.0, ...}}, ...]}

For each run the coordinator writes an overlay config (the candidate's config
+ dotted overrides + sweep metadata) under ``<sweep_dir>/run_configs/`` and
starts ``implementation.py`` with ``--outdir <sweep_dir>/runs``. The
coordinator alone writes ``sweep.log`` and ``sweep_status.json`` (rewritten at
every transition and progress check). Failed runs are recorded, not retried:
a second attempt needs a recorded cause (LOG.md, sweep ledger).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(PAPER_DIR / "utils"))

from lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)
from summarize_sweep import write_summary  # noqa: E402

from runtime_lib.config import deep_update, load_config  # noqa: E402

logger = logging.getLogger("sweep")


def code_fingerprint() -> dict:
    """SHA-256 of every file under lib/, utils/, configs/ (the code a sweep actually ran)."""
    import hashlib

    result = {}
    for name in ("lib", "utils", "configs"):
        combined = hashlib.sha256()
        for path in sorted(
            q
            for q in (PAPER_DIR / name).rglob("*")
            if q.is_file() and "__pycache__" not in q.parts
        ):
            combined.update(
                f"{path.relative_to(PAPER_DIR)}:{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode()
            )
        result[name] = combined.hexdigest()
    return result


PROGRESS_SECONDS = 30


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    node = cfg
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def main() -> int:
    """Run the sweep described by the spec; return 0 unless the sweep failed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "spec",
        help="Sweep spec, relative to the paper folder (configs/sweeps/<name>.json)",
    )
    parser.add_argument(
        "--device", default=None, help="Passed to every run (e.g. cuda:0)"
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Passed to every run (relative to the paper folder)",
    )
    args = parser.parse_args()

    spec = json.loads((PAPER_DIR / args.spec).read_text())
    sweep_id = spec["sweep_id"]
    sweep_dir = (
        PAPER_DIR / "outdir" / f"sweep_{sweep_id}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    (sweep_dir / "run_configs").mkdir(parents=True)
    (sweep_dir / "runs").mkdir()
    configure_logging(sweep_dir / "sweep.log")
    plan = [
        (c, seed, rep)
        for c in spec["candidates"]
        for rep, seed in enumerate(spec["seeds"])
    ]
    status = {
        "schema_version": 1,
        "sweep_id": sweep_id,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "expected_runs": len(plan),
        "runs": [],
        "summary_path": "summary.json",
        "selected_candidates": [],
        "error": None,
        "spec": spec,
        "code_fingerprint": code_fingerprint(),
        "device": args.device,
        "data_root": args.data_root,
    }
    status_path = sweep_dir / "sweep_status.json"
    write_json(status_path, status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | dir=%s",
        sweep_id,
        len(plan),
        sweep_dir,
    )
    try:
        for index, (candidate, seed, repetition) in enumerate(plan):
            label = candidate["label"]
            candidate_values = {
                "label": label,
                "config": candidate["config"],
                **candidate.get("overrides", {}),
            }
            cfg = load_config(PAPER_DIR / candidate["config"])
            for dotted, value in candidate.get("overrides", {}).items():
                _set_dotted(cfg, dotted, value)
            cfg = deep_update(
                cfg,
                {
                    "sweep_id": sweep_id,
                    "candidate": candidate_values,
                    "repetition": repetition,
                },
            )
            overlay = sweep_dir / "run_configs" / f"{index:03d}_{label}_seed{seed}.json"
            overlay.write_text(json.dumps(cfg, indent=2))
            command = [
                sys.executable,
                str(REPO_ROOT / "implementation.py"),
                "--paper",
                PAPER_DIR.name,
                "--config",
                str(overlay),
                "--seed",
                str(seed),
                "--outdir",
                str(sweep_dir / "runs"),
            ]
            if args.device:
                command += ["--device", args.device]
            if args.data_root:
                command += ["--data-root", args.data_root]
            entry = {
                "run_id": None,
                "candidate": candidate_values,
                "seed": seed,
                "repetition": repetition,
                "status": "RUNNING",
                "run_dir": None,
                "run_log": None,
                "run_status": None,
                "config_path": None,
                "metrics_path": None,
                "command": " ".join(command),
            }
            status["runs"].append(entry)
            write_json(status_path, status)
            logger.info(
                "CANDIDATE_STARTED | index=%d/%d | label=%s | seed=%d",
                index + 1,
                len(plan),
                label,
                seed,
            )
            before = set((sweep_dir / "runs").glob("run_*"))
            resolved = deep_update(
                load_config(PAPER_DIR / "configs" / "defaults.json"), cfg
            )
            if resolved["model"]["backend"] == "rbm_tuned":
                # Upper bound: every grid fit plus one refit per sigma.
                tuning = resolved["tuning"]
                total_steps = tuning["n_iter"] * (
                    len(tuning["hidden_units"]) * len(tuning["learning_rates"])
                    + len(tuning["sigmas"])
                )
                progress_unit = "epochs"
            else:
                total_steps = resolved["training"]["steps"]
                progress_unit = "steps"
            with (sweep_dir / "run_configs" / f"{index:03d}_stdout.txt").open(
                "w"
            ) as stdout:
                process = subprocess.Popen(
                    command, cwd=REPO_ROOT, stdout=stdout, stderr=subprocess.STDOUT
                )
                last_progress = time.monotonic()
                while process.poll() is None:
                    time.sleep(2)
                    if time.monotonic() - last_progress < PROGRESS_SECONDS:
                        continue
                    last_progress = time.monotonic()
                    new = sorted(set((sweep_dir / "runs").glob("run_*")) - before)
                    if new and (new[-1] / "run.log").exists():
                        done = len(
                            re.findall(
                                r"TRAIN_(?:STEP|EPOCH)_COMPLETED",
                                (new[-1] / "run.log").read_text(),
                            )
                        )
                        entry["progress"] = {
                            "current": done,
                            "total": total_steps,
                            "unit": progress_unit,
                        }
                        write_json(status_path, status)
                        logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))
            new = sorted(set((sweep_dir / "runs").glob("run_*")) - before)
            run_dir = new[-1] if new else None
            run_status = (
                json.loads((run_dir / "run_status.json").read_text())
                if run_dir and (run_dir / "run_status.json").exists()
                else {}
            )
            entry.pop("progress", None)
            if run_dir is not None:
                rel = run_dir.relative_to(sweep_dir)
                entry.update(
                    run_id=run_dir.name,
                    run_dir=str(rel),
                    run_log=str(rel / "run.log"),
                    run_status=str(rel / "run_status.json"),
                    config_path=str(rel / "config_snapshot.json"),
                    metrics_path=str(rel / "metrics.json"),
                )
            if process.returncode == 0 and run_status.get("status") == "COMPLETED":
                entry["status"] = "COMPLETED"
                logger.info(
                    "CANDIDATE_COMPLETED | run_id=%s | metrics=%s",
                    entry["run_id"],
                    entry["metrics_path"],
                )
            else:
                entry["status"] = "FAILED"
                entry["error"] = run_status.get("error") or {
                    "type": "ProcessExit",
                    "message": f"exit {process.returncode}",
                }
                logger.info(
                    "CANDIDATE_FAILED | run_id=%s | exit=%s | error=%s",
                    entry["run_id"],
                    process.returncode,
                    entry["error"],
                )
            write_json(status_path, status)
            logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))
        write_summary(sweep_dir)
        completed = sum(r["status"] == "COMPLETED" for r in status["runs"])
        status.update(
            status="COMPLETED" if completed == len(plan) else "PARTIAL",
            completed_at=utc_now(),
        )
        write_json(status_path, status)
        event = (
            "SWEEP_COMPLETED" if status["status"] == "COMPLETED" else "SWEEP_PARTIAL"
        )
        logger.info(
            "%s | summary=summary.json | completed=%d/%d | selected=none (comparison sweep)",
            event,
            completed,
            len(plan),
        )
        return 0
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.error("SWEEP_FAILED | %s\n%s", exc, traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
