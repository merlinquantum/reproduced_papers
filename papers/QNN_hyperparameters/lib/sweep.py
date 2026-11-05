"""Resumable sweep coordinator.

A sweep config (``mode: sweep``) holds a base run config plus a ``sweep``
block::

    "sweep": {
        "id": "E-B_mechanism",
        "output_dir": "outdir/sweeps/E-B_mechanism",
        "candidate_mode": "qnn",
        "workers": 8,
        "seeds": [1, 2, 3],
        "factors": [
            {"name": "data", "levels": [{"label": "rice", "set": {"dataset.name": "rice"}}]},
            {"name": "circuit", "levels": "paper_grid"}
        ]
    }

Candidates are the Cartesian product of the factor levels times the seeds.
Each candidate runs in its own process and directory
``<output_dir>/runs/<run_id>/`` with its own ``run.log``; only the
coordinator writes ``sweep.log`` and ``sweep_status.json``. The output
directory is fixed by the config (not timestamped), so relaunching the same
sweep skips candidates whose ``run_status.json`` is already ``COMPLETED``.
That makes the sweep safe to run detached on a remote machine and to resume
after a stop or timeout.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from .circuits import paper_grid
from .experiment_logging import (
    configure_logging,
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
STATUS_REFRESH_SECONDS = 30


def set_dotted(config: dict, dotted_key: str, value) -> None:
    """Set ``config["a"]["b"] = value`` for ``dotted_key == "a.b"``."""

    *parents, leaf = dotted_key.split(".")
    node = config
    for key in parents:
        if key not in node:
            raise KeyError(f"unknown config path {dotted_key!r}")
        node = node[key]
    node[leaf] = value


def _levels(factor: dict) -> list[dict]:
    """Factor levels; ``"paper_grid"`` or ``{"paper_grid": [indices]}`` expand
    to the author's circuit configurations (labels ``cNN`` = released ``index``)."""

    levels = factor["levels"]
    if levels == "paper_grid" or (isinstance(levels, dict) and "paper_grid" in levels):
        grid = paper_grid()
        indices = range(len(grid)) if levels == "paper_grid" else levels["paper_grid"]
        return [
            {
                "label": f"c{index:02d}",
                "set": {f"model.{key}": value for key, value in grid[index].items()},
                "grid_index": index,
            }
            for index in indices
        ]
    return levels


def expand_candidates(cfg: dict) -> list[dict]:
    """Return every candidate as ``{"run_id", "candidate", "seed", "repetition", "config"}``."""

    sweep = cfg["sweep"]
    base = deepcopy(cfg)
    base.pop("sweep")
    base["mode"] = sweep["candidate_mode"]
    factors = sweep["factors"]
    level_lists = [_levels(factor) for factor in factors]
    candidates = []
    for combination in itertools.product(*level_lists):
        for repetition, seed in enumerate(sweep["seeds"]):
            config = deepcopy(base)
            values = {}
            for factor, level in zip(factors, combination):
                for key, value in level["set"].items():
                    set_dotted(config, key, value)
                values[factor["name"]] = level["label"]
                if "grid_index" in level:
                    values["grid_index"] = level["grid_index"]
            config["seed"] = int(seed)
            run_id = "_".join(level["label"] for level in combination) + f"_s{seed}"
            candidates.append(
                {
                    "run_id": run_id,
                    "candidate": values,
                    "seed": int(seed),
                    "repetition": repetition,
                    "config": config,
                }
            )
    run_ids = [c["run_id"] for c in candidates]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("sweep factor labels do not give unique run ids")
    return candidates


def _run_candidate(
    config: dict, run_dir: str, sweep_id: str, candidate: dict, repetition: int
) -> str:
    """Worker process body: one candidate with its own logs and evidence."""

    from .runner import run_single

    run_single(
        config,
        Path(run_dir),
        sweep_id=sweep_id,
        candidate=candidate,
        repetition=repetition,
        console=False,
    )
    return run_dir


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _run_entry(item: dict, runs_dir: Path, status: str) -> dict:
    relative = f"runs/{item['run_id']}"
    return {
        "run_id": item["run_id"],
        "candidate": item["candidate"],
        "seed": item["seed"],
        "repetition": item["repetition"],
        "status": status,
        "run_dir": relative,
        "run_log": f"{relative}/run.log",
        "run_status": f"{relative}/run_status.json",
        "config_path": f"{relative}/config_snapshot.json",
        "metrics_path": f"{relative}/metrics.json",
    }


def write_summary(sweep_dir: Path, entries: list[dict]) -> Path:
    """Build ``summary.csv``/``summary.json`` from completed runs' metrics files."""

    rows = []
    for entry in entries:
        if entry["status"] != "COMPLETED":
            continue
        metrics = _read_json(sweep_dir / entry["metrics_path"])
        if metrics is None:
            raise FileNotFoundError(
                f"completed run without metrics: {entry['metrics_path']}"
            )
        row = {"run_id": entry["run_id"], "seed": entry["seed"], **entry["candidate"]}
        for split in ("train", "val", "test"):
            if split in metrics:
                row[f"{split}_accuracy"] = metrics[split]["accuracy"]
                row[f"{split}_f1_weighted"] = metrics[split]["f1_weighted"]
        for key in (
            "evaluations",
            "fit_seconds",
            "final_loss",
            "num_weights",
            "n_qubits",
            "parameters",
            "model",
        ):
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(sweep_dir / "summary.csv", index=False)
    summary_path = sweep_dir / "summary.json"
    write_json(
        summary_path,
        {
            "rows": rows,
            "n_completed": len(rows),
            "source": "metrics.json of each COMPLETED run",
        },
    )
    return summary_path


def run_sweep(cfg: dict, run_dir: Path) -> None:
    """Coordinate a sweep; resumable through its fixed ``output_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved sweep configuration.
    run_dir : pathlib.Path
        The runtime's timestamped directory; receives a pointer to the sweep.
    """

    sweep = cfg["sweep"]
    sweep_dir = (PROJECT_DIR / sweep["output_dir"]).resolve()
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(sweep_dir / "sweep.log")
    runtime_log = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    runtime_log.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().addHandler(runtime_log)
    write_json(
        run_dir / "sweep_pointer.json",
        {"sweep_id": sweep["id"], "sweep_dir": str(sweep_dir)},
    )

    candidates = expand_candidates(cfg)
    status = {
        "schema_version": 1,
        "sweep_id": sweep["id"],
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "expected_runs": len(candidates),
        "runs": [],
        "summary_path": None,
        "selected_candidates": [],
        "error": None,
    }
    pending = []
    for item in candidates:
        previous = _read_json(runs_dir / item["run_id"] / "run_status.json")
        if previous and previous.get("status") == "COMPLETED":
            status["runs"].append(_run_entry(item, runs_dir, "COMPLETED"))
        else:
            status["runs"].append(_run_entry(item, runs_dir, "PLANNED"))
            pending.append(item)
    entries = {entry["run_id"]: entry for entry in status["runs"]}
    write_json(sweep_dir / "sweep_status.json", status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | already_completed=%d | workers=%d | dir=%s",
        sweep["id"],
        len(candidates),
        len(candidates) - len(pending),
        sweep["workers"],
        sweep_dir,
    )

    # One simulation thread per worker; the pool provides the parallelism.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"

    def refresh() -> None:
        for entry in status["runs"]:
            if entry["status"] == "RUNNING":
                progress = _read_json(sweep_dir / entry["run_dir"] / "progress.json")
                if progress:
                    entry["progress"] = progress
        write_json(sweep_dir / "sweep_status.json", status)
        logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))

    try:
        with ProcessPoolExecutor(
            max_workers=int(sweep["workers"]), mp_context=get_context("spawn")
        ) as pool:
            futures = {}
            for item in pending:
                future = pool.submit(
                    _run_candidate,
                    item["config"],
                    str(runs_dir / item["run_id"]),
                    sweep["id"],
                    item["candidate"],
                    item["repetition"],
                )
                futures[future] = item
            # The pool starts at most ``workers`` jobs; mark them as they start.
            while futures:
                for future, item in futures.items():
                    entry = entries[item["run_id"]]
                    if entry["status"] == "PLANNED" and future.running():
                        entry["status"] = "RUNNING"
                        logger.info(
                            "CANDIDATE_STARTED | run_id=%s | candidate=%s",
                            item["run_id"],
                            item["candidate"],
                        )
                done, _ = wait(
                    list(futures),
                    timeout=STATUS_REFRESH_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    item = futures.pop(future)
                    entry = entries[item["run_id"]]
                    entry.pop("progress", None)
                    try:
                        future.result()
                        entry["status"] = "COMPLETED"
                        logger.info(
                            "CANDIDATE_COMPLETED | run_id=%s | metrics=%s",
                            item["run_id"],
                            entry["metrics_path"],
                        )
                    except Exception as exc:
                        entry["status"] = "FAILED"
                        logger.error(
                            "CANDIDATE_FAILED | run_id=%s | error=%s: %s",
                            item["run_id"],
                            type(exc).__name__,
                            exc,
                        )
                refresh()
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(sweep_dir / "sweep_status.json", status)
        logger.exception("SWEEP_FAILED | sweep_id=%s", sweep["id"])
        raise

    summary_path = write_summary(sweep_dir, status["runs"])
    completed = sum(entry["status"] == "COMPLETED" for entry in status["runs"])
    status["summary_path"] = summary_path.name
    status["completed_at"] = utc_now()
    if completed == len(candidates):
        status["status"] = "COMPLETED"
        write_json(sweep_dir / "sweep_status.json", status)
        logger.info(
            "SWEEP_COMPLETED | summary=%s | selected=%s",
            summary_path.name,
            "none (factorial study, no selection)",
        )
    elif completed:
        status["status"] = "PARTIAL"
        write_json(sweep_dir / "sweep_status.json", status)
        logger.warning(
            "SWEEP_PARTIAL | completed=%d/%d | summary=%s",
            completed,
            len(candidates),
            summary_path.name,
        )
    else:
        status["status"] = "FAILED"
        status["error"] = {
            "type": "AllCandidatesFailed",
            "message": "no candidate completed",
        }
        write_json(sweep_dir / "sweep_status.json", status)
        logger.error("SWEEP_FAILED | sweep_id=%s | no candidate completed", sweep["id"])
        raise RuntimeError(f"sweep {sweep['id']} had no completed candidate")
