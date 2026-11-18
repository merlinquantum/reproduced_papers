"""Run several experiment configs concurrently as one batch (sweep coordinator).

Each config is one run of the shared runtime (``implementation.py``), which
trains all of the config's agents in one process. On the remote L4 the
machine's queue runs one job at a time, so concurrent arms have to live inside
one launch: this coordinator is that launch.

Layout (all paths relative to the paper folder)::

    outdir/batches/<batch_id>/
        sweep.log, sweep_status.json, summary.json   (coordinator only)
        runs/<config_stem>/run_YYYYMMDD-HHMMSS/      (one worker each)

Each worker writes only its own ``run.log``. The summary is built from each
run's ``metrics.json``; progress comes from the workers' ``run.log``
(``TRAIN_EPISODE_COMPLETED`` lines) and is never used as a result.

Example (from the repository root)::

    python papers/quantum_agents_deep_q_learning/utils/run_batch.py \\
        --batch-id e1_frozenlake --device cuda:0 --max-parallel 8 \\
        --configs e1_frozenlake_l5_original e1_frozenlake_l10_original
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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

logger = logging.getLogger("run_batch")

PAPER_NAME = PROJECT_DIR.name
_EPISODE_PATTERN = re.compile(
    r"TRAIN_EPISODE_COMPLETED \| agent_seed=\d+ \| episode=(\d+)/(\d+)"
)


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def _latest_episode(run_log: Path) -> int | None:
    """Highest episode index logged so far (progress only, read from the log tail)."""
    if not run_log.exists():
        return None
    with run_log.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 65536))
        tail = handle.read().decode("utf-8", errors="replace")
    episodes = [int(match.group(1)) for match in _EPISODE_PATTERN.finditer(tail)]
    return max(episodes) if episodes else None


def _summarise_run(run_dir: Path) -> dict:
    """Aggregate one run's ``metrics.json`` (full precision, no rounding)."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    agents = metrics["agents"]
    solved = [agent["solved_at_episode_index"] for agent in agents if agent["solved"]]
    return {
        "experiment": metrics["experiment"],
        "agents": len(agents),
        "solved_agents": len(solved),
        "solved_at_episode_index": [
            agent["solved_at_episode_index"] for agent in agents
        ],
        "solve_summary": metrics["solve_summary"],
        "greedy_eval_mean_over_agents": metrics["greedy_eval_mean_over_agents"],
        "lockstep_iterations": metrics["lockstep_iterations"],
        "seconds_per_lockstep_iteration": metrics["seconds_per_lockstep_iteration"],
    }


class Batch:
    """Coordinator state: one entry per config, rewritten atomically to disk.

    Parameters
    ----------
    batch_id : str
        Batch (sweep) identifier, also the folder name.
    configs : list of str
        Config stems under ``configs/``.
    device : str
        Torch device passed to every run (``--device``).
    max_parallel : int
        Number of runs alive at the same time.
    threads_per_run : int
        ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` for every worker.
    poll_seconds : float
        Interval between worker checks and status rewrites.
    """

    def __init__(
        self, batch_id, configs, device, max_parallel, threads_per_run, poll_seconds
    ):
        self.poll_seconds = poll_seconds
        self.batch_id = batch_id
        self.device = device
        self.max_parallel = max_parallel
        self.threads_per_run = threads_per_run
        self.sweep_dir = PROJECT_DIR / "outdir" / "batches" / batch_id
        if self.sweep_dir.exists():
            raise FileExistsError(f"Batch folder already exists: {self.sweep_dir}")
        (self.sweep_dir / "runs").mkdir(parents=True)
        self.status_path = self.sweep_dir / "sweep_status.json"
        self.processes: dict[str, subprocess.Popen] = {}
        self.entries = {}
        for stem in configs:
            config_path = PROJECT_DIR / "configs" / f"{stem}.json"
            config = json.loads(config_path.read_text())
            self.entries[stem] = {
                "run_id": None,
                "candidate": None,
                "config_file": f"configs/{stem}.json",
                "experiment": config["experiment"],
                "seed": config["agents"]["seeds"],
                "repetition": None,
                "status": "PLANNED",
                "run_dir": None,
                "run_log": None,
                "run_status": None,
                "config_path": None,
                "metrics_path": None,
                "command": None,
                "max_episodes": config["training"]["max_episodes"],
            }
        self.status = {
            "schema_version": 1,
            "sweep_id": batch_id,
            "status": "RUNNING",
            "started_at": utc_now(),
            "completed_at": None,
            "code": git_state(REPO_ROOT),
            "expected_runs": len(configs),
            "runs": [],
            "summary_path": None,
            "selected_candidates": [],
            "error": None,
            "device": device,
            "max_parallel": max_parallel,
        }

    def write_status(self) -> None:
        self.status["runs"] = list(self.entries.values())
        write_json(self.status_path, self.status)
        if self.status["status"] == "RUNNING":
            logger.info("SWEEP_PROGRESS | %s", sweep_progress(self.status))

    def _run_dir(self, stem: str) -> Path | None:
        found = sorted((self.sweep_dir / "runs" / stem).glob("run_*"))
        return found[-1] if found else None

    def start(self, stem: str) -> None:
        output_dir = self.sweep_dir / "runs" / stem
        command = [
            sys.executable,
            str(REPO_ROOT / "implementation.py"),
            "--paper",
            PAPER_NAME,
            "--config",
            f"configs/{stem}.json",
            "--device",
            self.device,
            "--outdir",
            str(output_dir),
        ]
        environment = dict(
            os.environ,
            ALPS_SWEEP_ID=self.batch_id,
            OMP_NUM_THREADS=str(self.threads_per_run),
            MKL_NUM_THREADS=str(self.threads_per_run),
        )
        # Workers inherit no handler from the coordinator: their stdout goes to
        # a per-run console file next to the run folder, never to sweep.log.
        console = (self.sweep_dir / "runs" / f"{stem}.console.txt").open("w")
        self.processes[stem] = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=console,
            stderr=subprocess.STDOUT,
        )
        entry = self.entries[stem]
        entry["status"] = "RUNNING"
        entry["command"] = " ".join(
            [
                "python implementation.py --paper",
                PAPER_NAME,
                f"--config configs/{stem}.json --device {self.device}",
                f"--outdir {_relative(output_dir, PROJECT_DIR)}",
            ]
        )
        entry["progress"] = {
            "current": 0,
            "total": entry["max_episodes"],
            "unit": "episodes (cap)",
        }
        logger.info(
            "CANDIDATE_STARTED | config=%s | pid=%d", stem, self.processes[stem].pid
        )
        # Distinct run_YYYYMMDD-HHMMSS names across concurrent workers.
        time.sleep(1.1)

    def _attach_paths(self, stem: str) -> Path | None:
        run_dir = self._run_dir(stem)
        if run_dir is None:
            return None
        entry = self.entries[stem]
        entry["run_id"] = run_dir.name
        entry["run_dir"] = _relative(run_dir, self.sweep_dir)
        entry["run_log"] = _relative(run_dir / "run.log", self.sweep_dir)
        entry["run_status"] = _relative(run_dir / "run_status.json", self.sweep_dir)
        entry["config_path"] = _relative(
            run_dir / "config_snapshot.json", self.sweep_dir
        )
        entry["metrics_path"] = _relative(run_dir / "metrics.json", self.sweep_dir)
        status_file = run_dir / "run_status.json"
        if status_file.exists():
            # The validator requires the sweep candidate to equal the run's own.
            entry["candidate"] = json.loads(status_file.read_text())["candidate"]
        return run_dir

    def poll(self) -> None:
        for stem, process in list(self.processes.items()):
            entry = self.entries[stem]
            run_dir = self._attach_paths(stem)
            if run_dir is not None:
                episode = _latest_episode(run_dir / "run.log")
                if episode is not None:
                    entry["progress"]["current"] = episode
            return_code = process.poll()
            if return_code is None:
                continue
            del self.processes[stem]
            entry.pop("progress", None)
            run_status = None
            if run_dir is not None and (run_dir / "run_status.json").exists():
                run_status = json.loads((run_dir / "run_status.json").read_text())
            metrics_ok = run_dir is not None and (run_dir / "metrics.json").exists()
            if (
                return_code == 0
                and run_status
                and run_status["status"] == "COMPLETED"
                and metrics_ok
            ):
                entry["status"] = "COMPLETED"
                logger.info(
                    "CANDIDATE_COMPLETED | run_id=%s | config=%s | metrics=%s",
                    entry["run_id"],
                    stem,
                    entry["metrics_path"],
                )
            else:
                entry["status"] = "FAILED"
                logger.error(
                    "CANDIDATE_FAILED | run_id=%s | config=%s | exit=%s | run_status=%s",
                    entry["run_id"],
                    stem,
                    return_code,
                    None if run_status is None else run_status["status"],
                )

    def run(self) -> int:
        logger.info(
            "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | device=%s | max_parallel=%d",
            self.batch_id,
            len(self.entries),
            self.device,
            self.max_parallel,
        )
        self.write_status()
        pending = list(self.entries)
        while pending or self.processes:
            while pending and len(self.processes) < self.max_parallel:
                self.start(pending.pop(0))
                self.write_status()
            time.sleep(self.poll_seconds)
            self.poll()
            self.write_status()
        return self.finish()

    def finish(self) -> int:
        summary = {"sweep_id": self.batch_id, "generated_at": utc_now(), "runs": {}}
        for stem, entry in self.entries.items():
            if entry["status"] == "COMPLETED":
                summary["runs"][stem] = _summarise_run(
                    self.sweep_dir / entry["run_dir"]
                )
        summary_path = self.sweep_dir / "summary.json"
        write_json(summary_path, summary)
        statuses = [entry["status"] for entry in self.entries.values()]
        self.status["summary_path"] = _relative(summary_path, self.sweep_dir)
        self.status["completed_at"] = utc_now()
        if all(status == "COMPLETED" for status in statuses):
            self.status["status"] = "COMPLETED"
            self.write_status()
            logger.info(
                "SWEEP_COMPLETED | summary=%s | selected=[]",
                self.status["summary_path"],
            )
            return 0
        if any(status == "COMPLETED" for status in statuses):
            self.status["status"] = "PARTIAL"
            self.status["error"] = "some runs failed: " + ", ".join(
                stem
                for stem, entry in self.entries.items()
                if entry["status"] != "COMPLETED"
            )
            self.write_status()
            logger.warning(
                "SWEEP_PARTIAL | summary=%s | %s",
                self.status["summary_path"],
                self.status["error"],
            )
            return 1
        self.status["status"] = "FAILED"
        self.status["error"] = "no run completed"
        self.write_status()
        logger.error("SWEEP_FAILED | summary=%s", self.status["summary_path"])
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-id", required=True)
    parser.add_argument(
        "--configs", nargs="+", required=True, help="Config stems in configs/"
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-parallel", type=int, required=True)
    parser.add_argument("--threads-per-run", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    batch = Batch(
        args.batch_id,
        args.configs,
        args.device,
        args.max_parallel,
        args.threads_per_run,
        args.poll_seconds,
    )
    configure_logging(batch.sweep_dir / "sweep.log")
    try:
        return batch.run()
    except BaseException:
        for process in batch.processes.values():
            process.terminate()
        batch.status["status"] = "FAILED"
        batch.status["completed_at"] = utc_now()
        batch.status["error"] = "coordinator interrupted"
        batch.write_status()
        logger.exception("SWEEP_FAILED | coordinator interrupted")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
