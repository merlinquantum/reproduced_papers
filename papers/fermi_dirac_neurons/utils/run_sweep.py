"""Sweep coordinator: one ``implementation.py`` process per (config, seed).

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/run_sweep.py \
        papers/fermi_dirac_neurons/configs/sweeps/<sweep>.json

A sweep spec (JSON) declares ``sweep_id``, ``configs`` (names under
``configs/``), ``seeds``, ``max_workers`` and the predeclared ``plan``
(metrics, aggregation, comparison rule). Layout::

    outdir/sweeps/<sweep_id>/
        sweep.log, sweep_status.json, summary.json, summary.csv
        runs/<run_id>/attempt<k>/run_<timestamp>/   (run.log, run_status.json, metrics.json, ...)
        runs/<run_id>/attempt<k>/worker.out

Each planned run gets at most two attempts. Only this coordinator writes
``sweep.log`` and ``sweep_status.json``; workers write their own ``run.log``.
Workers run with one BLAS thread each so parallel runs do not oversubscribe
the CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(REPO_ROOT))

from fermi_dirac_neurons.lib.data import state_count  # noqa: E402
from fermi_dirac_neurons.lib.experiment_logging import (  # noqa: E402
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)
from fermi_dirac_neurons.lib.runner import describe_candidate  # noqa: E402
from fermi_dirac_neurons.utils.summarize_sweep import summarize  # noqa: E402
from runtime_lib.config import deep_update  # noqa: E402

logger = logging.getLogger("fermi_dirac_neurons.sweep")
MAX_ATTEMPTS = 2
PROGRESS_EVENT = "TRAIN_EPOCH_COMPLETED"
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def paper_code_fingerprint() -> str:
    """SHA-256 over the paper's code, CLI schema, configs and requirements."""
    digest = hashlib.sha256()
    patterns = (
        "lib/*.py",
        "utils/*.py",
        "cli.json",
        "configs/*.json",
        "configs/sweeps/*.json",
        "requirements.txt",
    )
    files = sorted({path for pattern in patterns for path in PROJECT_DIR.glob(pattern)})
    for path in files:
        digest.update(str(path.relative_to(PROJECT_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolved_config(config_name: str) -> dict:
    """Overlay ``configs/<name>.json`` on ``defaults.json`` as the runtime does."""
    defaults = json.loads((PROJECT_DIR / "configs" / "defaults.json").read_text())
    overlay = json.loads((PROJECT_DIR / "configs" / f"{config_name}.json").read_text())
    return deep_update(defaults, overlay)


GIB = 2**30
COMPLEX_BYTES = 16


def estimate_memory_gib(cfg: dict) -> float:
    """Estimate one run's peak resident memory from its resolved config.

    Calibrated on measured peaks (single-threaded workers, 2026-11-09):
    training runs store every state as a dense ``d x d`` complex matrix and
    peaked at 0.58, 1.46 and 6.10-6.58 GiB for 7-, 8- and 9-qubit E11
    configs, below ``0.4 + 1.1 x stored states``; gradient-variance cells
    hold a few dozen dense ``d x d`` work matrices and peaked at 0.48, 0.74
    and 1.37 GiB for n = 9, 10, 11, below ``0.4 + 24 d^2 x 16 B``. Both models
    over-estimate the measurements, never under-estimate them.

    Raises
    ------
    ValueError
        If the task is unknown.
    """
    task = cfg["experiment"]["task"]
    if task == "train":
        data_cfg = cfg["data"]
        n_qubits = int(data_cfg["n_qubits"])
        n_states = state_count(data_cfg["train_states"], n_qubits) + state_count(
            data_cfg["validation_states"], n_qubits
        )
        return 0.4 + 1.1 * n_states * 4**n_qubits * COMPLEX_BYTES / GIB
    if task == "gradient_variance":
        n_qubits = int(cfg["gradient_variance"]["n_qubits"])
        return 0.4 + 24 * 4**n_qubits * COMPLEX_BYTES / GIB
    if task == "contextuality_scan":
        blocks = [
            block
            for block in (
                cfg["contextuality"]["invariance"],
                cfg["contextuality"]["graded"],
            )
            if block
        ]
        n_qubits = max(max(block["n_qubits"]) for block in blocks)
        # Cached Pauli basis: 4^n matrices of size d x d.
        return 0.5 + 4**n_qubits * 4**n_qubits * COMPLEX_BYTES / GIB
    if task == "pauli_compressibility":
        # Dense term stack, eigenvectors and the d x d Walsh-Hadamard work arrays.
        n_qubits = int(cfg["pauli_compressibility"]["n_qubits"])
        return 0.5 + 64 * 4**n_qubits * COMPLEX_BYTES / GIB
    if task == "theory_checks":
        return 0.5
    if task == "linear_reference":
        # Training-run states plus the dense Pauli term stack of B4 (the
        # largest feature set, weight <= max(max_weights)).
        data_cfg = cfg["data"]
        n_qubits = int(data_cfg["n_qubits"])
        n_states = state_count(data_cfg["train_states"], n_qubits) + state_count(
            data_cfg["validation_states"], n_qubits
        )
        weight = max(cfg["linear_reference"]["max_weights"])
        n_terms = sum(math.comb(n_qubits, w) * 3**w for w in range(1, weight + 1))
        # Calibrated 2026-11-10: p5_ref_fig8_logistic_7q peaked at 1.07 GiB.
        return 0.6 + (1.3 * n_states + 2 * n_terms) * 4**n_qubits * COMPLEX_BYTES / GIB
    # Phase 4 photonic tasks, calibrated 2026-11-10 (single-threaded): the
    # largest cells peaked at 0.45 GiB (photonic_train, d = 330), 1.04 and
    # 1.72 GiB (photonic_gradient_variance, d = 1540 and 2002) and 0.41 and
    # 1.12 GiB (kerr_gradient_variance, d = 364 and 2002).
    if task == "photonic_train":
        d = _fock_dimension(cfg["photonic"])
        return 0.5 + 64 * d**2 * COMPLEX_BYTES / GIB
    if task == "photonic_gradient_variance":
        d = _fock_dimension(cfg["photonic_gradient_variance"])
        return 0.4 + 24 * d**2 * COMPLEX_BYTES / GIB
    if task == "kerr_gradient_variance":
        d = _fock_dimension(cfg["kerr_gradient_variance"])
        return 0.4 + 16 * d**2 * COMPLEX_BYTES / GIB
    raise ValueError(f"No memory model for task {task!r}")


def _fock_dimension(block: dict) -> int:
    return math.comb(block["n_modes"] + block["n_photons"] - 1, block["n_photons"])


def _progress_total(config_name: str) -> int | None:
    config = json.loads((PROJECT_DIR / "configs" / f"{config_name}.json").read_text())
    return config.get("training", {}).get("epochs")


def _configure_logging(sweep_dir: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)sZ [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (
        logging.FileHandler(sweep_dir / "sweep.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


class Coordinator:
    """Plan, launch, track and summarize one sweep."""

    def __init__(self, spec: dict, spec_path: Path, memory_budget_gib: float) -> None:
        self.spec = spec
        self.memory_budget_gib = float(memory_budget_gib)
        attempts_used = spec.get("attempts_used", {})
        # Extra attempts beyond MAX_ATTEMPTS exist only with a recorded user
        # authorization: {"<config>__s<seed>": {"attempts": k, "authorized_by":
        # "user", "quote": "...", "date": "..."}}.
        extra_attempts = spec.get("authorized_extra_attempts", {})
        for key, grant in extra_attempts.items():
            missing = {"attempts", "authorized_by", "quote", "date"} - set(grant)
            if missing:
                raise ValueError(
                    f"authorized_extra_attempts[{key!r}] lacks {sorted(missing)}"
                )
        self.sweep_id = spec["sweep_id"]
        self.sweep_dir = PROJECT_DIR / "outdir" / "sweeps" / self.sweep_id
        self.lock = threading.Lock()
        self.runs: list[dict] = []
        for config_name in spec["configs"]:
            if not (PROJECT_DIR / "configs" / f"{config_name}.json").exists():
                raise FileNotFoundError(
                    f"Sweep config not found: configs/{config_name}.json"
                )
            for repetition, seed in enumerate(spec["seeds"]):
                self.runs.append(
                    {
                        "run_id": f"{self.sweep_id}__{config_name}__s{seed}",
                        "config": config_name,
                        "candidate": describe_candidate(resolved_config(config_name)),
                        "memory_estimate_gib": round(
                            estimate_memory_gib(resolved_config(config_name)), 2
                        ),
                        "attempts_used_before_this_sweep": int(
                            attempts_used.get(f"{config_name}__s{seed}", 0)
                        ),
                        "attempt_limit": MAX_ATTEMPTS
                        + int(
                            extra_attempts.get(f"{config_name}__s{seed}", {}).get(
                                "attempts", 0
                            )
                        ),
                        "seed": int(seed),
                        "repetition": repetition,
                        "status": "PLANNED",
                        "run_dir": None,
                        "run_log": None,
                        "run_status": None,
                        "config_path": None,
                        "metrics_path": None,
                        "attempts": [],
                    }
                )
        too_large = [
            run["run_id"]
            for run in self.runs
            if run["memory_estimate_gib"] > self.memory_budget_gib
        ]
        if too_large:
            raise ValueError(
                f"Runs exceed the {self.memory_budget_gib} GiB memory budget on their own: {too_large}"
            )
        exhausted = [
            run["run_id"]
            for run in self.runs
            if run["attempts_used_before_this_sweep"] >= run["attempt_limit"]
        ]
        if exhausted:
            raise ValueError(
                f"Runs already used {MAX_ATTEMPTS} attempts; another needs the user's explicit authorization and its own spec: {exhausted}"
            )
        unknown = (set(attempts_used) | set(extra_attempts)) - {
            f"{run['config']}__s{run['seed']}" for run in self.runs
        }
        if unknown:
            raise ValueError(
                f"attempts_used or authorized_extra_attempts name runs that are not planned: {sorted(unknown)}"
            )
        self.status = {
            "schema_version": 1,
            "memory_budget_gib": self.memory_budget_gib,
            "sweep_id": self.sweep_id,
            "status": "RUNNING",
            "started_at": None,
            "completed_at": None,
            "code": {
                **git_state(REPO_ROOT),
                "paper_code_sha256": paper_code_fingerprint(),
            },
            "spec_path": str(spec_path.resolve().relative_to(REPO_ROOT)),
            "spec": spec,
            "expected_runs": len(self.runs),
            "runs": self.runs,
            "summary_path": None,
            "selected_candidates": [],
            "error": None,
        }

    def _write_status(self, log_progress: bool = True) -> None:
        with self.lock:
            write_json(self.sweep_dir / "sweep_status.json", self.status)
            if log_progress and self.status["status"] == "RUNNING":
                logger.info("SWEEP_PROGRESS | %s", sweep_progress(self.status))

    def _command(self, run: dict, outdir: Path) -> list[str]:
        return [
            sys.executable,
            str(REPO_ROOT / "implementation.py"),
            "--paper",
            "fermi_dirac_neurons",
            "--config",
            f"configs/{run['config']}.json",
            "--seed",
            str(run["seed"]),
            "--outdir",
            str(outdir),
            "--sweep-id",
            self.sweep_id,
            "--run-id",
            run["run_id"],
            "--repetition",
            str(run["repetition"]),
        ]

    def _track_progress(
        self, run: dict, outdir: Path, process: subprocess.Popen, total: int | None
    ) -> None:
        while True:
            try:
                process.wait(timeout=20)
                return
            except subprocess.TimeoutExpired:
                pass
            logs = list(outdir.glob("run_*/run.log"))
            if total and logs:
                current = (
                    logs[0]
                    .read_text(encoding="utf-8", errors="replace")
                    .count(PROGRESS_EVENT)
                )
                with self.lock:
                    run["progress"] = {
                        "current": current,
                        "total": total,
                        "unit": "epoch",
                    }
                self._write_status()

    def _execute(self, run: dict) -> None:
        total = _progress_total(run["config"])
        first_attempt = run["attempts_used_before_this_sweep"] + 1
        for attempt in range(first_attempt, run["attempt_limit"] + 1):
            outdir = self.sweep_dir / "runs" / run["run_id"] / f"attempt{attempt}"
            outdir.mkdir(parents=True, exist_ok=True)
            command = self._command(run, outdir)
            with self.lock:
                run["status"] = "RUNNING"
            event = (
                "CANDIDATE_STARTED"
                if attempt == first_attempt
                else "CANDIDATE_RETRY_STARTED"
            )
            logger.info(
                "%s | run_id=%s | config=%s | seed=%s | attempt=%d",
                event,
                run["run_id"],
                run["config"],
                run["seed"],
                attempt,
            )
            self._write_status()
            started = time.time()
            with (outdir / "worker.out").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, **SINGLE_THREAD_ENV},
                )
                self._track_progress(run, outdir, process, total)
            run_dirs = sorted(outdir.glob("run_*"))
            record = {
                "attempt": attempt,
                "command": " ".join(command),
                "exit_code": process.returncode,
                "elapsed_s": round(time.time() - started, 1),
                "run_dir": str(run_dirs[0].relative_to(self.sweep_dir))
                if run_dirs
                else None,
            }
            completed = False
            if process.returncode == 0 and len(run_dirs) == 1:
                status_file = run_dirs[0] / "run_status.json"
                metrics_file = run_dirs[0] / "metrics.json"
                if status_file.exists() and metrics_file.exists():
                    run_status = json.loads(status_file.read_text())
                    json.loads(metrics_file.read_text())
                    completed = run_status.get("status") == "COMPLETED"
            with self.lock:
                run["attempts"].append(record)
                if completed:
                    relative = run_dirs[0].relative_to(self.sweep_dir)
                    run.update(
                        status="COMPLETED",
                        run_dir=str(relative),
                        run_log=str(relative / "run.log"),
                        run_status=str(relative / "run_status.json"),
                        config_path=str(relative / "config_snapshot.json"),
                        metrics_path=str(relative / "metrics.json"),
                    )
                    if total:
                        run["progress"] = {
                            "current": total,
                            "total": total,
                            "unit": "epoch",
                        }
            if completed:
                logger.info(
                    "CANDIDATE_COMPLETED | run_id=%s | metrics=%s | elapsed_s=%.1f",
                    run["run_id"],
                    run["metrics_path"],
                    record["elapsed_s"],
                )
                self._write_status()
                return
            logger.warning(
                "CANDIDATE_ATTEMPT_FAILED | run_id=%s | attempt=%d | exit_code=%s",
                run["run_id"],
                attempt,
                process.returncode,
            )
        with self.lock:
            run["status"] = "FAILED"
        logger.error(
            "CANDIDATE_FAILED | run_id=%s | attempts=%d",
            run["run_id"],
            run["attempt_limit"],
        )
        self._write_status()

    def _schedule(self) -> None:
        """Start runs in declared order while slots and the memory budget allow.

        First fit: a run that does not fit yet waits, and a later, lighter run
        may start in the remaining memory. Each run keeps its memory
        reservation for both of its attempts.
        """
        pending = list(self.runs)
        running: dict = {}
        reserved = 0.0
        with ThreadPoolExecutor(max_workers=int(self.spec["max_workers"])) as pool:
            while pending or running:
                for run in list(pending):
                    if len(running) >= int(self.spec["max_workers"]):
                        break
                    if reserved + run["memory_estimate_gib"] <= self.memory_budget_gib:
                        pending.remove(run)
                        reserved += run["memory_estimate_gib"]
                        running[pool.submit(self._execute, run)] = run
                if not running:
                    raise RuntimeError(
                        "No pending run fits the memory budget; this should have been caught at planning"
                    )
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    run = running.pop(future)
                    reserved -= run["memory_estimate_gib"]
                    future.result()

    def run(self) -> int:
        """Execute every planned run and write the summary; return a process exit code."""
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        _configure_logging(self.sweep_dir)
        self.status["started_at"] = utc_now()
        logger.info(
            "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | workers=%d | memory_budget_gib=%g",
            self.sweep_id,
            len(self.runs),
            self.spec["max_workers"],
            self.memory_budget_gib,
        )
        self._write_status(log_progress=False)
        try:
            self._schedule()
            summary_path = summarize(self.sweep_dir, self.status)
            completed = sum(run["status"] == "COMPLETED" for run in self.runs)
            self.status["summary_path"] = summary_path.name
            self.status["completed_at"] = utc_now()
            if completed == len(self.runs):
                self.status["status"] = "COMPLETED"
                self._write_status(log_progress=False)
                logger.info(
                    "SWEEP_COMPLETED | summary=%s | selected=none (comparison sweep, no selection)",
                    summary_path.name,
                )
                return 0
            self.status["status"] = "PARTIAL" if completed else "FAILED"
            self._write_status(log_progress=False)
            event = "SWEEP_PARTIAL" if completed else "SWEEP_FAILED"
            logger.error(
                "%s | completed=%d/%d | summary=%s",
                event,
                completed,
                len(self.runs),
                summary_path.name,
            )
            return 1
        except Exception as exc:
            self.status.update(
                status="FAILED",
                completed_at=utc_now(),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            self._write_status(log_progress=False)
            logger.exception("SWEEP_FAILED | sweep_id=%s", self.sweep_id)
            raise


def main() -> int:
    """Parse the spec path and run the sweep."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", type=Path, help="Sweep spec JSON")
    parser.add_argument(
        "--memory-budget-gib",
        type=float,
        required=True,
        help="Upper bound on the summed memory estimates of concurrently running workers",
    )
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    return Coordinator(spec, args.spec, args.memory_budget_gib).run()


if __name__ == "__main__":
    raise SystemExit(main())
