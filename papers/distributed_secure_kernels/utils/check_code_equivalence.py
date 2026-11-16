"""Check that the current code reproduces the metrics of earlier sweeps exactly.

For every candidate of the given sweeps, re-runs seed 0 with the current code
(through the shared runtime, into ``outdir/code_equivalence_<stamp>/``) and
compares its ``metrics.json`` with the sweep's seed-0 run, ignoring wall-clock
fields (``seconds``). Writes ``report.json`` and ``check.log`` in that folder.

The runs record ``code.paper_tree_sha256``, so the report ties the earlier
results to an identified code state.

Usage::

    .venv/bin/python papers/distributed_secure_kernels/utils/check_code_equivalence.py --workers 5 <sweep dir> [...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parents[1]
sys.path[:0] = [str(PROJECT_DIR.parent), str(REPO_DIR)]

from distributed_secure_kernels.lib.experiment_logging import (  # noqa: E402
    configure_logging,
    write_json,
)

logger = logging.getLogger("code_equivalence")


def _strip_timing(value):
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items() if k != "seconds"}
    if isinstance(value, list):
        return [_strip_timing(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("sweeps", nargs="+")
    args = parser.parse_args(argv)
    out_dir = (
        PROJECT_DIR / "outdir" / f"code_equivalence_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True)
    configure_logging(out_dir / "check.log")
    jobs = []
    for sweep in args.sweeps:
        status = json.loads((Path(sweep) / "sweep_status.json").read_text())
        for run in status["runs"]:
            if run["seed"] == 0:
                jobs.append(
                    {
                        "sweep": Path(sweep).name,
                        "config": run["candidate"]["config"],
                        "reference": str(Path(sweep) / run["metrics_path"]),
                    }
                )
    logger.info("CHECK_STARTED | candidates=%d | workers=%d", len(jobs), args.workers)
    threads = str(max(1, (os.cpu_count() or 1) // max(1, args.workers)))
    env = {**os.environ, "OMP_NUM_THREADS": threads, "VECLIB_MAXIMUM_THREADS": threads}
    pending, running = list(enumerate(jobs)), {}
    while pending or running:
        while pending and len(running) < args.workers:
            index, job = pending.pop(0)
            target = out_dir / f"{index:02d}_{Path(job['config']).stem}"
            command = [
                sys.executable,
                str(REPO_DIR / "implementation.py"),
                "--paper",
                PROJECT_DIR.name,
                "--config",
                job["config"],
                "--seed",
                "0",
                "--outdir",
                str(target),
            ]
            running[index] = (
                subprocess.Popen(
                    command,
                    cwd=REPO_DIR,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ),
                job,
                target,
            )
        time.sleep(3)
        for index, (process, job, target) in list(running.items()):
            if process.poll() is None:
                continue
            del running[index]
            run_dir = sorted(target.glob("run_*"))[-1]
            new = json.loads((run_dir / "metrics.json").read_text())
            old = json.loads(Path(job["reference"]).read_text())
            job.update(
                exit_code=process.returncode,
                run_dir=os.path.relpath(run_dir, out_dir),
                identical=_strip_timing(new) == _strip_timing(old),
                paper_tree_sha256=json.loads((run_dir / "run_status.json").read_text())[
                    "code"
                ]["paper_tree_sha256"],
            )
            logger.info(
                "CANDIDATE_CHECKED | sweep=%s | config=%s | identical=%s",
                job["sweep"],
                job["config"],
                job["identical"],
            )
    identical = all(job.get("identical") for job in jobs)
    write_json(
        out_dir / "report.json", {"all_identical": identical, "candidates": jobs}
    )
    logger.info("CHECK_COMPLETED | all_identical=%s | report=report.json", identical)
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
