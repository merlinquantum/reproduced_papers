"""Verify sweep comparability and recompute every summary from raw metrics.json.

Usage::

    python papers/photonic_MMD_generative/utils/verify_sweeps.py <sweep_dir> [<sweep_dir> ...]

For each sweep:
- comparability: all included runs share one code commit and dirty flag, one
  launch (hence one code copy), and every candidate uses one train/test SHA-256
  pair (a sweep may vary the dataset across candidates, e.g. the noise study's
  m=12 and m=16 sizes, but never within a candidate);
- recomputation: mean and std (ddof=1) of every metric per candidate, from the
  runs' metrics.json, compared with summary.json at full precision.
Prints one line per sweep and exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from summarize_sweep import flatten_metrics


def verify(sweep_dir: Path) -> list[str]:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    summary = json.loads((sweep_dir / "summary.json").read_text())
    problems = []
    commits, dirty = set(), set()
    hashes = defaultdict(set)
    values = defaultdict(lambda: defaultdict(list))
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        run_status = json.loads((sweep_dir / run["run_status"]).read_text())
        commits.add(run_status["code"]["commit"])
        dirty.add(run_status["code"]["dirty"])
        subset = run_status["dataset"]["subset"]
        key = json.dumps(run["candidate"], sort_keys=True)
        hashes[key].add((subset["train_sha256"], subset["test_sha256"]))
        flat = flatten_metrics(
            json.loads((sweep_dir / run["metrics_path"]).read_text())
        )
        for metric, value in flat.items():
            values[key][metric].append(value)
    if len(commits) != 1 or len(dirty) != 1:
        problems.append(f"code state differs: commits={commits} dirty={dirty}")
    for key, pairs in hashes.items():
        if len(pairs) != 1:
            problems.append(
                f"{json.loads(key)['label']}: dataset hashes differ within the candidate ({len(pairs)} variants)"
            )
    comparability_problems = len(problems)
    for key, metrics in values.items():
        recorded = summary["candidates"][key]["metrics"]
        for metric, series in metrics.items():
            arr = np.array(series, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else None
            rec = recorded[metric]
            if rec["n"] != len(arr) or rec["mean"] != mean or rec["std"] != std:
                problems.append(
                    f"{json.loads(key)['label']}/{metric}: recomputed {mean}, {std}, n={len(arr)} vs {rec}"
                )
    print(
        f"{status['sweep_id']}: runs={sum(len(next(iter(m.values()))) for m in values.values())} "
        f"commit={sorted(commits)} dirty={sorted(dirty)} "
        f"datasets={len(set().union(*hashes.values()))} (one per candidate: {'OK' if comparability_problems == 0 else 'NO'}) "
        f"summary_recomputed={'OK' if len(problems) == comparability_problems else 'MISMATCH'}"
    )
    return problems


def main() -> int:
    """Verify every given sweep."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    problems = []
    for sweep_dir in parser.parse_args().sweep_dirs:
        problems += [f"{sweep_dir.name}: {p}" for p in verify(sweep_dir.resolve())]
    for problem in problems:
        print("PROBLEM:", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
