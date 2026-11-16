"""Independent recomputation of every sweep summary row and of the paper comparison.

For each sweep directory: reads every run's ``metrics.json`` directly (not
``summary.json``), recomputes the seed mean and seed std (ddof = 1) of the
5-fold mean accuracy for every (candidate, method, model), and compares them
with ``summary.json`` at full precision. Then recomputes
``results/paper_comparison.json`` from the summaries.

Usage::

    .venv/bin/python papers/distributed_secure_kernels/utils/verify_sweeps.py <sweep dir> [<sweep dir> ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

from distributed_secure_kernels.utils.compare_with_paper import compare  # noqa: E402
from distributed_secure_kernels.utils.summarize_sweep import _models  # noqa: E402


def recompute(sweep_dir: Path) -> tuple[int, list[str]]:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    per_key = defaultdict(dict)
    for run in status["runs"]:
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        candidate = Path(run["candidate"]["config"]).stem
        for method, outcome in metrics["methods"].items():
            for model, summary in _models(outcome).items():
                per_key[(candidate, method, model)][run["seed"]] = summary["accuracy"][
                    "mean"
                ]
    summary = {
        (r["candidate"], r["method"], r["model"]): r["accuracy"]
        for r in json.loads((sweep_dir / "summary.json").read_text())["rows"]
    }
    mismatches = []
    for key, per_seed in per_key.items():
        values = np.array([per_seed[s] for s in sorted(per_seed)])
        mean, std = values.mean(), values.std(ddof=1)
        row = summary.get(key)
        if (
            row is None
            or row["seed_mean"] != mean
            or not np.isclose(row["seed_std"], std, rtol=0, atol=1e-15)
        ):
            mismatches.append(f"{key}: recomputed {mean!r} / {std!r}, summary {row}")
    if set(per_key) != set(summary):
        mismatches.append(f"row sets differ: {set(per_key) ^ set(summary)}")
    return len(per_key), mismatches


def main(argv: list[str]) -> int:
    failures = 0
    for arg in argv:
        rows, mismatches = recompute(Path(arg))
        failures += len(mismatches)
        print(
            f"{Path(arg).name}: {rows} rows recomputed from metrics.json, {len(mismatches)} mismatch(es)"
        )
        for line in mismatches:
            print("  ", line)
    paper_dirs = [
        Path(a)
        for a in argv
        if any(
            k in Path(a).name
            for k in ("table1_original", "noise_original", "shots_original")
        )
    ]
    if len(paper_dirs) == 3:
        recomputed = compare(sorted(paper_dirs))
        stored = json.loads(
            (PROJECT_DIR / "results" / "paper_comparison.json").read_text()
        )
        same = recomputed == stored
        failures += 0 if same else 1
        print(
            f"results/paper_comparison.json recomputed from the summaries: {'identical' if same else 'DIFFERENT'} ({len(stored)} rows, {sum(r['matched'] for r in stored)} matched)"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
