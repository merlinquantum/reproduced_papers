"""Phase 5: apply the predeclared selection rule to the tuning-grid batches.

Rule (LOG.md, Phase 5 Comparison Plan): for each model, the metric is the
median over the 5 selection agents (seeds 100-104) of the 0-based solve
episode index, unsolved agents censored at 5000; lower is better. Candidates
whose median is within 5% of the best median are tied; among tied candidates
the smaller batch size wins, then the smaller target-update interval.

The script checks that every planned candidate (3 batch sizes x 2 target
intervals per model) is present exactly once, COMPLETED, with 5 agents of
seeds 100-104 and finite metrics, and writes the full table and the
selection to ``--output``.

Example (from the paper folder)::

    python utils/select_phase5.py --batches outdir/batches/e5_select_nn \\
        --models mlp_5_5 mlp_14_14 --output results/phase5_selection_nn.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

BATCHES = (16, 32, 64)
TARGETS = (1, 30)
SEEDS = [100, 101, 102, 103, 104]
CAP = 5000
TIE_TOLERANCE = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = {}
    for batch_dir in args.batches:
        for entry in json.loads((batch_dir / "sweep_status.json").read_text())["runs"]:
            stem = Path(entry["config_file"]).stem
            if stem in entries:
                raise ValueError(f"Duplicate candidate {stem}")
            entries[stem] = (batch_dir, entry)

    result = {"rule": __doc__.split("\n\n")[1], "models": {}}
    for model in args.models:
        suffix = "_classical" if model.startswith("mlp") else ""
        rows = []
        for batch, target in itertools.product(BATCHES, TARGETS):
            stem = f"e5_select_{model}_b{batch}_t{target}{suffix}"
            if stem not in entries:
                raise KeyError(f"Missing planned candidate {stem}")
            batch_dir, entry = entries[stem]
            if entry["status"] != "COMPLETED":
                raise ValueError(f"{stem} is {entry['status']}")
            metrics = json.loads((batch_dir / entry["metrics_path"]).read_text())
            agents = metrics["agents"]
            if sorted(a["seed"] for a in agents) != SEEDS:
                raise ValueError(f"{stem} has seeds {[a['seed'] for a in agents]}")
            censored = [
                a["solved_at_episode_index"] if a["solved"] else CAP for a in agents
            ]
            if not all(math.isfinite(v) for v in censored):
                raise ValueError(f"Non-finite metric in {stem}")
            rows.append(
                {
                    "config": stem,
                    "batch_size": batch,
                    "update_target_every": target,
                    "median_censored": float(np.median(censored)),
                    "mean_censored": float(np.mean(censored)),
                    "solved": int(sum(a["solved"] for a in agents)),
                    "censored_values": censored,
                    "wall_clock_seconds": metrics["wall_clock_seconds"],
                    "run_dir": str(batch_dir / entry["run_dir"]),
                }
            )
        best = min(row["median_censored"] for row in rows)
        tied = [
            row for row in rows if row["median_censored"] <= best * (1 + TIE_TOLERANCE)
        ]
        selected = min(
            tied, key=lambda row: (row["batch_size"], row["update_target_every"])
        )
        result["models"][model] = {
            "candidates": rows,
            "best_median": best,
            "tied_candidates": [row["config"] for row in tied],
            "selected": selected["config"],
            "selected_batch_size": selected["batch_size"],
            "selected_update_target_every": selected["update_target_every"],
        }
        print(f"{model}:")
        for row in rows:
            mark = (
                "  <- selected"
                if row is selected
                else ("  (tied)" if row in tied else "")
            )
            print(
                f"  b{row['batch_size']:>2} t{row['update_target_every']:>2}: median "
                f"{row['median_censored']:7.1f}  solved {row['solved']}/5{mark}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
