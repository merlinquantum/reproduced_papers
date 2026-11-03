"""Export per-run scalar metrics of every sweep to a committed CSV.

Usage::

    python3 papers/virtual_qnn/utils/export_per_run.py \
        --sweeps-root papers/virtual_qnn/outdir/sweeps --out papers/virtual_qnn/results/per_run_metrics.csv

One row per COMPLETED run, with sweep ID, run ID, seed, repetition, the swept
grid values and every scalar metric of its ``metrics.json``. This lets the
notebook show individual seeds without the git-ignored ``outdir/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweeps-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    for status_path in sorted(args.sweeps_root.glob("*/sweep_status.json")):
        status = json.loads(status_path.read_text())
        for run in status["runs"]:
            if run["status"] != "COMPLETED":
                continue
            metrics = json.loads((status_path.parent / run["metrics_path"]).read_text())
            row = {
                "sweep_id": status["sweep_id"],
                "run_id": run["run_id"],
                "seed": run["seed"],
                "repetition": run["repetition"],
            }
            row.update({f"grid{k}": v for k, v in run["grid_point"].items()})
            row.update(
                {
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float, str, bool))
                    and not isinstance(v, dict)
                }
            )
            rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(args.out, len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
