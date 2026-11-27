"""Export per-seed values of the headline quantities to a compact CSV.

Usage (from the paper folder)::

    python utils/export_per_seed.py --run outdir/run_A --run outdir/run_B ... \
        --output results/per_seed_key_rows.csv.gz

The aggregates in ``results/`` keep only mean, std and CI per row identity.
This file keeps the individual seeds, so the notebook can show every seed and
recompute the aggregates. Rows kept: effective dimensions and geometric
differences, the bound, and, for the four tasks, the best classical model, the
two quantum kernels and their paired differences. Each row is labelled with its
family (``gate``, ``photonic`` with k = floor(m/2), ``photonic_capped``) and the
machine that produced it.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

from lib import photonic_merlin, summary  # noqa: E402

TASK_QUANTITIES = {
    "best_classical",
    "quantum_kernel",
    "projected_quantum_kernel",
    "quantum_kernel_minus_best_classical",
    "projected_quantum_kernel_minus_best_classical",
}
PER_EMBEDDING_KERNEL = re.compile(
    r"(projected_)?quantum_kernel_(E1|E2|E3|PA|PB)(_minus_best_classical)?"
)
GEOMETRY_QUANTITIES = {
    "d_quantum_kernel",
    "d_projected_quantum_kernel",
    "g_quantum_kernel",
    "g_projected_quantum_kernel",
}
COLUMNS = (
    "family",
    "machine",
    "run",
    "seed",
    "n",
    "embedding",
    "component",
    "quantity",
    "train_size",
    "protocol",
    "value",
)


def run_family(config: dict) -> str:
    embeddings = set(config["study"]["embeddings"])
    if embeddings <= set(photonic_merlin.PHOTONIC_EMBEDDINGS):
        capped = config.get("photonic", {}).get("max_photons") is not None
        return "photonic_capped" if capped else "photonic"
    return "gate"


def keep(row: dict) -> bool:
    if row["component"] == "dimension_geometry":
        return row["quantity"] in GEOMETRY_QUANTITIES
    if row["component"] == "bound":
        return True
    # Dataset (C) units hold every embedding's kernels, named per embedding.
    return row["quantity"] in TASK_QUANTITIES or bool(
        PER_EMBEDDING_KERNEL.fullmatch(row["quantity"])
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    rows_out = []
    for run_dir in args.run:
        status = json.loads((run_dir / "run_status.json").read_text())
        if status["status"] != "COMPLETED":
            raise ValueError(f"{run_dir} has status {status['status']}")
        config = json.loads((run_dir / "config_snapshot.json").read_text())
        family = run_family(config)
        machine = (
            "L4 GPU float32"
            if config["device"].startswith("cuda")
            else "M5 CPU float64"
        )
        for line in (run_dir / "records.jsonl").read_text().splitlines():
            for row in summary.record_rows(json.loads(line)):
                if keep(row):
                    rows_out.append(
                        {
                            "family": family,
                            "machine": machine,
                            "run": run_dir.name,
                            **row,
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A .gz output is gzip-compressed (about 15x smaller); pandas reads it directly.
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {len(rows_out)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
