"""Merge the records of one or more study runs and aggregate them over seeds.

Usage (from the paper folder)::

    python utils/aggregate_study.py --run outdir/run_A --run outdir/run_B \
        --expected-seeds 0-9 --expected-sizes 3-16 --output results/study_reduced_local

Writes ``<output>.json.gz`` (coverage report plus summary rows with the paper
values attached; compact gzip-compressed JSON, see ``lib.result_io``). Coverage is checked against the
planned seed x system-size x embedding matrix, duplicate records are rejected,
and every merged run must be COMPLETED with the same conventions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

from lib import paper_values, summary  # noqa: E402
from lib.result_io import dump_json  # noqa: E402

CONSISTENT_BLOCKS = ("conventions", "kernels", "engineered", "models", "photonic")
# Runs made before the photonic block existed used k = floor(m/2) photons.
DEFAULT_BLOCKS = {"photonic": {"max_photons": None}}


def parse_range(text: str) -> list[int]:
    """Parse ``"0-9"`` or ``"3,4,8"`` into a list of integers."""
    values = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            values += list(range(int(low), int(high) + 1))
        else:
            values.append(int(part))
    return values


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    """Return the config snapshot and records of a COMPLETED run."""
    status = json.loads((run_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{run_dir} has status {status['status']}")
    config = json.loads((run_dir / "config_snapshot.json").read_text())
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text().splitlines()
        if line
    ]
    return config, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--expected-seeds", required=True)
    parser.add_argument("--expected-sizes", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--not-run",
        action="append",
        default=[],
        metavar="EMBEDDING:N",
        help="Cell declared out of scope (e.g. PB:18); removed from the expected set "
        "and listed in the report",
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = parser.parse_args(argv)

    configs, records, sources = [], [], {}
    for run_dir in args.run:
        config, run_records = load_run(run_dir)
        configs.append(config)
        for record in run_records:
            key = (record["seed"], record["n"], record.get("embedding"))
            if key in sources:
                raise ValueError(
                    f"Duplicate record {key} in {run_dir} and {sources[key]}"
                )
            sources[key] = run_dir.name
            records.append(record)
    for block in CONSISTENT_BLOCKS:
        values = {
            json.dumps(config.get(block, DEFAULT_BLOCKS.get(block)), sort_keys=True)
            for config in configs
        }
        if len(values) != 1:
            raise ValueError(f"Runs disagree on config block '{block}'")
    reference = configs[0]
    embeddings = reference["study"]["embeddings"]
    components = set(reference["study"]["components"])
    unit_kinds = ([*embeddings] if components - {"classical_dataset"} else []) + (
        [None] if "classical_dataset" in components else []
    )
    expected = {
        (seed, n, kind)
        for seed in parse_range(args.expected_seeds)
        for n in parse_range(args.expected_sizes)
        for kind in unit_kinds
    }
    declared_not_run = []
    for cell in args.not_run:
        embedding, n = cell.split(":")
        removed = {key for key in expected if key[1] == int(n) and key[2] == embedding}
        if not removed:
            raise ValueError(f"--not-run {cell} matches no expected record")
        expected -= removed
        declared_not_run.append(cell)
    observed = set(sources)
    missing, unexpected = (
        sorted(expected - observed, key=str),
        sorted(observed - expected, key=str),
    )

    rows = [row for record in records for row in summary.record_rows(record)]
    for row in rows:
        if not math.isfinite(row["value"]):
            raise ValueError(f"Non-finite value in {row}")
    reference_values = paper_values.load_paper_values(args.data_root)
    aggregated = summary.aggregate_rows(rows, reference_values)
    flagged_geometry = sum(
        1
        for record in records
        for result in record.get("dimension_geometry", {}).get("g", {}).values()
        if not result["threshold_met"]
    )
    report = {
        "runs": [str(run) for run in args.run],
        "expected_records": len(expected),
        "observed_records": len(observed),
        "missing": [list(key) for key in missing],
        "unexpected": [list(key) for key in unexpected],
        "declared_not_run": declared_not_run,
        "coverage": "COMPLETE" if not missing and not unexpected else "PARTIAL",
        "g_threshold_not_met_count": flagged_geometry,
        "conventions": reference["conventions"],
        "kernels": reference["kernels"],
        "engineered": reference["engineered"],
        "summary": aggregated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_json(
        report, args.output.with_name(args.output.name + ".json.gz"), sort_keys=True
    )
    print(
        f"coverage={report['coverage']} observed={len(observed)}/{len(expected)} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"g_threshold_not_met={flagged_geometry} rows={len(aggregated)}"
    )
    return 0 if report["coverage"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
