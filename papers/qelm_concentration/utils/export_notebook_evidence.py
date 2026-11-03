"""Export per-seed evidence from validated runs into one compact file for the notebook.

The raw run directories under ``outdir/`` are not committed. The notebook therefore loads
``results/notebook_evidence.json``, which keeps every individual evaluation behind the
README headline numbers (per split, reservoir and shot seed), so the notebook can show
seed-level spread and recompute every aggregate.

Usage (from the paper folder)::

    python utils/export_notebook_evidence.py --output results/notebook_evidence.json

Every source run must have ``run_status.json`` with status ``COMPLETED``, finite metrics
and the expected record counts; otherwise the script exits with an error.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]

# Source runs (the same runs as `utils/compare_phase5.py` and the README; see LOG.md).
# Landsat QELM systems: label -> (machine, run directories).
LANDSAT_QELM_RUNS = {
    "qubit N=16 (min-max)": ("laptop", ["outdir/run_20261023-173640"]),
    "qubit N=24 (min-max)": (
        "gpu-reproduced-papers-2",
        [
            "outdir/landsat_n24_eval/run_20261026-094113",
            "outdir/landsat_n24_remainder/run_20261026-084131",
        ],
    ),
    "qubit N=16 (z-score)": (
        "laptop",
        ["outdir/landsat_n16_standard/run_20261026-155536"],
    ),
    "qubit N=24 (z-score)": (
        "gpu-reproduced-papers-2",
        ["outdir/landsat_n24_standard/run_20261026-171518"],
    ),
    "photonic 16/4": ("laptop", ["outdir/merlin_landsat_m16/run_20261028-121441"]),
    "photonic 24/6": ("laptop", ["outdir/merlin_landsat_m24/run_20261028-121443"]),
}
LANDSAT_CLASSICAL_RUNS = {
    "minmax": "outdir/run_20261023-172941",
    "standard": "outdir/landsat_classical_standard/run_20261028-152509",
}
NARMA_RUNS = {
    "qubit N=8": "outdir/narma_n8/run_20261023-181420",
    "qubit N=12": "outdir/narma_n12/run_20261023-181420",
    "qubit N=16": "outdir/narma_n16/run_20261026-094003",
    "qubit N=20": "outdir/narma_n20/run_20261027-113613",
    "qubit N=24": "outdir/narma_n24/run_20261027-120754",
    "photonic 16/4": "outdir/merlin_narma_m16/run_20261028-121446",
    "photonic 24/6": "outdir/merlin_narma_m24/run_20261028-121449",
}
# Gap diagnostic D1 (exact probabilities vs 1e5 shots) for both input scalings.
GAP_RUNS = {
    "minmax": "outdir/gap_d1d2/run_20261026-144316",
    "standard": "outdir/gap_d1_standard/run_20261026-155453",
}
CONCENTRATION_RUNS = {
    "qubit": "outdir/concentration/run_20261023-181420",
    "qubit_scrambling": "outdir/concentration_scrambling/run_20261023-182036",
    "photonic": "outdir/merlin_concentration/run_20261028-121440",
}
PARETO_RUNS = {n: f"outdir/pareto_n{n}/run_20261026-094431" for n in (6, 8, 10, 12)}
PARETO_COLUMNS = (
    "trial",
    "is_table_I",
    "narma_noiseless",
    "narma_noisy",
    "bond_dimension",
    "variability",
)


class EvidenceError(RuntimeError):
    """A source run is missing, incomplete or inconsistent."""


def _load_run(relative: str) -> tuple[dict, dict]:
    run_dir = PAPER_DIR / relative
    status = json.loads((run_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise EvidenceError(f"{relative}: status {status['status']}")
    return status, json.loads((run_dir / "metrics.json").read_text())


def _check_finite(values, where: str) -> None:
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise EvidenceError(f"{where}: non-finite value")


def landsat_qelm(label: str, machine: str, runs: list[str]) -> dict:
    """Every headline, cutoff and shots evaluation, plus train-size means per split."""
    readouts, cutoff, shots = [], [], []
    train_size = defaultdict(list)
    seen = set()
    hardware = None
    for relative in runs:
        _, metrics = _load_run(relative)
        hardware = metrics.get("hardware", hardware)
        for record in metrics["records"]:
            seeds = (
                record["split_seed"],
                record["reservoir_seed"],
                record["shot_seed"],
            )
            analysis = record["analysis"]
            if analysis == "readouts":
                key = (record["readout"], *seeds)
                if key in seen:
                    raise EvidenceError(f"{label}: duplicate {key}")
                seen.add(key)
                readouts.append(
                    [
                        record["readout"],
                        *seeds,
                        record["f1_macro"],
                        # The first N = 16 run predates train metrics: stored as null.
                        record.get("train_f1_macro"),
                        record["n_features"],
                    ]
                )
            elif analysis == "cutoff_sweep":
                cutoff.append(
                    [
                        record["readout"],
                        *seeds,
                        record["nsr_cutoff"],
                        record["f1_macro"],
                        record["n_eigentasks_retained"],
                        record["n_eigentasks_total"],
                    ]
                )
            elif analysis == "shots_sweep":
                shots.append([*seeds, record["shots"], record["f1_macro"]])
            elif analysis == "train_size_sweep":
                train_size[(record["split_seed"], record["train_size"])].append(
                    record["f1_macro"]
                )
    counts = defaultdict(int)
    for row in readouts:
        counts[row[0]] += 1
    if set(counts.values()) != {45}:
        raise EvidenceError(f"{label}: expected 45 evaluations per readout, {counts}")
    _check_finite([row[4] for row in readouts], label)
    return {
        "machine": machine,
        "runs": runs,
        "hardware": hardware,
        "readouts_columns": [
            "readout",
            "split_seed",
            "reservoir_seed",
            "shot_seed",
            "f1_macro",
            "train_f1_macro",
            "n_features",
        ],
        "readouts": readouts,
        "cutoff_columns": [
            "readout",
            "split_seed",
            "reservoir_seed",
            "shot_seed",
            "nsr_cutoff",
            "f1_macro",
            "n_retained",
            "n_total",
        ],
        "cutoff": cutoff,
        "shots_columns": [
            "split_seed",
            "reservoir_seed",
            "shot_seed",
            "shots",
            "f1_macro",
        ],
        "shots": shots,
        "train_size_columns": ["split_seed", "train_size", "f1_macro_mean", "n"],
        "train_size": [
            [split, size, float(np.mean(values)), len(values)]
            for (split, size), values in sorted(train_size.items())
        ],
    }


def landsat_classical(relative: str) -> dict:
    """Per-split F1 macro of every classical model (ELM values are init means)."""
    _, metrics = _load_run(relative)
    per_model = defaultdict(dict)
    for split, result in metrics["splits"].items():
        for name, value in result["paper_baselines"].items():
            per_model[f"{name} (untuned)"][split] = value["f1_macro_mean"]
        for name, value in result["matched_elm"].items():
            per_model[f"elm {name}"][split] = value["f1_macro_mean"]
        for name, value in result["tuned"].items():
            per_model[f"{name} (tuned)"][split] = value["f1_macro"]
        per_model["ridge degree-2"][split] = result["ridge_degree2"]["f1_macro"]
    for name, values in per_model.items():
        if len(values) != 5:
            raise EvidenceError(f"{relative} {name}: {len(values)} splits")
        _check_finite(list(values.values()), name)
    return {"run": relative, "f1_macro_per_split": per_model}


def narma(relative: str) -> dict:
    _, metrics = _load_run(relative)
    rows = [
        [
            record["model"],
            record["condition"],
            record["sequence_seed"],
            record.get("reservoir_seed"),
            record["n_features"],
            record["cumulative_r2"],
            record["r2"],
        ]
        for record in metrics["records"]
    ]
    _check_finite([row[5] for row in rows], relative)
    return {
        "run": relative,
        "window": metrics["window"],
        "orders": metrics["orders"],
        "columns": [
            "model",
            "condition",
            "sequence_seed",
            "reservoir_seed",
            "n_features",
            "cumulative_r2",
            "r2_per_order",
        ],
        "records": rows,
    }


def gap(relative: str) -> dict:
    _, metrics = _load_run(relative)
    rows = [
        [
            record["condition"],
            record["readout"],
            record["split_seed"],
            record["reservoir_seed"],
            record["f1_macro"],
            record["train_f1_macro"],
        ]
        for record in metrics["records"]
    ]
    _check_finite([row[4] for row in rows], relative)
    return {
        "run": relative,
        "columns": [
            "condition",
            "readout",
            "split_seed",
            "reservoir_seed",
            "f1_macro",
            "train_f1_macro",
        ],
        "records": rows,
    }


def concentration(relative: str) -> dict:
    _, metrics = _load_run(relative)
    return {"run": relative, "records": metrics["records"]}


def pareto(relative: str) -> dict:
    with (PAPER_DIR / relative / "trials.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 500:
        raise EvidenceError(f"{relative}: {len(rows)} trials, expected 500")
    table = [
        [
            int(row["trial"]),
            row["is_table_I"] == "True",
            *(float(row[c]) for c in PARETO_COLUMNS[2:]),
        ]
        for row in rows
    ]
    _check_finite([value for row in table for value in row[2:]], relative)
    return {"run": relative, "columns": list(PARETO_COLUMNS), "trials": table}


def build_evidence() -> dict:
    return {
        "description": (
            "Per-seed evidence exported from validated run directories by "
            "utils/export_notebook_evidence.py; aggregated numbers in README.md and "
            "results/*.json are means of these records."
        ),
        "landsat_qelm": {
            label: landsat_qelm(label, machine, runs)
            for label, (machine, runs) in LANDSAT_QELM_RUNS.items()
        },
        "landsat_classical": {
            scaling: landsat_classical(run)
            for scaling, run in LANDSAT_CLASSICAL_RUNS.items()
        },
        "narma": {label: narma(run) for label, run in NARMA_RUNS.items()},
        "gap_exact_vs_shots": {scaling: gap(run) for scaling, run in GAP_RUNS.items()},
        "concentration": {
            label: concentration(run) for label, run in CONCENTRATION_RUNS.items()
        },
        "pareto": {str(n): pareto(run) for n, run in PARETO_RUNS.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        evidence = build_evidence()
    except EvidenceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    args.output.write_text(json.dumps(evidence, separators=(",", ":")))
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
