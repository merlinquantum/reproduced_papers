"""Recompute the Phase 3 selection results from stored artifacts (audit P3-07, P3-09).

Usage (from the paper folder)::

    python utils/verify_phase3_selection.py \
        --pareto-runs outdir/pareto_n6/run_A ... \
        --classical-run outdir/run_B \
        --narma-runs outdir/narma_n8/run_C ...

Checks, each compared against the run's own ``metrics.json``:

- E1: trial count, finite objectives, unique trial ids, Table I as trial 0 with the config
  hyperparameters, and Table I's Pareto rank and normalised front distance in every
  projection. Ranks and distances are recomputed here with an independent dominance
  count, not with ``lib.pareto_task``.
- E10 tuned baselines: re-runs the validation-split grid search for every split from the
  run's config snapshot and compares the selected parameters and test F1.
- E2 ESN: re-runs the (spectral radius, input scale) selection on the train tail for every
  sequence and compares the chosen parameters per order and the test R^2.

Writes ``results/phase3_selection_verification.json`` and exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(PAPER_DIR.parents[1]))

from lib.baselines import tune_and_evaluate  # noqa: E402
from lib.data import make_landsat_split  # noqa: E402
from lib.narma_task import esn_r2, narma_sequence_from_config  # noqa: E402
from lib.readout import alpha_grid  # noqa: E402

OBJECTIVE_COLUMNS = {
    "all_objectives": (
        "narma_noiseless",
        "narma_noisy",
        "bond_dimension",
        "variability",
    ),
    "fig2a_narma_noisy_vs_bond": ("narma_noisy", "bond_dimension"),
    "fig2b_variability_vs_bond": ("variability", "bond_dimension"),
}
TABLE_I_KEYS = (
    "input_strength",
    "b_mean",
    "b_std",
    "h_mean",
    "h_std",
    "J_mean",
    "J_std",
)
TOLERANCE = 1e-9


def _completed_metrics(run_dir: Path) -> tuple[dict, dict]:
    status = json.loads((run_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{run_dir} is {status['status']}")
    metrics = json.loads((run_dir / status["metrics_path"]).read_text())
    config = json.loads((run_dir / status["config_path"]).read_text())
    return metrics, config


def _first_front_mask(points: np.ndarray) -> np.ndarray:
    """True where no other point is >= in every column and > in one (maximisation)."""
    dominated_by_count = np.array(
        [
            np.sum(np.all(points >= p, axis=1) & np.any(points > p, axis=1))
            for p in points
        ]
    )
    return dominated_by_count == 0


def _rank(points: np.ndarray, index: int) -> int:
    remaining = np.ones(len(points), dtype=bool)
    rank = 0
    while True:
        indices = np.flatnonzero(remaining)
        front = indices[_first_front_mask(points[indices])]
        if index in front:
            return rank
        remaining[front] = False
        rank += 1


def _distance(points: np.ndarray, index: int) -> float:
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    normalised = (points - low) / span
    front = normalised[_first_front_mask(points)]
    return float(np.min(np.linalg.norm(front - normalised[index], axis=1)))


def verify_pareto(run_dir: Path) -> dict:
    metrics, config = _completed_metrics(run_dir)
    with (run_dir / metrics["trials_path"]).open() as handle:
        rows = list(csv.DictReader(handle))
    trial_ids = [int(row["trial"]) for row in rows]
    table_rows = [i for i, row in enumerate(rows) if row["is_table_I"] == "True"]
    expected_trials = int(config["pareto_search"]["n_trials"])
    checks = {
        "n_rows": len(rows),
        "n_trials_config": expected_trials,
        "n_trials_metrics": metrics["n_trials"],
        "count_matches": len(rows) == expected_trials == metrics["n_trials"],
        "trial_ids_unique_0_to_n_minus_1": sorted(trial_ids) == list(range(len(rows))),
        "all_objectives_finite": all(
            np.isfinite(float(row[key]))
            for row in rows
            for key in OBJECTIVE_COLUMNS["all_objectives"]
        ),
        "table_I_is_single_trial_0": table_rows == [0] and trial_ids[0] == 0,
        "table_I_matches_config": all(
            abs(float(rows[0][key]) - float(config["hyperparameters"][key])) < TOLERANCE
            for key in TABLE_I_KEYS
        ),
        "projections": {},
    }
    for name, columns in OBJECTIVE_COLUMNS.items():
        points = np.array([[float(row[c]) for c in columns] for row in rows])
        rank, distance = _rank(points, 0), _distance(points, 0)
        stored = metrics["analysis"][name]
        checks["projections"][name] = {
            "rank": rank,
            "stored_rank": stored["reference_rank"],
            "front_distance": distance,
            "stored_front_distance": stored["reference_front_distance"],
            "match": rank == stored["reference_rank"]
            and abs(distance - stored["reference_front_distance"]) < TOLERANCE,
        }
    checks["ok"] = all(
        value for key, value in checks.items() if isinstance(value, bool)
    ) and all(p["match"] for p in checks["projections"].values())
    return {"run": str(run_dir), "n_qubits": metrics["n_qubits"], **checks}


def verify_tuned_baselines(run_dir: Path) -> dict:
    metrics, config = _completed_metrics(run_dir)
    landsat_cfg = config["dataset"]["landsat"]
    # Snapshots written before the input_scaling option existed used min-max scaling, the
    # only behaviour at the time (same rule as landsat_task.load_saved_probabilities).
    input_scaling_inferred = "input_scaling" not in landsat_cfg
    landsat_cfg.setdefault("input_scaling", "minmax")
    exp = config["landsat_classical"]["tuned_baselines"]
    entries = []
    for split_seed, split_results in metrics["splits"].items():
        data = make_landsat_split(config, int(split_seed))
        for name, grid in exp["grids"].items():
            result = tune_and_evaluate(
                name,
                grid,
                data.train_raw,
                data.train_y,
                data.test_raw,
                data.test_y,
                float(exp["validation_fraction"]),
                int(split_seed),
            )
            stored = split_results["tuned"][name]
            entries.append(
                {
                    "split_seed": int(split_seed),
                    "model": name,
                    "n_candidates": int(np.prod([len(v) for v in grid.values()])),
                    "params": result["params"],
                    "stored_params": stored["params"],
                    "test_f1_macro": result["f1_macro"],
                    "stored_test_f1_macro": stored["f1_macro"],
                    "match": result["params"] == stored["params"]
                    and abs(result["f1_macro"] - stored["f1_macro"]) < TOLERANCE,
                }
            )
    return {
        "run": str(run_dir),
        "input_scaling_inferred_as_minmax": input_scaling_inferred,
        "entries": entries,
        "ok": all(entry["match"] for entry in entries),
    }


def verify_esn(run_dir: Path) -> dict:
    metrics, config = _completed_metrics(run_dir)
    exp = config["narma_scaling"]
    alphas = alpha_grid(config["readout"])
    window = int(metrics["window"])
    stored_by_seed = {
        record["sequence_seed"]: record
        for record in metrics["records"]
        if record["model"] == "esn"
    }
    entries = []
    for sequence_seed in exp["sequence_seeds"]:
        sequence = narma_sequence_from_config(
            config, window, np.random.default_rng(int(sequence_seed))
        )
        r2_values, chosen = esn_r2(sequence, exp["esn"], alphas, int(sequence_seed))
        stored = stored_by_seed[int(sequence_seed)]
        entries.append(
            {
                "sequence_seed": int(sequence_seed),
                "params_match": chosen == stored["chosen_params"],
                "max_abs_r2_difference": float(
                    np.max(np.abs(np.array(r2_values) - np.array(stored["r2"])))
                ),
            }
        )
    for entry in entries:
        entry["match"] = entry["params_match"] and entry["max_abs_r2_difference"] < 1e-6
    return {
        "run": str(run_dir),
        "n_qubits": metrics["n_qubits"],
        "n_candidates": len(exp["esn"]["spectral_radius"])
        * len(exp["esn"]["input_scale"]),
        "entries": entries,
        "ok": all(entry["match"] for entry in entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pareto-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--classical-run", type=Path, required=True)
    parser.add_argument("--narma-runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_DIR / "results" / "phase3_selection_verification.json",
    )
    args = parser.parse_args()
    report = {
        "e1_pareto": [verify_pareto(run) for run in args.pareto_runs],
        "e10_tuned_baselines": verify_tuned_baselines(args.classical_run),
        "e2_esn": [verify_esn(run) for run in args.narma_runs],
    }
    report["ok"] = (
        all(entry["ok"] for entry in report["e1_pareto"])
        and report["e10_tuned_baselines"]["ok"]
        and all(entry["ok"] for entry in report["e2_esn"])
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for entry in report["e1_pareto"]:
        print(f"E1 N={entry['n_qubits']}: ok={entry['ok']}")
    print(f"E10 tuned: ok={report['e10_tuned_baselines']['ok']}")
    for entry in report["e2_esn"]:
        print(f"E2 ESN N={entry['n_qubits']}: ok={entry['ok']}")
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
