"""Phase 5 comparison: qubit QNN (E-B), classical baselines (E-F), photonic (E-P).

Implements the protocol frozen in ``LOG.md`` under "Phase 5 Comparison Plan":
on Rice (PCA 7) and Glass (LDA 5) with MinMax inputs, each system family's
configuration with the highest seed-mean validation accuracy is selected and
its test accuracy reported, with the family's whole-grid mean alongside. It
also writes the C2 COBYLA-vs-SPSA timing test paired over E-B configurations.

Usage (from the repository root)::

    python papers/QNN_hyperparameters/utils/compare_systems.py \
        papers/QNN_hyperparameters/outdir/sweeps

Writes ``results/phase5_*.csv`` and ``results/phase5_comparison.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

from QNN_hyperparameters.lib.sweep_analysis import load_sweep  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "results"
PCA_SIGNS_BEFORE_OPTION = "installed"
FLOAT_TOLERANCE = 1e-12

# Dataset label per family; E-F names its datasets "<data>-<features>".
DATASETS = {"rice": "rice-pca", "glass": "glass-lda"}

# Factors defining one configuration (seeds are aggregated over) per family.
FAMILIES = {
    "qubit QNN (E-B)": {
        "sweep": "E-B_mechanism",
        "config_columns": ["init", "circuit", "optimizer"],
        "data_column_value": lambda data: data,
        "provenance": "local re-run (Qiskit, CPU, 1024 shots)",
        "scope": "partial: 4 of 84 circuits, 2 of 3 optimizers, 3 seeds",
    },
    "classical (E-F)": {
        "sweep": "E-F_baselines",
        "config_columns": ["model"],
        "data_column_value": lambda data: DATASETS[data],
        "provenance": "local run (scikit-learn, CPU, 5-fold CV on train)",
        "scope": "planned baseline grid, 10 seeds",
    },
    "photonic MerLin (E-P, adaptation)": {
        "sweep": "E-P_photonic_mechanism",
        "config_columns": ["init", "photons", "depth"],
        "data_column_value": lambda data: data,
        "provenance": "local simulation (MerLin 0.4.1, L4 GPU, exact expectations)",
        "scope": "adaptation, simulated, 3 seeds",
    },
}


def _run_extras(sweep_dir: Path, run_id: str) -> dict:
    """Read the fields ``load_sweep`` does not keep: capacity, data identity, timing."""

    run_dir = sweep_dir / "runs" / run_id
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    started = datetime.fromisoformat(status["started_at"].replace("Z", "+00:00"))
    completed = datetime.fromisoformat(status["completed_at"].replace("Z", "+00:00"))
    dataset = config["dataset"]
    return {
        "run_id": run_id,
        # QNN and photonic report trainable weights; classical models report
        # fitted parameters (SVM: support-vector coefficients and intercepts).
        "capacity": metrics.get("num_weights", metrics.get("parameters")),
        "preprocessing": status["dataset"]["preprocessing"],
        "splits": json.dumps(status["dataset"]["splits"], sort_keys=True),
        # E-B and E-F ran before `dataset.pca_signs` existed, when the installed
        # scikit-learn convention was the only behaviour; their clean-commit
        # reruns record "installed" and reproduce identical tables (LOG.md).
        "pca_signs": dataset.get("pca_signs", PCA_SIGNS_BEFORE_OPTION),
        "fit_seconds_reported": metrics.get("fit_seconds"),
        "status_seconds": (completed - started).total_seconds(),
    }


def load_family(sweeps_root: Path, name: str) -> pd.DataFrame:
    """Load one family's complete sweep with capacity, data identity and timing."""

    spec = FAMILIES[name]
    sweep_dir = sweeps_root / spec["sweep"]
    frame, coverage = load_sweep(sweep_dir)
    if not coverage["complete"]:
        raise SystemExit(
            f"{spec['sweep']}: PARTIAL SWEEP, refusing to compare: {coverage}"
        )
    extras = pd.DataFrame(
        [_run_extras(sweep_dir, run_id) for run_id in frame["run_id"]]
    )
    frame = frame.merge(extras, on="run_id", validate="one_to_one")
    frame["family"] = name
    frame["expected_runs_in_sweep"] = coverage["expected_runs"]
    return frame


def check_identical_data(families: dict[str, pd.DataFrame]) -> dict:
    """Fail unless every family used the same splits, preprocessing and PCA signs per dataset."""

    report = {}
    for data in DATASETS:
        for scaling in ("raw", "minmax"):
            identities = {}
            for name, frame in families.items():
                subset = frame[
                    (frame["data"] == FAMILIES[name]["data_column_value"](data))
                    & (frame["scaling"] == scaling)
                ]
                identities[name] = sorted(
                    set(
                        zip(
                            subset["preprocessing"],
                            subset["splits"],
                            subset["pca_signs"].astype(str),
                        )
                    )
                )
            distinct = {json.dumps(value) for value in identities.values()}
            if len(distinct) != 1 or len(next(iter(identities.values()))) != 1:
                raise SystemExit(
                    f"data identity differs for {data}/{scaling}: {identities}"
                )
            report[f"{data}/{scaling}"] = next(iter(identities.values()))[0]
    return report


def select_by_validation(
    frame: pd.DataFrame, config_columns: list[str]
) -> pd.DataFrame:
    """Seed-aggregate each configuration and sort by validation mean, best first."""

    grouped = frame.groupby(config_columns)
    summary = grouped.agg(
        runs=("seed", "size"),
        val_mean=("val_accuracy", "mean"),
        val_sd=("val_accuracy", "std"),
        test_mean=("test_accuracy", "mean"),
        test_sd=("test_accuracy", "std"),
        test_f1_mean=("test_f1", "mean"),
        capacity=("capacity", "max"),
    ).reset_index()
    return summary.sort_values("val_mean", ascending=False, kind="mergesort")


def comparison_table(families: dict[str, pd.DataFrame], identity: dict) -> pd.DataFrame:
    """One row per (dataset, scaling, family): selected configuration plus whole-grid mean."""

    rows = []
    for data in DATASETS:
        for scaling in ("minmax", "raw"):
            for name, frame in families.items():
                spec = FAMILIES[name]
                subset = frame[
                    (frame["data"] == spec["data_column_value"](data))
                    & (frame["scaling"] == scaling)
                ]
                ranking = select_by_validation(subset, spec["config_columns"])
                selected = ranking.iloc[0]
                timing_column = (
                    "fit_seconds_reported"
                    if subset["fit_seconds_reported"].notna().all()
                    else "status_seconds"
                )
                rows.append(
                    {
                        "data": data,
                        "preprocessing": identity[f"{data}/{scaling}"][0],
                        "scaling": scaling,
                        "system": name,
                        "provenance": spec["provenance"],
                        "scope": spec["scope"],
                        "runs_observed": len(subset),
                        "configurations": len(ranking),
                        "selected_configuration": ", ".join(
                            f"{column}={selected[column]}"
                            for column in spec["config_columns"]
                        ),
                        "selected_runs": int(selected["runs"]),
                        "selected_val_mean": selected["val_mean"],
                        "selected_test_mean": selected["test_mean"],
                        "selected_test_sd": selected["test_sd"],
                        "selected_test_f1_mean": selected["test_f1_mean"],
                        "grid_test_mean": subset["test_accuracy"].mean(),
                        "grid_test_sd": subset["test_accuracy"].std(),
                        "capacity_min": int(subset["capacity"].min()),
                        "capacity_max": int(subset["capacity"].max()),
                        "selected_capacity": int(selected["capacity"]),
                        "seconds_per_run_mean": subset[timing_column].mean(),
                        "seconds_per_run_max": subset[timing_column].max(),
                        "timing_source": timing_column,
                    }
                )
    return pd.DataFrame(rows)


def pairwise_separation(
    table: pd.DataFrame, test_sizes: dict[str, int]
) -> pd.DataFrame:
    """Apply the frozen tie rule to every pair of systems on MinMax test accuracy."""

    rows = []
    minmax = table[table["scaling"] == "minmax"]
    for data, group in minmax.groupby("data", sort=False):
        records = group.to_dict("records")
        for index, first in enumerate(records):
            for second in records[index + 1 :]:
                difference = first["selected_test_mean"] - second["selected_test_mean"]
                one_sample = 1.0 / test_sizes[data]
                sd_threshold = 2.0 * max(
                    first["selected_test_sd"], second["selected_test_sd"]
                )
                rows.append(
                    {
                        "data": data,
                        "system_a": first["system"],
                        "system_b": second["system"],
                        "test_mean_a_minus_b": difference,
                        "one_test_sample": one_sample,
                        "two_x_larger_sd": sd_threshold,
                        "separated": bool(
                            abs(difference) > max(one_sample, sd_threshold)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def matched_capacity(
    qubit: pd.DataFrame, classical: pd.DataFrame, test_sizes: dict[str, int]
) -> pd.DataFrame:
    """Phase 1 matching criterion: the MLP parameter-matched to the selected QNN, MinMax inputs."""

    rows = []
    for data, label in DATASETS.items():
        qubit_subset = qubit[(qubit["data"] == data) & (qubit["scaling"] == "minmax")]
        selected = select_by_validation(
            qubit_subset, FAMILIES["qubit QNN (E-B)"]["config_columns"]
        )
        selected = selected.iloc[0]
        mlp = classical[
            (classical["data"] == label)
            & (classical["scaling"] == "minmax")
            & (classical["model"] == "mlp")
        ]
        difference = mlp["test_accuracy"].mean() - selected["test_mean"]
        threshold = max(
            1.0 / test_sizes[data],
            2.0 * max(mlp["test_accuracy"].std(), selected["test_sd"]),
        )
        rows.append(
            {
                "data": data,
                "qubit_capacity": int(selected["capacity"]),
                "mlp_capacity": int(mlp["capacity"].max()),
                "qubit_runs": int(selected["runs"]),
                "mlp_runs": len(mlp),
                "qubit_test_mean": selected["test_mean"],
                "qubit_test_sd": selected["test_sd"],
                "mlp_test_mean": mlp["test_accuracy"].mean(),
                "mlp_test_sd": mlp["test_accuracy"].std(),
                "mlp_minus_qubit": difference,
                "separation_threshold": threshold,
                "separated": bool(abs(difference) > threshold),
            }
        )
    return pd.DataFrame(rows)


def selection_sensitivity(
    families: dict[str, pd.DataFrame], identity: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidates within one validation sample of each family's best, and how the separations move.

    The frozen protocol has no selection tie rule (amendment of 2026-10-29,
    LOG.md). One validation sample (1/n_val) is the smallest accuracy
    difference the validation split can resolve, so every candidate within it
    of the best is treated as near-tied, and each family pair is re-scored
    with the frozen separation rule for every near-tied combination.
    """

    near_tied_rows, sensitivity_rows = [], []
    for data in DATASETS:
        sizes = json.loads(identity[f"{data}/minmax"][1])
        one_val_sample, one_test_sample = 1.0 / sizes["val"], 1.0 / sizes["test"]
        candidates = {}
        for name, frame in families.items():
            spec = FAMILIES[name]
            subset = frame[
                (frame["data"] == spec["data_column_value"](data))
                & (frame["scaling"] == "minmax")
            ]
            ranking = select_by_validation(subset, spec["config_columns"])
            best = ranking["val_mean"].iloc[0]
            near = ranking[
                ranking["val_mean"] >= best - one_val_sample - FLOAT_TOLERANCE
            ]
            candidates[name] = near
            for row in near.itertuples():
                near_tied_rows.append(
                    {
                        "data": data,
                        "system": name,
                        "configuration": ", ".join(
                            f"{column}={getattr(row, column)}"
                            for column in spec["config_columns"]
                        ),
                        "val_mean": row.val_mean,
                        "val_gap_to_best": best - row.val_mean,
                        # Seed means of identical accuracies can differ by float rounding.
                        "exact_tie_with_best": bool(
                            abs(best - row.val_mean) < FLOAT_TOLERANCE
                        ),
                        "test_mean": row.test_mean,
                        "test_sd": row.test_sd,
                    }
                )
        names = list(candidates)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                outcomes = []
                for a in candidates[first].itertuples():
                    for b in candidates[second].itertuples():
                        difference = a.test_mean - b.test_mean
                        threshold = max(
                            one_test_sample, 2.0 * max(a.test_sd, b.test_sd)
                        )
                        outcomes.append(
                            (abs(difference) > threshold, np.sign(difference))
                        )
                sensitivity_rows.append(
                    {
                        "data": data,
                        "system_a": first,
                        "system_b": second,
                        "combinations": len(outcomes),
                        "separated_a_higher": sum(
                            sep and sign > 0 for sep, sign in outcomes
                        ),
                        "separated_b_higher": sum(
                            sep and sign < 0 for sep, sign in outcomes
                        ),
                        "not_separated": sum(not sep for sep, _ in outcomes),
                    }
                )
    return pd.DataFrame(near_tied_rows), pd.DataFrame(sensitivity_rows)


def optimizer_timing(qubit: pd.DataFrame) -> pd.DataFrame:
    """C2 timing: COBYLA vs SPSA paired over identical E-B configurations and seeds."""

    keys = ["data", "scaling", "init", "circuit", "seed"]
    paired = qubit.pivot_table(
        index=keys, columns="optimizer", values=["fit_seconds_reported", "evaluations"]
    ).dropna()
    rows = []
    for data, group in paired.groupby(level="data"):
        cobyla = group[("fit_seconds_reported", "cobyla")].to_numpy()
        spsa = group[("fit_seconds_reported", "spsa")].to_numpy()
        rows.append(
            {
                "data": data,
                "pairs": len(group),
                "cobyla_seconds_mean": cobyla.mean(),
                "spsa_seconds_mean": spsa.mean(),
                "cobyla_minus_spsa_seconds_mean": float(np.mean(cobyla - spsa)),
                "cobyla_evaluations_mean": group[("evaluations", "cobyla")].mean(),
                "spsa_evaluations_mean": group[("evaluations", "spsa")].mean(),
                "wilcoxon_p": float(wilcoxon(cobyla, spsa).pvalue),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sweeps_root", type=Path, help="directory holding the sweep folders"
    )
    args = parser.parse_args()

    families = {name: load_family(args.sweeps_root, name) for name in FAMILIES}
    identity = check_identical_data(families)
    table = comparison_table(families, identity)
    test_sizes = {
        data: json.loads(identity[f"{data}/minmax"][1])["test"] for data in DATASETS
    }
    separation = pairwise_separation(table, test_sizes)
    timing = optimizer_timing(families["qubit QNN (E-B)"])
    matched = matched_capacity(
        families["qubit QNN (E-B)"], families["classical (E-F)"], test_sizes
    )
    near_tied, sensitivity = selection_sensitivity(families, identity)

    RESULTS_DIR.mkdir(exist_ok=True)
    outputs = {
        "systems": RESULTS_DIR / "phase5_systems.csv",
        "separation": RESULTS_DIR / "phase5_separation.csv",
        "matched_capacity": RESULTS_DIR / "phase5_matched_capacity.csv",
        "optimizer_timing": RESULTS_DIR / "phase5_optimizer_timing.csv",
        "near_tied_selection": RESULTS_DIR / "phase5_near_tied_selection.csv",
        "separation_sensitivity": RESULTS_DIR / "phase5_separation_sensitivity.csv",
    }
    table.to_csv(outputs["systems"], index=False)
    separation.to_csv(outputs["separation"], index=False)
    matched.to_csv(outputs["matched_capacity"], index=False)
    timing.to_csv(outputs["optimizer_timing"], index=False)
    near_tied.to_csv(outputs["near_tied_selection"], index=False)
    sensitivity.to_csv(outputs["separation_sensitivity"], index=False)
    summary = {
        "protocol": "LOG.md, Phase 5 Comparison Plan (2026-10-29T11:30Z)",
        "data_identity": identity,
        "runs_per_family": {name: len(frame) for name, frame in families.items()},
        "outputs": {key: path.name for key, path in outputs.items()},
    }
    (RESULTS_DIR / "phase5_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.drop(columns=["provenance", "scope"]).to_string(index=False))
        print(separation.to_string(index=False))
        print(matched.to_string(index=False))
        print(timing.to_string(index=False))
        print(near_tied.to_string(index=False))
        print(sensitivity.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
