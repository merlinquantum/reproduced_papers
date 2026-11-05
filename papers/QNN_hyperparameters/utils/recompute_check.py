"""Independent recalculation of every sweep table in ``results/``.

Reads each run's ``metrics.json`` directly (not ``summary.json`` and not
``lib.sweep_analysis``), recomputes the reported statistics, and prints the
largest absolute discrepancy per table and column. Writes nothing.

Bootstrap confidence intervals are not recomputed: they depend on the
analysis code's random stream, and the interval endpoints are only used for
the ``negligible`` flag, which the Holm p-values already bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

PAPER_DIR = Path(__file__).resolve().parents[1]
SWEEPS = PAPER_DIR / "outdir" / "sweeps"
RESULTS = PAPER_DIR / "results"

# Run ids abbreviate names that the released files and tables spell out.
OPTIMIZER_NAMES = {"cobyla": "cobyla", "spsa": "spsa", "nm": "neldermead"}
ARM_NAMES = {"uniformarm": "uniform", "betaarm": "beta"}
MODEL_NAMES = {"lr": "logistic_regression", "svm": "rbf_svm", "mlp": "mlp"}
REPLICATE_ARMS = {"uniform": "normal", "beta": "beta_mu"}


def load_runs(sweep: str, fields: list[str]) -> pd.DataFrame:
    """One row per run: run-id fields plus val and test accuracy."""

    rows = []
    for metrics_path in sorted((SWEEPS / sweep / "runs").glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        row = dict(zip(fields, metrics_path.parent.name.split("_"), strict=True))
        row["val_accuracy"] = metrics["val"]["accuracy"]
        row["test_accuracy"] = metrics["test"]["accuracy"]
        rows.append(row)
    return pd.DataFrame(rows)


def holm(p_values: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values, in the input order."""

    order = p_values.sort_values().index
    count = len(p_values)
    adjusted, running = {}, 0.0
    for rank, label in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[label]))
        adjusted[label] = running
    return pd.Series(adjusted)[p_values.index]


def paired_row(data, effect, comparison, differences):
    differences = np.asarray(differences)
    p_value = (
        1.0 if np.allclose(differences, 0) else float(wilcoxon(differences).pvalue)
    )
    return {
        "data": data,
        "effect": effect,
        "comparison": comparison,
        "n_pairs": len(differences),
        "mean_diff": differences.mean(),
        "wilcoxon_p": p_value,
    }


def mechanism_rows(frame, pair_keys, metric):
    """Scaling at fixed init and init at fixed scaling, per dataset."""

    rows = []
    for data, subset in frame.groupby("data"):
        wide = subset.pivot_table(
            index=pair_keys, columns=["scaling", "init"], values=metric
        )
        for init in ("default", "beta", "zeros"):
            diff = (wide[("minmax", init)] - wide[("raw", init)]).dropna()
            rows.append(
                paired_row(data, "scaling", f"minmax - raw | init={init}", diff)
            )
        for scaling in ("raw", "minmax"):
            for init in ("beta", "zeros"):
                diff = (wide[(scaling, init)] - wide[(scaling, "default")]).dropna()
                rows.append(
                    paired_row(
                        data, "init", f"{init} - default | scaling={scaling}", diff
                    )
                )
    return pd.DataFrame(rows)


def resource_rows(frame, metric):
    rows = []
    for data, subset in frame.groupby("data"):
        for factor, high, low, others in (
            ("photons", "n4", "n2", ["init", "depth", "seed"]),
            ("depth", "d2", "d1", ["init", "photons", "seed"]),
        ):
            wide = subset.pivot_table(
                index=others, columns=["scaling", factor], values=metric
            )
            for scaling in ("raw", "minmax"):
                diff = (wide[(scaling, high)] - wide[(scaling, low)]).dropna()
                rows.append(
                    paired_row(
                        data, factor, f"{high} - {low} | scaling={scaling}", diff
                    )
                )
    return pd.DataFrame(rows)


def with_holm(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["holm_p"] = holm(table["wilcoxon_p"])
    return table


def effects_tables():
    mechanism = load_runs(
        "E-B_mechanism", ["data", "scaling", "init", "circuit", "optimizer", "seed"]
    )
    photonic = load_runs(
        "E-P_photonic_mechanism",
        ["data", "scaling", "init", "photons", "depth", "seed"],
    )
    for split in ("val", "test"):
        metric = f"{split}_accuracy"
        yield (
            f"E-B effects ({split})",
            with_holm(
                mechanism_rows(mechanism, ["circuit", "optimizer", "seed"], metric)
            ),
            RESULTS / f"E-B_mechanism_effects_{split}.csv",
        )
        photonic_table = pd.concat(
            [
                with_holm(
                    mechanism_rows(photonic, ["photons", "depth", "seed"], metric)
                ),
                with_holm(resource_rows(photonic, metric)),
            ]
        )
        yield (
            f"E-P effects ({split})",
            photonic_table,
            RESULTS / f"E-P_photonic_mechanism_effects_{split}.csv",
        )


def grid_runs(sweep: str) -> pd.DataFrame:
    runs = load_runs(sweep, ["arm", "optimizer", "grid", "seed"])
    runs["init"] = runs["arm"].map(ARM_NAMES)
    runs["optimizer"] = runs["optimizer"].map(OPTIMIZER_NAMES)
    runs["index"] = runs["grid"].str[1:].astype(int)
    return runs


def agreement_table(runs, released, noisy):
    rice = released[(released["dataset"] == "rice") & (released["noisy"] == noisy)]
    rows = []
    for (init, optimizer), ours in runs.groupby(["init", "optimizer"]):
        same_optimizer = rice[rice["optimizer"] == optimizer]
        released_arm = same_optimizer[same_optimizer["init"] == init]
        replicate_arm = same_optimizer[same_optimizer["init"] == REPLICATE_ARMS[init]]
        # The replicate floor covers only the configurations we ran (12 of 84
        # for the noisy subsets), as predeclared for the agreement tables.
        ours_by_config = ours.set_index("index")["val_accuracy"]
        paired = pd.DataFrame(
            {
                "ours": ours_by_config,
                "released": released_arm.set_index("index")["accuracy"],
                "replicate": replicate_arm.set_index("index")["accuracy"],
            }
        ).reindex(ours_by_config.index)
        both = paired[["ours", "released"]].dropna()
        gap = (both["ours"] - both["released"]).abs()
        floor = paired[["released", "replicate"]].dropna()
        floor_gap = (floor["released"] - floor["replicate"]).abs()
        row = {
            "init": init,
            "optimizer": optimizer,
            "n_pairs": len(both),
            "mean_diff": (both["ours"] - both["released"]).mean(),
            "median_abs_diff": gap.median(),
            "pearson": np.corrcoef(both["ours"], both["released"])[0, 1],
        }
        if len(floor) >= 10:
            threshold = floor_gap.quantile(0.95)
            row["within_floor"] = (gap <= threshold).mean()
            row["floor_mwu_p"] = mannwhitneyu(
                gap, floor_gap, alternative="greater"
            ).pvalue
        rows.append(row)
    return pd.DataFrame(rows)


def noise_table(noisy_runs, noiseless_runs, released):
    rice = released[released["dataset"] == "rice"]
    rows = []
    for (init, optimizer), ours in noisy_runs.groupby(["init", "optimizer"]):
        reference = rice[(rice["init"] == init) & (rice["optimizer"] == optimizer)]
        quiet = noiseless_runs[
            (noiseless_runs["init"] == init)
            & (noiseless_runs["optimizer"] == optimizer)
        ]
        paired = pd.DataFrame(
            {
                "ours_noisy": ours.set_index("index")["val_accuracy"],
                "ours_noiseless": quiet.set_index("index")["val_accuracy"],
                "released_noisy": reference[reference["noisy"]].set_index("index")[
                    "accuracy"
                ],
                "released_noiseless": reference[~reference["noisy"]].set_index("index")[
                    "accuracy"
                ],
            }
        ).dropna()
        if paired.empty:
            continue
        rows.append(
            {
                "init": init,
                "optimizer": optimizer,
                "n_configs": len(paired),
                "ours_noise_change": (
                    paired["ours_noisy"] - paired["ours_noiseless"]
                ).mean(),
                "released_noise_change": (
                    paired["released_noisy"] - paired["released_noiseless"]
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def grid_tables(released):
    sweeps = {
        name: grid_runs(name)
        for name in (
            "E-C_rice_grid",
            "E-C2_rice_grid_author_signs",
            "E-D_rice_noisy",
            "E-D2_rice_noisy_author_signs",
        )
    }
    for name, noisy in (
        ("E-C_rice_grid", False),
        ("E-C2_rice_grid_author_signs", False),
        ("E-D_rice_noisy", True),
        ("E-D2_rice_noisy_author_signs", True),
    ):
        yield (
            f"{name} agreement",
            agreement_table(sweeps[name], released, noisy),
            RESULTS / f"{name}_agreement.csv",
        )
    for noisy_name, noiseless_name in (
        ("E-D_rice_noisy", "E-C_rice_grid"),
        ("E-D2_rice_noisy_author_signs", "E-C2_rice_grid_author_signs"),
    ):
        yield (
            f"{noisy_name} noise effect",
            noise_table(sweeps[noisy_name], sweeps[noiseless_name], released),
            RESULTS / f"{noisy_name}_noise_effect.csv",
        )


def baseline_table():
    runs = load_runs("E-F_baselines", ["data", "scaling", "model", "seed"])
    runs["model"] = runs["model"].map(MODEL_NAMES)
    table = (
        runs.groupby(["data", "scaling", "model"])
        .agg(
            n=("val_accuracy", "count"),
            val_mean=("val_accuracy", "mean"),
            val_sd=("val_accuracy", "std"),
            test_mean=("test_accuracy", "mean"),
            test_sd=("test_accuracy", "std"),
        )
        .reset_index()
    )
    yield "E-F baselines", table, RESULTS / "E-F_baselines_baselines_vs_qnn.csv"


PHASE5_FAMILIES = {
    "qubit QNN (E-B)": (
        "E-B_mechanism",
        ["data", "scaling", "init", "circuit", "optimizer", "seed"],
        ["init", "circuit", "optimizer"],
    ),
    "classical (E-F)": (
        "E-F_baselines",
        ["data", "scaling", "model", "seed"],
        ["model"],
    ),
    "photonic MerLin (E-P, adaptation)": (
        "E-P_photonic_mechanism",
        ["data", "scaling", "init", "photons", "depth", "seed"],
        ["init", "photons", "depth"],
    ),
}
PHASE5_DATASETS = {"rice": "rice-pca", "glass": "glass-lda"}


def phase5_split_sizes() -> dict[str, dict[str, int]]:
    """Validation and test sizes per dataset, read from one E-B run status each."""

    sizes = {}
    for data in PHASE5_DATASETS:
        status_path = next(
            (SWEEPS / "E-B_mechanism" / "runs").glob(f"{data}_minmax_*/run_status.json")
        )
        sizes[data] = json.loads(status_path.read_text())["dataset"]["splits"]
    return sizes


def phase5_candidates(name: str, data: str, scaling: str) -> pd.DataFrame:
    """Seed-aggregated configurations, best validation mean first.

    Exact ties (within 1e-12) break on the configuration fields in
    alphabetical order, which is the order the reported selection resolves.
    """

    sweep, fields, config = PHASE5_FAMILIES[name]
    runs = load_runs(sweep, fields)
    if "model" in runs:
        runs["model"] = runs["model"].map(MODEL_NAMES)
    label = PHASE5_DATASETS[data] if sweep == "E-F_baselines" else data
    runs = runs[(runs["data"] == label) & (runs["scaling"] == scaling)]
    table = (
        runs.groupby(config)
        .agg(
            runs=("val_accuracy", "count"),
            val_mean=("val_accuracy", "mean"),
            test_mean=("test_accuracy", "mean"),
            test_sd=("test_accuracy", "std"),
        )
        .reset_index()
    )
    table["configuration"] = table[config].apply(
        lambda row: ", ".join(f"{column}={row[column]}" for column in config), axis=1
    )
    table["val_rank_key"] = table["val_mean"].round(12)
    table = table.sort_values(
        ["val_rank_key", *config], ascending=[False, *[True] * len(config)]
    )
    return table.reset_index(drop=True), runs


def phase5_tables():
    sizes = phase5_split_sizes()
    systems, near_rows, separation_rows, sensitivity_rows, matched_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    for data in PHASE5_DATASETS:
        one_val, one_test = 1 / sizes[data]["val"], 1 / sizes[data]["test"]
        selected, near = {}, {}
        for scaling in ("minmax", "raw"):
            for name in PHASE5_FAMILIES:
                ranking, runs = phase5_candidates(name, data, scaling)
                best = ranking.iloc[0]
                systems.append(
                    {
                        "data": data,
                        "scaling": scaling,
                        "system": name,
                        "runs_observed": len(runs),
                        "selected_val_mean": best["val_mean"],
                        "selected_test_mean": best["test_mean"],
                        "selected_test_sd": best["test_sd"],
                        "grid_test_mean": runs["test_accuracy"].mean(),
                        "grid_test_sd": runs["test_accuracy"].std(),
                    }
                )
                if scaling == "minmax":
                    selected[name] = best
                    near[name] = ranking[
                        ranking["val_mean"] >= best["val_mean"] - one_val - 1e-12
                    ]
                    for row in near[name].itertuples():
                        near_rows.append(
                            {
                                "data": data,
                                "system": name,
                                "configuration": row.configuration,
                                "val_mean": row.val_mean,
                                "test_mean": row.test_mean,
                                "test_sd": row.test_sd,
                            }
                        )
        names = list(PHASE5_FAMILIES)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                a, b = selected[first], selected[second]
                difference = a["test_mean"] - b["test_mean"]
                threshold = max(one_test, 2 * max(a["test_sd"], b["test_sd"]))
                separation_rows.append(
                    {
                        "data": data,
                        "system_a": first,
                        "system_b": second,
                        "test_mean_a_minus_b": difference,
                        "separated": int(abs(difference) > threshold),
                    }
                )
                counts = {"higher_a": 0, "higher_b": 0, "none": 0}
                for x in near[first].itertuples():
                    for y in near[second].itertuples():
                        gap = x.test_mean - y.test_mean
                        if abs(gap) <= max(one_test, 2 * max(x.test_sd, y.test_sd)):
                            counts["none"] += 1
                        else:
                            counts["higher_a" if gap > 0 else "higher_b"] += 1
                sensitivity_rows.append(
                    {
                        "data": data,
                        "system_a": first,
                        "system_b": second,
                        "combinations": sum(counts.values()),
                        "separated_a_higher": counts["higher_a"],
                        "separated_b_higher": counts["higher_b"],
                        "not_separated": counts["none"],
                    }
                )
        _, classical_runs = phase5_candidates("classical (E-F)", data, "minmax")
        mlp = classical_runs[classical_runs["model"] == "mlp"]["test_accuracy"]
        qubit = selected["qubit QNN (E-B)"]
        matched_rows.append(
            {
                "data": data,
                "qubit_test_mean": qubit["test_mean"],
                "mlp_test_mean": mlp.mean(),
                "mlp_minus_qubit": mlp.mean() - qubit["test_mean"],
                "separated": int(
                    abs(mlp.mean() - qubit["test_mean"])
                    > max(one_test, 2 * max(mlp.std(), qubit["test_sd"]))
                ),
            }
        )
    yield "Phase 5 systems", pd.DataFrame(systems), RESULTS / "phase5_systems.csv"
    yield (
        "Phase 5 separation",
        pd.DataFrame(separation_rows),
        RESULTS / "phase5_separation.csv",
    )
    yield (
        "Phase 5 matched capacity",
        pd.DataFrame(matched_rows),
        RESULTS / "phase5_matched_capacity.csv",
    )
    yield (
        "Phase 5 near-tied selection",
        pd.DataFrame(near_rows),
        RESULTS / "phase5_near_tied_selection.csv",
    )
    yield (
        "Phase 5 separation sensitivity",
        pd.DataFrame(sensitivity_rows),
        RESULTS / "phase5_separation_sensitivity.csv",
    )


def compare(name, recomputed, reported_path) -> float:
    """Print per-column max |recomputed - reported|; return the overall max."""

    reported = pd.read_csv(reported_path)
    keys = [
        column
        for column in (
            "data",
            "effect",
            "comparison",
            "init",
            "optimizer",
            "scaling",
            "model",
            "system",
            "system_a",
            "system_b",
            "configuration",
        )
        if column in recomputed.columns
    ]
    merged = recomputed.merge(
        reported, on=keys, how="outer", suffixes=("", "_reported"), indicator=True
    )
    unmatched = merged[merged["_merge"] != "both"]
    # Reported rows with no data at all (e.g. a noise row whose noiseless arm
    # was never run) are not discrepancies.
    if "n_configs_reported" in merged:
        unmatched = unmatched[unmatched["n_configs_reported"].fillna(0) > 0]
    numeric = [
        column
        for column in recomputed.columns
        if column not in keys and f"{column}_reported" in merged
    ]
    # Reported boolean flags (e.g. ``separated``) compare as 0/1.
    gaps = {
        column: (
            merged[column].astype(float) - merged[f"{column}_reported"].astype(float)
        )
        .abs()
        .max()
        for column in numeric
    }
    worst = max(gaps.values())
    print(
        f"{name}: {int((merged['_merge'] == 'both').sum())} rows, "
        f"unmatched {len(unmatched)}, max |diff| {worst:.1e}"
    )
    print("   " + ", ".join(f"{column} {gap:.1e}" for column, gap in gaps.items()))
    return max(worst, float(len(unmatched)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--released",
        type=Path,
        required=True,
        help="released_results_tidy.csv from the E-A re-analysis run",
    )
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    released = pd.read_csv(args.released)

    worst = 0.0
    for name, recomputed, reported_path in (
        *effects_tables(),
        *grid_tables(released),
        *baseline_table(),
        *phase5_tables(),
    ):
        worst = max(worst, compare(name, recomputed, reported_path))
    verdict = "MATCH" if worst <= args.tolerance else "MISMATCH"
    print(
        f"\n{verdict}: largest discrepancy {worst:.1e} (tolerance {args.tolerance:g})"
    )
    raise SystemExit(0 if verdict == "MATCH" else 1)


if __name__ == "__main__":
    main()
