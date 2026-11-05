"""Predeclared analyses of the sweep results (see LOG.md, "Sweep Records").

Every value is read from the runs' ``metrics.json`` and ``run_status.json``;
logs are never parsed. ``load_sweep`` checks coverage against the sweep's
expected run set before anything is aggregated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    mannwhitneyu,
    pearsonr,
    spearmanr,
    wilcoxon,
)


def load_sweep(sweep_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Return one row per completed run plus a coverage report.

    Parameters
    ----------
    sweep_dir : pathlib.Path
        Sweep directory containing ``sweep_status.json`` and ``runs/``.

    Returns
    -------
    tuple
        ``(frame, coverage)``; ``frame`` has the candidate columns, ``seed``
        and the train/val/test accuracy and weighted F1. ``coverage`` reports
        expected, observed, missing, duplicate and non-finite runs and whether
        the sweep is complete.
    """

    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    rows, missing, non_finite = [], [], []
    for entry in status["runs"]:
        run_status_path = sweep_dir / entry["run_status"]
        metrics_path = sweep_dir / entry["metrics_path"]
        run_status = (
            json.loads(run_status_path.read_text(encoding="utf-8"))
            if run_status_path.exists()
            else {}
        )
        if run_status.get("status") != "COMPLETED" or not metrics_path.exists():
            missing.append(entry["run_id"])
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "run_id": entry["run_id"],
            "seed": run_status["seed"],
            **run_status["candidate"],
        }
        for split in ("train", "val", "test"):
            row[f"{split}_accuracy"] = metrics[split]["accuracy"]
            row[f"{split}_f1"] = metrics[split]["f1_weighted"]
        for key in ("evaluations", "fit_seconds", "final_loss", "parameters", "model"):
            if key in metrics:
                row[key] = metrics[key]
        if not np.isfinite([row["val_accuracy"], row["test_accuracy"]]).all():
            non_finite.append(entry["run_id"])
            continue
        rows.append(row)
    frame = pd.DataFrame(rows)
    duplicates = int(frame["run_id"].duplicated().sum()) if len(frame) else 0
    coverage = {
        "expected_runs": status["expected_runs"],
        "observed_runs": len(frame),
        "missing": missing,
        "non_finite": non_finite,
        "duplicates": duplicates,
        "complete": len(frame) == status["expected_runs"]
        and not missing
        and not non_finite
        and duplicates == 0,
    }
    return frame, coverage


def _holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""

    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_effect(differences: np.ndarray, rng: np.random.Generator) -> dict:
    """Mean paired difference, bootstrap 95 % CI and two-sided Wilcoxon p."""

    boot = rng.choice(differences, size=(5000, len(differences)), replace=True).mean(
        axis=1
    )
    p_value = (
        1.0 if np.allclose(differences, 0) else float(wilcoxon(differences).pvalue)
    )
    return {
        "n_pairs": int(len(differences)),
        "mean_diff": float(differences.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "wilcoxon_p": p_value,
    }


def mechanism_effects(
    frame: pd.DataFrame, metric: str = "val_accuracy", margin: float = 0.02
) -> pd.DataFrame:
    """E-B: scaling effect at fixed init and init effects at fixed scaling.

    Pairs are formed per dataset on (circuit, optimizer, seed). Holm
    correction is applied across the whole family of comparisons. The
    ``negligible`` column marks effects whose 95 % CI lies inside
    ``+-margin``.
    """

    keys = ["circuit", "optimizer", "seed"]
    rng = np.random.default_rng(0)
    rows = []
    for dataset, data in frame.groupby("data"):
        table = data.pivot_table(index=keys, columns=["scaling", "init"], values=metric)
        for init in ("default", "beta", "zeros"):
            diff = (table[("minmax", init)] - table[("raw", init)]).dropna().to_numpy()
            rows.append(
                {
                    "data": dataset,
                    "effect": "scaling",
                    "comparison": f"minmax - raw | init={init}",
                    **paired_effect(diff, rng),
                }
            )
        for scaling in ("raw", "minmax"):
            for init in ("beta", "zeros"):
                diff = (
                    (table[(scaling, init)] - table[(scaling, "default")])
                    .dropna()
                    .to_numpy()
                )
                rows.append(
                    {
                        "data": dataset,
                        "effect": "init",
                        "comparison": f"{init} - default | scaling={scaling}",
                        **paired_effect(diff, rng),
                    }
                )
    result = pd.DataFrame(rows)
    result["holm_p"] = _holm(result["wilcoxon_p"].tolist())
    result["negligible"] = result["ci95"].map(
        lambda ci: -margin < ci[0] and ci[1] < margin
    )
    return result


def photonic_effects(
    frame: pd.DataFrame, metric: str = "val_accuracy", margin: float = 0.02
) -> pd.DataFrame:
    """E-P (Phase 4, photonic-native): the E-B family plus resource effects.

    Per dataset: scaling (minmax - raw at fixed init) and init (beta and zeros
    - default at fixed scaling), paired on (photons, depth, seed), Holm
    across that family: the E-B rule. Resources: photons (n4 - n2) and
    reuploading depth (d2 - d1) at fixed scaling, paired on the remaining
    factors, Holm within that family.
    """

    rng = np.random.default_rng(0)
    mechanism, resources = [], []
    for dataset, data in frame.groupby("data"):
        table = data.pivot_table(
            index=["photons", "depth", "seed"],
            columns=["scaling", "init"],
            values=metric,
        )
        for init in ("default", "beta", "zeros"):
            diff = (table[("minmax", init)] - table[("raw", init)]).dropna().to_numpy()
            mechanism.append(
                {
                    "data": dataset,
                    "effect": "scaling",
                    "comparison": f"minmax - raw | init={init}",
                    **paired_effect(diff, rng),
                }
            )
        for scaling in ("raw", "minmax"):
            for init in ("beta", "zeros"):
                diff = (
                    (table[(scaling, init)] - table[(scaling, "default")])
                    .dropna()
                    .to_numpy()
                )
                mechanism.append(
                    {
                        "data": dataset,
                        "effect": "init",
                        "comparison": f"{init} - default | scaling={scaling}",
                        **paired_effect(diff, rng),
                    }
                )
        for factor, high, low, others in (
            ("photons", "n4", "n2", ["init", "depth", "seed"]),
            ("depth", "d2", "d1", ["init", "photons", "seed"]),
        ):
            levels = data.pivot_table(
                index=others, columns=["scaling", factor], values=metric
            )
            for scaling in ("raw", "minmax"):
                diff = (
                    (levels[(scaling, high)] - levels[(scaling, low)])
                    .dropna()
                    .to_numpy()
                )
                resources.append(
                    {
                        "data": dataset,
                        "effect": factor,
                        "comparison": f"{high} - {low} | scaling={scaling}",
                        **paired_effect(diff, rng),
                    }
                )
    families = []
    for family in (mechanism, resources):
        table = pd.DataFrame(family)
        table["holm_p"] = _holm(table["wilcoxon_p"].tolist())
        families.append(table)
    result = pd.concat(families, ignore_index=True)
    result["negligible"] = result["ci95"].map(
        lambda ci: -margin < ci[0] and ci[1] < margin
    )
    return result


def cell_means(
    frame: pd.DataFrame, factors: list[str], metric: str = "val_accuracy"
) -> pd.DataFrame:
    """Mean, standard deviation and run count of ``metric`` per factor cell."""

    return (
        frame.groupby(factors)[metric]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )


DATASET_KEYS = {
    "rice": "rice",
    "glass": "glass_identification",
    "cover": "covtype",
    "kdd": "kdd",
}


def baseline_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """E-F: mean and sd over seeds of val/test accuracy per data, scaling and model."""

    return (
        frame.groupby(["data", "scaling", "model"])
        .agg(
            val_mean=("val_accuracy", "mean"),
            val_sd=("val_accuracy", "std"),
            test_mean=("test_accuracy", "mean"),
            test_sd=("test_accuracy", "std"),
            parameters=("parameters", "median"),
            n=("val_accuracy", "count"),
        )
        .reset_index()
    )


def qnn_reference(released: pd.DataFrame) -> pd.DataFrame:
    """Best, 90th percentile and median released noiseless QNN val accuracy.

    Rows match the E-F ``data`` labels (``<dataset>-<preprocessing>``) and
    ``scaling``: the author's uniform and normal arms used raw inputs, the
    beta arm MinMax inputs (see LOG.md, "Paper vs reference code").
    """

    noiseless = released[
        ~released["noisy"] & released["init"].isin(["uniform", "normal", "beta"])
    ].copy()
    noiseless["scaling"] = np.where(noiseless["init"] == "beta", "minmax", "raw")
    rows = []
    for short, dataset in DATASET_KEYS.items():
        for preprocessing in ("pca", "lda"):
            for scaling in ("raw", "minmax"):
                subset = noiseless[
                    (noiseless["dataset"] == dataset)
                    & (noiseless["preprocessing"] == preprocessing)
                    & (noiseless["scaling"] == scaling)
                ]["accuracy"].dropna()
                if len(subset):
                    rows.append(
                        {
                            "data": f"{short}-{preprocessing}",
                            "scaling": scaling,
                            "qnn_best": float(subset.max()),
                            "qnn_p90": float(subset.quantile(0.9)),
                            "qnn_median": float(subset.median()),
                            "qnn_runs": int(len(subset)),
                        }
                    )
    return pd.DataFrame(rows)


GRID_ARMS = {"uniformarm": "uniform", "betaarm": "beta"}
GRID_OPTIMIZERS = {"cobyla": "cobyla", "spsa": "spsa", "nm": "neldermead"}
# Released arms that ran the identical code path (same input scaling, same
# qiskit-default initial point) and so act as an independent replicate.
REPLICATE_ARMS = {"uniform": "normal", "beta": "beta_mu"}


def grid_as_released(
    frame: pd.DataFrame, released: pd.DataFrame, noisy: bool = False
) -> pd.DataFrame:
    """E-C runs in the released-results layout, for the author's tests.

    ``accuracy`` and ``f1`` are the validation-split values, as in the
    released files; the circuit columns come from the released row with the
    same grid index. ``noisy`` labels the rows as FakePerth runs; default
    value is False.
    """

    circuits = released[
        (released["dataset"] == "rice")
        & ~released["noisy"]
        & (released["optimizer"] == "cobyla")
        & (released["init"] == "uniform")
    ][["index", "ansatz", "featuremap", "entanglement", "entanglement_featuremap"]]
    ours = pd.DataFrame(
        {
            "index": frame["grid_index"].astype(int),
            "dataset": "rice",
            "noisy": noisy,
            "optimizer": frame["optimizer"].map(GRID_OPTIMIZERS),
            "preprocessing": "pca",
            "init": frame["arm"].map(GRID_ARMS),
            "accuracy": frame["val_accuracy"],
            "f1": frame["val_f1"],
            "test_accuracy": frame["test_accuracy"],
            "iterations": frame["evaluations"],
        }
    )
    return ours.merge(circuits, on="index", how="left")


def grid_agreement(
    ours: pd.DataFrame, released: pd.DataFrame, noisy: bool = False
) -> pd.DataFrame:
    """E-C: per (arm, optimizer) agreement of our runs with the released runs.

    Our single run per configuration is paired with the released run by grid
    index. The noise floor is the |difference| between the released run and
    the released replicate arm (``REPLICATE_ARMS``) of the same
    configuration: two runs of identical code that differ only by chance.
    ``within_floor`` is the fraction of our |differences| at or below the
    95th percentile of that replicate |difference|; ``floor_mwu_p`` tests
    whether our |differences| are larger than the replicate ones. ``noisy``
    selects the released noisy runs; default value is False.
    """

    rice = released[(released["dataset"] == "rice") & (released["noisy"] == noisy)]
    rows = []
    for (init, optimizer), data in ours.groupby(["init", "optimizer"]):
        reference = rice[(rice["init"] == init) & (rice["optimizer"] == optimizer)]
        replicate = rice[
            (rice["init"] == REPLICATE_ARMS[init]) & (rice["optimizer"] == optimizer)
        ]
        paired = (
            data.set_index("index")["accuracy"]
            .to_frame("ours")
            .join(reference.set_index("index")["accuracy"].rename("released"))
            .join(replicate.set_index("index")["accuracy"].rename("replicate"))
        )
        both = paired[["ours", "released"]].dropna()
        gap = (both["ours"] - both["released"]).abs()
        floor_pairs = paired[["released", "replicate"]].dropna()
        floor_gap = (floor_pairs["released"] - floor_pairs["replicate"]).abs()
        row = {
            "init": init,
            "optimizer": optimizer,
            "n_pairs": int(len(both)),
            "ours_mean": float(both["ours"].mean()),
            "released_mean": float(both["released"].mean()),
            "mean_diff": float((both["ours"] - both["released"]).mean()),
            "median_abs_diff": float(gap.median()),
            "pearson": float(pearsonr(both["ours"], both["released"])[0]),
            "spearman": float(spearmanr(both["ours"], both["released"])[0]),
            "replicate_arm": REPLICATE_ARMS[init],
            "replicate_n": int(len(floor_pairs)),
        }
        if len(floor_pairs) >= 10:
            threshold = float(floor_gap.quantile(0.95))
            row.update(
                {
                    "replicate_pearson": float(
                        pearsonr(floor_pairs["released"], floor_pairs["replicate"])[0]
                    ),
                    "replicate_median_abs_diff": float(floor_gap.median()),
                    "floor_p95": threshold,
                    "within_floor": float((gap <= threshold).mean()),
                    "floor_mwu_p": float(
                        mannwhitneyu(gap, floor_gap, alternative="greater").pvalue
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def noise_effect(
    noisy: pd.DataFrame, noiseless: pd.DataFrame, released: pd.DataFrame
) -> pd.DataFrame:
    """E-D: accuracy change from adding noise, ours next to the author's.

    Both frames are in the ``grid_as_released`` layout. For each (arm,
    optimizer) the noisy run of a configuration is paired with the noiseless
    run of the same configuration, in our runs and in the released runs. The
    noisy budgets are the author's (half the noiseless ones), so the change
    includes the budget cut, as in the paper's comparison.
    """

    rice = released[released["dataset"] == "rice"]
    rows = []
    for (init, optimizer), data in noisy.groupby(["init", "optimizer"]):
        ours_noiseless = noiseless[
            (noiseless["init"] == init) & (noiseless["optimizer"] == optimizer)
        ].set_index("index")["accuracy"]
        reference = rice[(rice["init"] == init) & (rice["optimizer"] == optimizer)]
        paired = pd.DataFrame(
            {
                "ours_noisy": data.set_index("index")["accuracy"],
                "ours_noiseless": ours_noiseless,
                "released_noisy": reference[reference["noisy"]].set_index("index")[
                    "accuracy"
                ],
                "released_noiseless": reference[~reference["noisy"]].set_index("index")[
                    "accuracy"
                ],
            }
        ).dropna()
        ours_change = paired["ours_noisy"] - paired["ours_noiseless"]
        released_change = paired["released_noisy"] - paired["released_noiseless"]
        rows.append(
            {
                "init": init,
                "optimizer": optimizer,
                "n_configs": int(len(paired)),
                "ours_noisy_mean": float(paired["ours_noisy"].mean()),
                "released_noisy_mean": float(paired["released_noisy"].mean()),
                "ours_noise_change": float(ours_change.mean()),
                "released_noise_change": float(released_change.mean()),
                "ours_mean_abs_change": float(ours_change.abs().mean()),
                "released_mean_abs_change": float(released_change.abs().mean()),
            }
        )
    return pd.DataFrame(rows)
