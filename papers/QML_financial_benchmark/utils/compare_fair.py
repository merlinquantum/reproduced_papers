"""Phase 5 comparison table, generated from structured sweep artifacts.

Reads every sweep in ``utils/verify_sweeps.py::SWEEP_PLANS`` (completed runs
only, full-precision ``metrics.json``), the per-sweep ``statistics.csv`` files
and the Task 2/3 tables in ``results/``, and writes:

- ``results/phase5_comparison.csv``: one row per (claim setting, system) with
  role, provenance, scope, successful/expected runs, metric mean and std,
  matching basis, parameter range, mean wall-clock per training, and the
  commensurability status of its comparison with the quantum model;
- ``results/phase5_differences.csv``: quantum - control differences with CIs
  and p-values (the inputs of the claim outcomes and the figure);
- ``results/phase5_timing.csv``: wall-clock per training by model family.

Paper-reported values are rows with provenance ``paper``; everything else is
``local``. Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/compare_fair.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR)]

from utils.verify_sweeps import SWEEP_PLANS, find_sweep, variant_of  # noqa: E402

RESULTS = PROJECT_DIR / "results"

# Task 1 settings: (claim, label, sweep prefix, variant, scope, paper QNN AUC, paper ANN AUC)
T1_SETTINGS = [
    (
        "C2",
        "KCHOL 3-D lagged",
        "e1_kchol_2",
        "main",
        "paper-accurate",
        0.7968,
        0.7475,
        "e1_kchol",
    ),
    (
        "C2",
        "KCHOL 3-D same-day (diagnostic)",
        "e1_kchol_sameday",
        "main",
        "diagnostic",
        0.7968,
        0.7475,
        "e1_kchol_sameday",
    ),
    (
        "C3",
        "GARAN 3-D lagged",
        "e1_turkish_rest",
        "GARAN",
        "paper-accurate",
        0.7093,
        0.8102,
        "e1_turkish_rest",
    ),
    (
        "C3",
        "TCELL 3-D lagged",
        "e1_turkish_rest",
        "TCELL",
        "paper-accurate",
        0.7430,
        0.7597,
        "e1_turkish_rest",
    ),
    (
        "C3",
        "TUPRS 3-D lagged",
        "e1_turkish_rest",
        "TUPRS",
        "paper-accurate",
        0.9039,
        0.9042,
        "e1_turkish_rest",
    ),
    (
        "C3",
        "ULKER 3-D lagged",
        "e1_turkish_rest",
        "ULKER",
        "paper-accurate",
        0.8518,
        0.8760,
        "e1_turkish_rest",
    ),
    ("C3", "S&P 500 7-D", "e2_spx", "main", "paper-accurate", 0.7498, 0.7657, "e2_spx"),
    (
        "C1",
        "AAPL 64-D Table V timing",
        "e3_tablev_aapl",
        "main",
        "paper-accurate",
        0.635,
        0.597,
        "e3_tablev_aapl",
    ),
    (
        "C1",
        "AAPL 64-D strict timing",
        "e3_aapl",
        "main",
        "paper-accurate (strict timing)",
        0.635,
        0.597,
        "e3_aapl",
    ),
    (
        "C3",
        "BA 64-D strict timing",
        "e3_us_rest_seed",
        "BA",
        "paper-accurate (strict timing), 3 seeds",
        0.618,
        0.738,
        "e3_us_rest",
    ),
    (
        "C3",
        "GILD 64-D strict timing",
        "e3_us_rest_seed",
        "GILD",
        "paper-accurate (strict timing), 3 seeds",
        0.590,
        0.620,
        "e3_us_rest",
    ),
    (
        "C3",
        "DVN 64-D strict timing",
        "e3_us_rest_seed",
        "DVN",
        "paper-accurate (strict timing), 3 seeds",
        0.608,
        0.675,
        "e3_us_rest",
    ),
    (
        "C3",
        "LNC 64-D strict timing",
        "e3_us_rest_seed",
        "LNC",
        "paper-accurate (strict timing), 3 seeds",
        0.663,
        0.694,
        "e3_us_rest",
    ),
]
T1_MERLIN = {"S&P 500 7-D": "e2_merlin", "AAPL 64-D Table V timing": "e3_tablev_merlin"}
T1_SYSTEMS = [
    ("leak_free:qnn", "quantum model (paper method)"),
    ("leak_free:ann", "paper baseline (ANN grid, reproduced)"),
    ("leak_free:ann_parameter_matched", "fair control: parameter-matched ANN"),
    ("logistic_regression", "fair control: linear floor"),
    ("majority_class", "floor"),
]
T2_PAPER = {
    "F1": (0.3865, -0.0477),
    "F2": (1.2585, 0.4966),
    "F3": (0.7659, 0.8947),
    "F4": (-0.6936, 1.5228),
}
T3_PAPER_QLIKE = {
    "GSPC": {
        "qsvr_angle": -8.2937,
        "svr_rbf": -8.3277,
        "svr_linear": -7.8748,
        "garch_sigma2": -8.2488,
    },
    "AAPL": {
        "qsvr_angle": -7.3888,
        "svr_rbf": -7.4101,
        "svr_linear": -7.2522,
        "garch_sigma2": -7.2831,
    },
    "GARAN": {
        "qsvr_angle": -6.2121,
        "svr_rbf": -6.2196,
        "svr_linear": -6.0486,
        "garch_sigma2": -6.1398,
    },
    "KCHOL": {
        "qsvr_angle": -6.4352,
        "svr_rbf": -6.4247,
        "svr_linear": -6.2930,
        "garch_sigma2": -6.3593,
    },
}
TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def completed_runs(prefix: str, variant: str) -> tuple[list[Path], int]:
    """Run directories of one variant across every sweep whose name starts with ``prefix``, and the planned count."""
    run_dirs, expected = [], 0
    for name, (_, variants, seeds, _) in SWEEP_PLANS.items():
        if not name.startswith(prefix) or variant not in variants:
            continue
        sweep = find_sweep(name)
        status = json.loads((sweep / "sweep_status.json").read_text())
        run_dirs += [
            sweep / r["run_dir"]
            for r in status["runs"]
            if r["status"] == "COMPLETED" and variant_of(r) == variant
        ]
        expected += len(seeds)
    return run_dirs, expected


def family_of(candidate: str) -> str:
    return candidate.split("_")[0]


def training_seconds(run_dir: Path, event: str) -> dict[str, list[float]]:
    """Seconds between consecutive ``event`` lines, attributed to the candidate (or model) named on the later line."""
    durations: dict[str, list[float]] = {}
    previous = None
    for line in (run_dir / "run.log").read_text().splitlines():
        match = TIMESTAMP.match(line)
        if not match:
            continue
        stamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        if "TRAINING_STARTED" in line:
            previous = stamp
        elif event in line and previous is not None:
            name = re.search(r"(?:candidate|model)=([^ |]+)", line).group(1)
            durations.setdefault(
                family_of(name) if event != "TRAIN_REFIT_COMPLETED" else name, []
            ).append((stamp - previous).total_seconds())
            previous = stamp
    return durations


def task1_rows(rows: list, differences: list, timing: list) -> None:
    for (
        claim,
        label,
        prefix,
        variant,
        scope,
        paper_qnn,
        paper_ann,
        stats_name,
    ) in T1_SETTINGS:
        run_dirs, expected = completed_runs(prefix, variant)
        metrics = [
            json.loads((d / "metrics.json").read_text())["test"] for d in run_dirs
        ]
        for key, role in T1_SYSTEMS:
            aucs = np.array([m[key]["auc"] for m in metrics], dtype=float)
            if not np.all(np.isfinite(aucs)):
                raise ValueError(f"Non-finite AUC for {label} {key}")
            parameters = [m[key]["parameters"] for m in metrics]
            rows.append(
                {
                    "claim": claim,
                    "setting": label,
                    "system": key,
                    "role": role,
                    "provenance": "local",
                    "scope": scope,
                    "runs": f"{len(aucs)}/{expected}",
                    "metric": "test AUC (max)",
                    "mean": aucs.mean(),
                    "std": aucs.std(ddof=1) if len(aucs) > 1 else np.nan,
                    "matching_basis": "trainable parameters (±10%)"
                    if "matched" in key
                    else "same data, folds, optimizer, selection rule",
                    "parameters": f"{min(parameters)}–{max(parameters)}",
                    "status": "COMMENSURABLE"
                    if "diagnostic" not in scope
                    else "PARTIALLY_COMMENSURABLE (diagnostic features)",
                }
            )
        for key, value in (("paper QNN", paper_qnn), ("paper ANN", paper_ann)):
            rows.append(
                {
                    "claim": claim,
                    "setting": label,
                    "system": key,
                    "role": "paper-reported",
                    "provenance": "paper",
                    "scope": "single run",
                    "runs": "1/1",
                    "metric": "test AUC (max)",
                    "mean": value,
                    "std": np.nan,
                    "matching_basis": "paper",
                    "parameters": "not reported" if key == "paper QNN" else "",
                    "status": "PARTIALLY_COMMENSURABLE (different data snapshot; single run)",
                }
            )
        for stats_file in [stats_name] + (
            [T1_MERLIN[label]] if label in T1_MERLIN else []
        ):
            for stats_path in [RESULTS / f"{stats_file}_statistics.csv"]:
                stats = pd.read_csv(stats_path)
                if "variant" in stats:
                    stats = (
                        stats[stats["variant"] == variant]
                        if variant != "main"
                        else stats
                    )
                for comparison in (
                    "leak_free:qnn - leak_free:ann",
                    "leak_free:qnn - leak_free:ann_parameter_matched",
                    "leak_free:qnn - logistic_regression",
                ):
                    seed_level = stats[
                        (stats["comparison"] == comparison)
                        & stats["scope"].str.startswith("seed-level")
                    ]
                    ensemble = stats[
                        (stats["comparison"] == comparison)
                        & (stats["scope"] == "seed-ensemble")
                    ]
                    if seed_level.empty:
                        continue
                    differences.append(
                        {
                            "claim": claim,
                            "setting": label,
                            "model": "photonic QNN (MerLin adaptation)"
                            if "merlin" in stats_path.name
                            else "QNN",
                            "comparison": comparison,
                            "difference": float(seed_level["difference"].iloc[0]),
                            "ci_low": float(seed_level["ci_low"].iloc[0]),
                            "ci_high": float(seed_level["ci_high"].iloc[0]),
                            "p_value": float(seed_level["t_p"].iloc[0]),
                            "ensemble_p": float(ensemble["bootstrap_p"].iloc[0])
                            if not ensemble.empty
                            else np.nan,
                            "source": stats_path.name,
                        }
                    )
        if label in T1_MERLIN:
            merlin_dirs, merlin_expected = completed_runs(T1_MERLIN[label], "main")
            merlin = np.array(
                [
                    json.loads((d / "metrics.json").read_text())["test"][
                        "leak_free:qnn"
                    ]["auc"]
                    for d in merlin_dirs
                ]
            )
            parameters = [
                json.loads((d / "metrics.json").read_text())["test"]["leak_free:qnn"][
                    "parameters"
                ]
                for d in merlin_dirs
            ]
            rows.append(
                {
                    "claim": claim,
                    "setting": label,
                    "system": "photonic QNN (MerLin)",
                    "role": "MerLin adaptation",
                    "provenance": "local",
                    "scope": "paper-accurate protocol, adaptation",
                    "runs": f"{len(merlin)}/{merlin_expected}",
                    "metric": "test AUC (max)",
                    "mean": merlin.mean(),
                    "std": merlin.std(ddof=1),
                    "matching_basis": "same data, folds, optimizer, selection; paired ANN grid",
                    "parameters": f"{min(parameters)}–{max(parameters)}",
                    "status": "COMMENSURABLE",
                }
            )
            run_dirs = run_dirs + merlin_dirs
        for run_dir in run_dirs:
            for family, seconds in training_seconds(
                run_dir, "CANDIDATE_FOLD_EVALUATED"
            ).items():
                timing += [
                    {"task": "T1", "setting": label, "family": family, "seconds": s}
                    for s in seconds
                ]


def task2_rows(rows: list, differences: list, timing: list) -> None:
    summary = pd.read_csv(RESULTS / "e5_trading.csv")
    comparisons = pd.read_csv(RESULTS / "e5_trading_comparisons.csv")
    roles = {
        "auc:qlstm": "quantum model (paper method)",
        "auc:lstm": "paper baseline (LSTM grid, reproduced)",
        "auc:lstm_parameter_matched": "fair control: parameter-matched LSTM",
        "buy_and_hold": "fair control: passive strategy",
        "always_flat": "floor",
        "logistic_regression": "fair control: linear threshold strategy",
    }
    for _, row in summary[summary["model"].isin(roles)].iterrows():
        rows.append(
            {
                "claim": "C5",
                "setting": f"S&P 500 regime {row['regime']}",
                "system": row["model"],
                "role": roles[row["model"]],
                "provenance": "local",
                "scope": "partial (seed-0 selection, 5 seeds)",
                "runs": f"{int(row['n_seeds'])}/5",
                "metric": "test Sharpe (max)",
                "mean": row["sharpe_mean"],
                "std": row["sharpe_std"],
                "matching_basis": "parameters (±25%)"
                if "matched" in row["model"]
                else "same split, loss, early stopping, τ grid, fee",
                "parameters": str(row["parameters"]),
                "status": "COMMENSURABLE",
            }
        )
    for regime, (paper_qlstm, paper_lstm) in T2_PAPER.items():
        for key, value in (("paper QLSTM", paper_qlstm), ("paper LSTM", paper_lstm)):
            rows.append(
                {
                    "claim": "C5",
                    "setting": f"S&P 500 regime {regime}",
                    "system": key,
                    "role": "paper-reported",
                    "provenance": "paper",
                    "scope": "single run",
                    "runs": "1/1",
                    "metric": "test Sharpe (max)",
                    "mean": value,
                    "std": np.nan,
                    "matching_basis": "paper",
                    "parameters": "",
                    "status": "PARTIALLY_COMMENSURABLE (single run inside the random-score null band, G3)",
                }
            )
    for _, row in comparisons[comparisons["model"].str.startswith("auc:")].iterrows():
        differences.append(
            {
                "claim": "C5" if row["metric"] == "sharpe" else "C6",
                "setting": f"S&P 500 regime {row['regime']}",
                "model": "QLSTM",
                "comparison": f"{row['model']} ({row['metric']})",
                "difference": row["difference_mean"],
                "ci_low": row["difference_mean"]
                - 2.776 * row["difference_std"] / np.sqrt(row["n_seeds"]),
                "ci_high": row["difference_mean"]
                + 2.776 * row["difference_std"] / np.sqrt(row["n_seeds"]),
                "p_value": row["t_p"],
                "ensemble_p": np.nan,
                "source": "e5_trading_comparisons.csv",
            }
        )
    for prefix in ("e5_stage1", "e5_stage2_F1F3", "e5_stage2_F4"):
        for name, (_, variants, _, _) in SWEEP_PLANS.items():
            if name != prefix:
                continue
            for variant in variants:
                for run_dir in completed_runs(prefix, variant)[0]:
                    for family, seconds in training_seconds(
                        run_dir, "CANDIDATE_EVALUATED"
                    ).items():
                        timing += [
                            {
                                "task": "T2",
                                "setting": variant,
                                "family": family,
                                "seconds": s,
                            }
                            for s in seconds
                        ]


def task3_rows(rows: list, differences: list, timing: list) -> None:
    table = pd.read_csv(RESULTS / "e6_with_garch_sigma2.csv")
    photonic = pd.read_csv(RESULTS / "e6_merlin_vs_qubit.csv")
    table = pd.concat(
        [table, photonic[photonic["model"] == "photonic_kernel"]], ignore_index=True
    )
    roles = {
        "qsvr_angle": "quantum model (paper method)",
        "svr_rbf": "paper baseline (RBF, γ scale/auto)",
        "svr_linear": "paper baseline",
        "svr_poly": "paper baseline",
        "garch_sigma2": "paper baseline (GARCH σ² only)",
        "svr_rbf_wide": "fair control: RBF, wide γ",
        "garch": "fair control: GARCH completed with known returns",
        "har": "fair control: HAR-RV",
        "persistence": "floor",
        "photonic_kernel": "MerLin adaptation",
    }
    for _, row in table[table["model"].isin(roles)].iterrows():
        rows.append(
            {
                "claim": "C7/C8/C10",
                "setting": f"RV {row['ticker']}",
                "system": row["model"],
                "role": roles[row["model"]],
                "provenance": "local",
                "scope": "preliminary (single Optuna seed)"
                if "svr" in row["model"] or "kernel" in row["model"]
                else "deterministic",
                "runs": "1/1",
                "metric": "QLIKE (min)",
                "mean": row["qlike"],
                "std": np.nan,
                "matching_basis": "same features, refits, 50 Optuna trials per refit"
                if "svr" in row["model"] or "kernel" in row["model"]
                else "same data and refits",
                "parameters": "",
                "status": "COMMENSURABLE",
            }
        )
        if row["model"] != "qsvr_angle":
            differences.append(
                {
                    "claim": "C7/C8/C10",
                    "setting": f"RV {row['ticker']}",
                    "model": "photonic kernel"
                    if row["model"] == "photonic_kernel"
                    else "QSVR angle",
                    "comparison": f"QLIKE(qsvr_angle) - QLIKE({row['model']}); negative = QSVR better",
                    "difference": table[
                        (table["ticker"] == row["ticker"])
                        & (table["model"] == "qsvr_angle")
                    ]["qlike"].iloc[0]
                    - row["qlike"],
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": row["dm_qlike_p"],
                    "ensemble_p": np.nan,
                    "source": "e6_with_garch_sigma2.csv / e6_merlin_vs_qubit.csv",
                }
            )
    for ticker, models in T3_PAPER_QLIKE.items():
        for model, value in models.items():
            rows.append(
                {
                    "claim": "C7/C8/C10",
                    "setting": f"RV {ticker}",
                    "system": f"paper {model}",
                    "role": "paper-reported",
                    "provenance": "paper",
                    "scope": "single run",
                    "runs": "1/1",
                    "metric": "QLIKE (min)",
                    "mean": value,
                    "std": np.nan,
                    "matching_basis": "paper",
                    "parameters": "",
                    "status": "INCOMPARABLE in level (unstated test window, R15); ranking comparable",
                }
            )
    for prefix in ("e6_e7_seed0", "e6_merlin"):
        _, variants, _, _ = SWEEP_PLANS[prefix]
        for variant in variants:
            if variant.startswith("e7"):
                continue
            for run_dir in completed_runs(prefix, variant)[0]:
                for model, seconds in training_seconds(
                    run_dir, "TRAIN_REFIT_COMPLETED"
                ).items():
                    timing += [
                        {
                            "task": "T3",
                            "setting": variant,
                            "family": model,
                            "seconds": s,
                        }
                        for s in seconds
                    ]


def main() -> int:
    rows, differences, timing = [], [], []
    task1_rows(rows, differences, timing)
    task2_rows(rows, differences, timing)
    task3_rows(rows, differences, timing)
    timing_table = (
        pd.DataFrame(timing)
        .groupby(["task", "family"])["seconds"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "phase5_comparison.csv", index=False)
    pd.DataFrame(differences).to_csv(RESULTS / "phase5_differences.csv", index=False)
    timing_table.to_csv(RESULTS / "phase5_timing.csv", index=False)
    with pd.option_context(
        "display.width", 250, "display.max_columns", 20, "display.max_colwidth", 40
    ):
        print(
            comparison[
                [
                    "claim",
                    "setting",
                    "system",
                    "provenance",
                    "runs",
                    "mean",
                    "std",
                    "parameters",
                    "status",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )
        print(pd.DataFrame(differences).round(4).to_string(index=False))
        print(timing_table.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
