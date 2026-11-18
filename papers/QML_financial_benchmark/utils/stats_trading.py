"""E5 seed statistics: join stage 1 (seed 0, full grid) with stage 2 (seeds 1-4, fixed selections).

For every regime finished in the stage-1 sweep, each seed-0 selection key
(``auc:qlstm``, ``auc:lstm``, ``auc:lstm_parameter_matched`` and the
``tau_sharpe:`` counterparts) names one candidate. Seed 0's test metrics come
from stage 1; seeds 1-4 come from the stage-2 runs' ``candidate:<name>``
entries (no re-selection). Floors (buy-and-hold, always-flat, logistic
regression) are taken from every run. Reports mean, std and per-seed values,
plus seed-paired QLSTM - LSTM Sharpe and AUC differences with a one-sample
t-test. Writes ``results/<--output>_per_seed.csv`` and ``results/<--output>.csv``.

Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/stats_trading.py \\
        --stage1 <S4 dir> --stage2 <S9 dir> [<more stage-2 dirs>] --output e5_trading
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
SELECTION_KEYS = [
    f"{rule}:{family}"
    for rule in ("auc", "tau_sharpe")
    for family in ("qlstm", "lstm", "lstm_parameter_matched")
]
FLOORS = ["buy_and_hold", "always_flat", "logistic_regression"]
METRICS = ["sharpe", "arc", "asd", "sortino", "max_drawdown", "auc"]


def completed_metrics(sweep_dir: Path) -> dict[tuple[str, int], dict]:
    """``metrics.json`` of completed runs keyed by (variant, seed)."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    return {
        (Path(run["run_dir"]).parts[0], run["seed"]): json.loads(
            (sweep_dir / run["metrics_path"]).read_text()
        )
        for run in status["runs"]
        if run["status"] == "COMPLETED"
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    stage1 = completed_metrics(args.stage1)
    stage2: dict[tuple[str, int], dict] = {}
    for sweep_dir in args.stage2:
        for key, metrics in completed_metrics(sweep_dir).items():
            if key in stage2:
                raise ValueError(f"Duplicate stage-2 run for {key}")
            stage2[key] = metrics

    rows = []
    for (regime, seed0), seed0_metrics in sorted(stage1.items()):
        if seed0 != 0:
            raise ValueError(
                f"Stage 1 must contain seed 0 only, found {seed0} for {regime}"
            )
        selections = {
            key: seed0_metrics["selection"][key]["candidate"] for key in SELECTION_KEYS
        }
        seeds = {
            0: seed0_metrics,
            **{seed: m for (variant, seed), m in stage2.items() if variant == regime},
        }
        for seed, metrics in sorted(seeds.items()):
            for key in SELECTION_KEYS:
                source = (
                    metrics["test"][key]
                    if seed == 0
                    else metrics["test"][f"candidate:{selections[key]}"]
                )
                if source.get("candidate", selections[key]) != selections[key]:
                    raise ValueError(f"{regime} seed {seed} {key}: candidate mismatch")
                rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "model": key,
                        "candidate": selections[key],
                        "parameters": source["parameters"],
                        **{metric: source[metric] for metric in METRICS},
                        "share_long": source.get("share_long"),
                        "share_short": source.get("share_short"),
                    }
                )
            for floor in FLOORS:
                source = metrics["test"][floor]
                rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "model": floor,
                        "candidate": floor,
                        "parameters": source["parameters"],
                        **{metric: source[metric] for metric in METRICS},
                        "share_long": source.get("share_long"),
                        "share_short": source.get("share_short"),
                    }
                )
    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby(["regime", "model"], sort=False)
        .agg(
            candidate=("candidate", "first"),
            parameters=("parameters", "first"),
            n_seeds=("seed", "nunique"),
            **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
            **{
                f"{metric}_std": (metric, lambda values: values.std(ddof=1))
                for metric in METRICS
            },
        )
        .reset_index()
    )
    comparisons = []
    for regime, frame in per_seed.groupby("regime"):
        for rule in ("auc", "tau_sharpe"):
            for baseline in ("lstm", "lstm_parameter_matched"):
                quantum = frame[frame.model == f"{rule}:qlstm"].set_index("seed")
                classical = frame[frame.model == f"{rule}:{baseline}"].set_index("seed")
                for metric in ("sharpe", "auc"):
                    differences = (quantum[metric] - classical[metric]).dropna()
                    result = (
                        stats.ttest_1samp(differences, 0.0)
                        if len(differences) > 1
                        else None
                    )
                    comparisons.append(
                        {
                            "regime": regime,
                            "model": f"{rule}:qlstm - {rule}:{baseline}",
                            "metric": metric,
                            "n_seeds": len(differences),
                            "difference_mean": differences.mean(),
                            "difference_std": differences.std(ddof=1)
                            if len(differences) > 1
                            else np.nan,
                            "positive_seeds": int((differences > 0).sum()),
                            "t_p": float(result.pvalue)
                            if result is not None
                            else np.nan,
                        }
                    )
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    per_seed.to_csv(results_dir / f"{args.output}_per_seed.csv", index=False)
    summary.to_csv(results_dir / f"{args.output}.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        results_dir / f"{args.output}_comparisons.csv", index=False
    )
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(
            summary[
                [
                    "regime",
                    "model",
                    "candidate",
                    "parameters",
                    "n_seeds",
                    "sharpe_mean",
                    "sharpe_std",
                    "arc_mean",
                    "auc_mean",
                    "auc_std",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )
        print(pd.DataFrame(comparisons).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
