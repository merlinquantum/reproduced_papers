"""Combine Task 3 forecasts from several sweeps on shared test dates and test them.

Joins the ``test_predictions.csv`` of an E6 sweep (``--base``, variants
``e6_<TICKER>``) with extra-model sweeps (``--extra``, variants ``<TICKER>``),
checks that realized values agree date by date, then reports per ticker and
model: QLIKE, MSE, R^2, DirAcc and Diebold-Mariano tests (QLIKE and MSE loss,
two-sided, HLN correction) against ``--reference`` (default ``qsvr_angle``).
Writes ``results/<--output>.csv``.

Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/stats_volatility.py \\
        --base <S5 dir> --extra <S10 dir> --output e6_with_garch_sigma2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR)]

from lib.metrics import diebold_mariano, forecast_metrics, qlike  # noqa: E402


def predictions_by_variant(sweep_dir: Path) -> dict[str, pd.DataFrame]:
    """``test_predictions.csv`` of every completed run, keyed by variant label."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    return {
        Path(run["run_dir"]).parts[0]: pd.read_csv(
            sweep_dir / run["run_dir"] / "test_predictions.csv"
        )
        for run in status["runs"]
        if run["status"] == "COMPLETED"
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base", type=Path, required=True, help="E6 sweep with variants e6_<TICKER>"
    )
    parser.add_argument(
        "--extra",
        type=Path,
        nargs="+",
        required=True,
        help="Sweeps with variants <TICKER>",
    )
    parser.add_argument("--reference", default="qsvr_angle")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    base = {
        label.removeprefix("e6_"): frame
        for label, frame in predictions_by_variant(args.base).items()
        if label.startswith("e6_")
    }
    rows = []
    for ticker, frame in sorted(base.items()):
        merged = frame
        for extra_dir in args.extra:
            extra = predictions_by_variant(extra_dir)[ticker]
            merged = merged.merge(
                extra, on="date", suffixes=("", "_extra"), validate="one_to_one"
            )
            if len(merged) != len(frame) or not np.allclose(
                merged["realized"], merged["realized_extra"], rtol=1e-12, atol=0
            ):
                raise ValueError(
                    f"{ticker}: test dates or realized values differ between sweeps"
                )
            merged = merged.drop(columns="realized_extra")
        realized = merged["realized"].to_numpy()
        previous = np.concatenate([[np.nan], realized[:-1]])
        reference = merged[args.reference].to_numpy()
        for model in [
            column for column in merged.columns if column not in ("date", "realized")
        ]:
            forecast = merged[model].to_numpy()
            metrics = forecast_metrics(realized[1:], forecast[1:], previous[1:])
            row = {"ticker": ticker, "model": model, **metrics}
            if model != args.reference:
                row["dm_qlike_statistic"], row["dm_qlike_p"] = diebold_mariano(
                    qlike(realized, forecast), qlike(realized, reference)
                )
                row["dm_mse_statistic"], row["dm_mse_p"] = diebold_mariano(
                    (realized - forecast) ** 2, (realized - reference) ** 2
                )
            rows.append(row)
    table = pd.DataFrame(rows)
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    table.to_csv(results_dir / f"{args.output}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
