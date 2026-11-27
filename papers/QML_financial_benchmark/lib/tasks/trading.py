"""Task 2: QLSTM vs LSTM long/short trading on S&P 500 regimes (§III-D, Table XII).

One run covers one regime, one split mode and one seed. Every QLSTM and LSTM
candidate is trained on the train slice with early stopping on the early-stop
slice. Per candidate, the thresholds ``(tau_long, tau_short)`` are calibrated
on the threshold-calibration slice by maximizing Sharpe (§III-D5). Selection
rules, both predeclared:

- ``auc`` (primary, R11): highest AUC on the model-select slice;
- ``tau_sharpe``: highest Sharpe on the threshold-calibration slice, the
  alternative sentence of §III-D5.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from .. import data
from ..metrics import (
    positions_from_probabilities,
    safe_auc,
    strategy_returns,
    trading_metrics,
)
from ..models import LSTMClassifier, QLSTMClassifier, count_parameters
from ..training import fit_classifier, predict_proba

logger = logging.getLogger(__name__)


def threshold_grid(bounds: list[float]) -> np.ndarray:
    """Inclusive grid ``[start, stop]`` with ``step``, rounded to avoid float drift."""
    start, stop, step = bounds
    return np.round(np.arange(start, stop + step / 2, step), 10)


def calibrate_thresholds(
    probabilities: np.ndarray, next_returns: np.ndarray, backtest_cfg: dict
) -> dict:
    """Sharpe-maximizing ``(tau_long, tau_short)`` on one slice.

    Ties go to the pair with fewer position changes, then to the first pair
    in grid order (``tau_long`` ascending, then ``tau_short`` ascending).
    """
    best = None
    fee = float(backtest_cfg["fee"])
    for tau_long in threshold_grid(backtest_cfg["tau_long_grid"]):
        for tau_short in threshold_grid(backtest_cfg["tau_short_grid"]):
            positions = positions_from_probabilities(probabilities, tau_long, tau_short)
            metrics = trading_metrics(
                strategy_returns(positions, next_returns, fee), positions
            )
            key = (metrics["sharpe"], -metrics["n_position_changes"])
            if best is None or key > best[0]:
                best = (
                    key,
                    {
                        "tau_long": float(tau_long),
                        "tau_short": float(tau_short),
                        **metrics,
                    },
                )
    return best[1]


def backtest(
    probabilities: np.ndarray, next_returns: np.ndarray, thresholds: dict, fee: float
) -> dict:
    """Trading metrics of the threshold rule on one slice."""
    positions = positions_from_probabilities(
        probabilities, thresholds["tau_long"], thresholds["tau_short"]
    )
    return trading_metrics(strategy_returns(positions, next_returns, fee), positions)


def trading_candidates(models_cfg: dict) -> list[dict]:
    """QLSTM and LSTM candidates in grid order.

    Each family takes either a grid (``n_qubits``/``hidden_size`` x ``layers``)
    or an explicit ``configs`` list of ``[size, layers]`` pairs, used by E5
    stage 2 to retrain exactly the seed-0 selections.
    """

    def family(name: str, size_key: str, prefix: str) -> list[dict]:
        family_cfg = models_cfg[name]
        pairs = family_cfg.get("configs") or [
            [size, layers]
            for size in family_cfg[size_key]
            for layers in family_cfg["layers"]
        ]
        return [
            {
                "model": name,
                "size": size,
                "layers": layers,
                "name": f"{name}_{prefix}{size}_L{layers}",
            }
            for size, layers in pairs
        ]

    return family("qlstm", "n_qubits", "q") + family("lstm", "hidden_size", "h")


def run_trading(
    cfg: dict, section: dict, run_dir: Path, status: dict, report_progress
) -> dict:
    """Train both recurrent families on one regime, calibrate, select and backtest.

    Artifacts written to ``run_dir``: ``candidates.csv`` (one row per
    candidate) and ``test_predictions.csv`` (test-slice probabilities,
    positions inputs and next-day returns).

    Returns
    -------
    dict
        Metrics for ``metrics.json``.
    """
    dataset_cfg, models_cfg, training_cfg = (
        section["dataset"],
        section["models"],
        section["training"],
    )
    backtest_cfg, selection_cfg = section["backtest"], section["selection"]
    device, dtype, seed = (
        torch.device(cfg["device"]),
        cfg["torch_dtype"],
        int(cfg["seed"]),
    )
    fee = float(backtest_cfg["fee"])

    frame = data.build_trading_frame(dataset_cfg, cfg["data_root"])
    windows, labels, next_returns, dates = data.trading_windows(
        frame, int(dataset_cfg["window"])
    )
    split = data.trading_split(dates, dataset_cfg)
    scaler = data.MinMaxScaler().fit(
        windows[split.train].reshape(-1, windows.shape[-1])
    )
    inputs = torch.as_tensor(scaler.transform(windows), dtype=dtype, device=device)
    label_tensor = torch.as_tensor(labels, dtype=dtype, device=device)
    slices = {
        name: getattr(split, name)
        for name in (
            "train",
            "early_stop",
            "model_select",
            "threshold_calibration",
            "test",
        )
    }
    status["dataset"] = {
        "name": f"yahoo_daily_prices:{dataset_cfg['ticker']}:regime_{split.regime}",
        "splits": {
            name: [
                str(dates[rows[0]].date()),
                str(dates[rows[-1]].date()),
                int(len(rows)),
            ]
            for name, rows in slices.items()
        },
        "preprocessing": f"min-max fitted on train windows; window={dataset_cfg['window']}; split_mode={dataset_cfg['split_mode']}",
        "subset": "full",
    }
    logger.info(
        "DATASET_READY | regime=%s | split_mode=%s | windows=%d | %s",
        split.regime,
        dataset_cfg["split_mode"],
        len(windows),
        " | ".join(
            f"{name}={dates[rows[0]].date()}..{dates[rows[-1]].date()} ({len(rows)})"
            for name, rows in slices.items()
        ),
    )

    candidates = trading_candidates(models_cfg)
    n_features = windows.shape[-1]
    logger.info(
        "TRAINING_STARTED | candidates=%d | max_epochs=%d | patience=%d | optimizer=adam | lr=%g | loss=weighted_bce",
        len(candidates),
        training_cfg["max_epochs"],
        training_cfg["patience"],
        training_cfg["lr"],
    )
    rows, slice_probabilities = [], {}
    for order, candidate in enumerate(candidates):
        torch.manual_seed(seed * 100_000 + order)
        if candidate["model"] == "qlstm":
            model = QLSTMClassifier(
                n_features,
                candidate["size"],
                candidate["layers"],
                models_cfg["qlstm"]["topology"],
            )
        else:
            model = LSTMClassifier(n_features, candidate["size"], candidate["layers"])
        model = model.to(device=device, dtype=dtype)
        fit = fit_classifier(
            model,
            inputs[split.train],
            label_tensor[split.train],
            inputs[split.early_stop],
            label_tensor[split.early_stop],
            training_cfg,
            seed=seed * 100_000 + order,
            context=f"candidate={candidate['name']}",
        )
        probabilities = {
            name: predict_proba(model, inputs[slices[name]])
            for name in ("model_select", "threshold_calibration", "test")
        }
        slice_probabilities[candidate["name"]] = probabilities
        thresholds = calibrate_thresholds(
            probabilities["threshold_calibration"],
            next_returns[split.threshold_calibration],
            backtest_cfg,
        )
        rows.append(
            {
                "name": candidate["name"],
                "order": order,
                "model": candidate["model"],
                "size": candidate["size"],
                "layers": candidate["layers"],
                "parameters": count_parameters(model),
                "best_epoch": fit["best_epoch"],
                "epochs_run": fit["epochs_run"],
                "early_stop_auc": fit["best_early_stop_auc"],
                "model_select_auc": safe_auc(
                    labels[split.model_select], probabilities["model_select"]
                ),
                "tau_long": thresholds["tau_long"],
                "tau_short": thresholds["tau_short"],
                "tau_sharpe": thresholds["sharpe"],
            }
        )
        logger.info(
            "CANDIDATE_EVALUATED | candidate=%s | parameters=%d | model_select_auc=%.6f | tau_sharpe=%.6f | tau=(%.2f, %.2f)",
            candidate["name"],
            rows[-1]["parameters"],
            rows[-1]["model_select_auc"],
            thresholds["sharpe"],
            thresholds["tau_long"],
            thresholds["tau_short"],
        )
        report_progress(len(rows), len(candidates), "candidates")
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "candidates.csv", index=False)
    logger.info("ARTIFACT_WRITTEN | path=candidates.csv | rows=%d", len(table))

    tie_tolerance = float(selection_cfg["tie_tolerance"])
    test_rows = split.test
    test_labels, test_returns = labels[test_rows], next_returns[test_rows]
    results: dict = {
        "regime": split.regime,
        "split_mode": dataset_cfg["split_mode"],
        "selection": {},
        "test": {},
    }
    predictions = {
        "date": dates[test_rows].strftime("%Y-%m-%d"),
        "label": test_labels,
        "next_return": test_returns,
    }

    def report(
        key: str, probabilities: np.ndarray, thresholds: dict, extra: dict
    ) -> None:
        logger.info("EVALUATION_STARTED | split=test | model=%s", key)
        metrics = {
            "auc": safe_auc(test_labels, probabilities),
            **backtest(probabilities, test_returns, thresholds, fee),
        }
        results["test"][key] = {
            **metrics,
            "tau_long": thresholds["tau_long"],
            "tau_short": thresholds["tau_short"],
            **extra,
        }
        predictions[key] = probabilities
        logger.info(
            "EVALUATION_COMPLETED | split=test | model=%s | auc=%.6f | arc=%.6f | asd=%.6f | sharpe=%.6f | sortino=%.6f | max_drawdown=%.6f",
            key,
            metrics["auc"],
            metrics["arc"],
            metrics["asd"],
            metrics["sharpe"],
            metrics["sortino"],
            metrics["max_drawdown"],
        )

    def pick(family_rows: pd.DataFrame, column: str) -> pd.Series:
        tied = family_rows[
            family_rows[column] >= family_rows[column].max() - tie_tolerance
        ]
        return tied.sort_values(["parameters", "order"]).iloc[0]

    for rule, column in (("auc", "model_select_auc"), ("tau_sharpe", "tau_sharpe")):
        for family in ("qlstm", "lstm"):
            chosen = pick(table[table["model"] == family], column)
            key = f"{rule}:{family}"
            results["selection"][key] = {
                "candidate": chosen["name"],
                column: float(chosen[column]),
                "parameters": int(chosen["parameters"]),
            }
            logger.info(
                "MODEL_SELECTED | rule=%s | family=%s | candidate=%s | %s=%.6f | parameters=%d",
                rule,
                family,
                chosen["name"],
                column,
                chosen[column],
                chosen["parameters"],
            )
            report(
                key,
                slice_probabilities[chosen["name"]]["test"],
                chosen[["tau_long", "tau_short"]].to_dict(),
                {"candidate": chosen["name"], "parameters": int(chosen["parameters"])},
            )

        # Fair baseline: the LSTM whose size is closest to the selected QLSTM, if within tolerance.
        qlstm_parameters = results["selection"][f"{rule}:qlstm"]["parameters"]
        lstm_rows = table[table["model"] == "lstm"].copy()
        lstm_rows["gap"] = (lstm_rows["parameters"] - qlstm_parameters).abs()
        closest = lstm_rows.sort_values(["gap", "order"]).iloc[0]
        relative_gap = float(closest["gap"]) / qlstm_parameters
        matched_key = f"{rule}:lstm_parameter_matched"
        results["selection"][matched_key] = {
            "candidate": closest["name"],
            "parameters": int(closest["parameters"]),
            "qlstm_parameters": qlstm_parameters,
            "relative_gap": relative_gap,
            "within_tolerance": relative_gap
            <= float(selection_cfg["parameter_match_tolerance"]),
        }
        report(
            matched_key,
            slice_probabilities[closest["name"]]["test"],
            closest[["tau_long", "tau_short"]].to_dict(),
            {
                "candidate": closest["name"],
                "parameters": int(closest["parameters"]),
                "relative_gap": relative_gap,
            },
        )

    if selection_cfg.get("report_all_candidates", False):
        # E5 stage 2: every listed candidate is a fixed seed-0 selection, reported
        # with its own tau calibration and no re-selection.
        for _, candidate_row in table.iterrows():
            report(
                f"candidate:{candidate_row['name']}",
                slice_probabilities[candidate_row["name"]]["test"],
                candidate_row[["tau_long", "tau_short"]].to_dict(),
                {
                    "candidate": candidate_row["name"],
                    "parameters": int(candidate_row["parameters"]),
                },
            )

    # Non-learned and linear floors on the same test slice.
    always_long = {"tau_long": -1.0, "tau_short": -1.0}
    report("buy_and_hold", np.ones(len(test_rows)), always_long, {"parameters": 0})
    never_trade = {"tau_long": 2.0, "tau_short": -1.0}
    report("always_flat", np.full(len(test_rows), 0.5), never_trade, {"parameters": 0})
    flat_inputs = scaler.transform(windows).reshape(len(windows), -1)
    fit_rows = np.concatenate([split.train, split.early_stop, split.model_select])
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        flat_inputs[fit_rows], labels[fit_rows]
    )
    logistic_thresholds = calibrate_thresholds(
        logistic.predict_proba(flat_inputs[split.threshold_calibration])[:, 1],
        next_returns[split.threshold_calibration],
        backtest_cfg,
    )
    report(
        "logistic_regression",
        logistic.predict_proba(flat_inputs[test_rows])[:, 1],
        logistic_thresholds,
        {"parameters": int(logistic.coef_.size + logistic.intercept_.size)},
    )

    pd.DataFrame(predictions).to_csv(run_dir / "test_predictions.csv", index=False)
    logger.info(
        "ARTIFACT_WRITTEN | path=test_predictions.csv | rows=%d", len(test_rows)
    )
    results["test_window"] = [
        str(dates[test_rows[0]].date()),
        str(dates[test_rows[-1]].date()),
    ]
    status["selection"] = {
        "name": "model_select_auc (primary) and tau_sharpe",
        "direction": "maximize",
        "split": "model_select slice (auc); threshold_calibration slice (tau_sharpe and thresholds)",
        "checkpoint": "best early-stop AUC epoch",
    }
    return results
