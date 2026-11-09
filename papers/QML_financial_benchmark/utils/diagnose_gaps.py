"""Phase 3 gap diagnostics (see LOG.md, "Gap diagnosis").

Subcommands, each writing ``results/gap_<name>.json``:

- ``sanity``: known-answer test. QNN (Hybrid-MQ, 3 qubits) and ANN trained
  with the production loop on a synthetic task whose label is a noisy linear
  function of 3 features; test AUC must approach the Bayes AUC.
- ``us64_timing``: G2. Fold-5 test AUC of logistic regression and a [64-32-1]
  ANN on the 64-D U.S. features, with all indices lagged 1-7 (our E3) versus
  the APAC/EU indices at lags 0-6 (same-day returns known before the U.S.
  close, the Table V convention) and U.S. indices at lags 1-7.
- ``trading_null``: G3. Per regime, the E5 threshold calibration and
  backtest applied to random scores; reports the Sharpe distribution and where
  the paper's Table XII values fall in it.

Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/diagnose_gaps.py sanity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

from lib import data  # noqa: E402
from lib.experiment_logging import write_json  # noqa: E402
from lib.metrics import (  # noqa: E402
    positions_from_probabilities,
    safe_auc,
    strategy_returns,
    trading_metrics,
)
from lib.models import MLPClassifier, QNNClassifier  # noqa: E402
from lib.tasks.trading import backtest, calibrate_thresholds  # noqa: E402
from lib.training import fit_classifier, predict_proba  # noqa: E402

DATA_ROOT = REPO_ROOT / "data"
CONFIGS = PROJECT_DIR / "configs"
TRAINING = {
    "lr": 1e-3,
    "batch_size": 32,
    "max_epochs": 300,
    "patience": 30,
    "class_weighting": "none",
}
PAPER_SHARPE = {
    "F1": (0.3865, -0.0477),
    "F2": (1.2585, 0.4966),
    "F3": (0.7659, 0.8947),
    "F4": (-0.6936, 1.5228),
}


def sanity() -> dict:
    """Synthetic 3-feature task with a planted linear signal; Bayes AUC estimated from the true logit."""
    generator = np.random.default_rng(0)
    features = generator.uniform(0, 1, size=(3000, 3))
    logit = 4.0 * (features @ np.array([1.0, -1.0, 0.5]) - 0.25)
    labels = (generator.uniform(size=3000) < 1 / (1 + np.exp(-logit))).astype(float)
    x, y = (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    train, early_stop, test = slice(0, 2000), slice(2000, 2400), slice(2400, 3000)
    results = {"bayes_auc": safe_auc(labels[test], logit[test])}
    for name, model in {
        "qnn_hybrid_mq_q3_L2": QNNClassifier(3, 3, 2, True, True, "angle", np.pi),
        "qnn_mq_q3_L2": QNNClassifier(3, 3, 2, False, True, "angle", np.pi),
        "ann_11": MLPClassifier(3, [11]),
    }.items():
        torch.manual_seed(0)
        fit = fit_classifier(
            model,
            x[train],
            y[train],
            x[early_stop],
            y[early_stop],
            TRAINING,
            0,
            f"sanity {name}",
        )
        results[name] = {
            "test_auc": safe_auc(labels[test], predict_proba(model, x[test])),
            "epochs_run": fit["epochs_run"],
        }
    return results


def us64_timing() -> dict:
    """Fold-5 test AUC of logistic regression and a [64-32-1] ANN for both timings, 5 U.S. tickers."""
    dataset_cfg = json.loads((CONFIGS / "defaults.json").read_text())["directional"][
        "dataset"
    ]
    results = {}
    for ticker in ["AAPL", "BA", "GILD", "DVN", "LNC"]:
        for timing, feature_timing in (
            ("all_lagged_1to7", "lagged"),
            ("apac_eu_same_day", "table_v"),
        ):
            frame = data.build_directional_frame(
                {
                    **dataset_cfg,
                    "feature_set": "64d",
                    "ticker": ticker,
                    "feature_timing": feature_timing,
                },
                DATA_ROOT,
            )
            fold = data.walk_forward_folds(frame.index, dataset_cfg)[-1]
            values = frame.drop(columns="label").to_numpy(copy=True)
            labels = frame["label"].to_numpy(copy=True)
            scaled = data.MinMaxScaler().fit(values[fold.train]).transform(values)
            fit_rows = np.concatenate([fold.train, fold.early_stop])
            logistic = LogisticRegression(max_iter=2000).fit(
                scaled[fit_rows], labels[fit_rows]
            )
            x, y = (
                torch.tensor(scaled, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32),
            )
            torch.manual_seed(0)
            ann = MLPClassifier(64, [32])
            fit_classifier(
                ann,
                x[fold.train],
                y[fold.train],
                x[fold.early_stop],
                y[fold.early_stop],
                TRAINING,
                0,
                f"{ticker} {timing}",
            )
            results[f"{ticker}:{timing}"] = {
                "logistic_test_auc": safe_auc(
                    labels[fold.validation],
                    logistic.predict_proba(scaled[fold.validation])[:, 1],
                ),
                "ann_test_auc": safe_auc(
                    labels[fold.validation], predict_proba(ann, x[fold.validation])
                ),
                "test_rows": int(len(fold.validation)),
            }
    return results


def trading_null(draws: int = 2000) -> dict:
    """Sharpe of the calibrated threshold rule applied to i.i.d. uniform scores, per regime."""
    base = json.loads((CONFIGS / "e5_original.json").read_text())["trading"]
    generator = np.random.default_rng(0)
    results = {}
    for regime in ["F1", "F2", "F3", "F4"]:
        dataset_cfg = {**base["dataset"], "regime": regime}
        frame = data.build_trading_frame(dataset_cfg, DATA_ROOT)
        _, _, next_returns, dates = data.trading_windows(
            frame, int(dataset_cfg["window"])
        )
        split = data.trading_split(dates, dataset_cfg)
        sharpes = []
        for _ in range(draws):
            scores_tau = generator.uniform(size=len(split.threshold_calibration))
            scores_test = generator.uniform(size=len(split.test))
            thresholds = calibrate_thresholds(
                scores_tau, next_returns[split.threshold_calibration], base["backtest"]
            )
            sharpes.append(
                backtest(
                    scores_test,
                    next_returns[split.test],
                    thresholds,
                    float(base["backtest"]["fee"]),
                )["sharpe"]
            )
        sharpes = np.asarray(sharpes)
        buy_and_hold = trading_metrics(
            strategy_returns(
                np.ones(len(split.test)),
                next_returns[split.test],
                float(base["backtest"]["fee"]),
            ),
            positions_from_probabilities(np.ones(len(split.test)), -1, -1),
        )["sharpe"]
        results[regime] = {
            "null_sharpe_quantiles": {
                str(q): float(np.quantile(sharpes, q))
                for q in (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)
            },
            "null_share_exactly_zero": float(np.mean(sharpes == 0.0)),
            "buy_and_hold_sharpe": buy_and_hold,
            "paper_qlstm_sharpe_percentile": float(
                np.mean(sharpes <= PAPER_SHARPE[regime][0])
            ),
            "paper_lstm_sharpe_percentile": float(
                np.mean(sharpes <= PAPER_SHARPE[regime][1])
            ),
            "draws": draws,
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("diagnostic", choices=["sanity", "us64_timing", "trading_null"])
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = {
        "sanity": sanity,
        "us64_timing": us64_timing,
        "trading_null": trading_null,
    }[args.diagnostic]()
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    write_json(results_dir / f"gap_{args.diagnostic}.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
