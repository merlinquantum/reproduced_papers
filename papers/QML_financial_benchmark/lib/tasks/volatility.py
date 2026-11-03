"""Task 3: one-day-ahead realized-variance forecasting, QSVR vs SVR vs GARCH (§III-E, Tables XIII-XIV).

Protocol per run (one asset, one feature set):

- expanding window: the first ``initial_train_days`` rows train the first
  model; every ``refit_every`` rows all models are refit on all earlier rows
  (§III-E1), optionally capped at ``max_train_days``;
- SVR and QSVR hyperparameters are tuned by Optuna TPE with the same trial
  budget for every model: fit on the first ``1 - validation_fraction`` of the
  training window, score mean QLIKE on the rest, then refit on the whole window;
- SVR-type models see min-max scaled features and a target divided by the
  training-window mean RV, because the raw RV (~1e-4) sits far inside the
  paper's epsilon range [1e-3, 1]; forecasts are mapped back to RV units;
- every forecast is floored at the smallest positive training-window RV so
  QLIKE stays defined; the number of floored forecasts is reported.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from arch import arch_model
from sklearn.svm import SVR

from .. import data
from ..kernels import amplitude_kernel, angle_feature_states, fidelity_kernel
from ..metrics import diebold_mariano, forecast_metrics, qlike
from ..photonic_merlin import photonic_feature_states

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SVR_MODELS = {
    "svr_linear",
    "svr_poly",
    "svr_rbf",
    "svr_rbf_wide",
    "qsvr_angle",
    "qsvr_amplitude",
    "photonic_kernel",
}
BASELINE_MODELS = {"persistence", "har", "garch", "garch_sigma2"}


def refit_blocks(n_rows: int, dataset_cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(train_rows, test_rows)`` per refit block of the expanding window."""
    initial, step = (
        int(dataset_cfg["initial_train_days"]),
        int(dataset_cfg["refit_every"]),
    )
    cap = dataset_cfg["max_train_days"]
    blocks = []
    for start in range(initial, n_rows, step):
        first_train = 0 if cap is None else max(0, start - int(cap))
        blocks.append(
            (np.arange(first_train, start), np.arange(start, min(start + step, n_rows)))
        )
    if not blocks:
        raise ValueError(
            f"No test rows: {n_rows} rows but initial_train_days={initial}"
        )
    return blocks


def suggest_params(trial: optuna.Trial, model: str, section: dict) -> dict:
    """Table VIII search space for one SVR/QSVR model."""
    tuning, qsvr = section["tuning"], section["qsvr"]
    params = {
        "C": trial.suggest_float("C", *tuning["C"], log=True),
        "epsilon": trial.suggest_float("epsilon", *tuning["epsilon"], log=True),
    }
    if model == "svr_poly":
        params["degree"] = trial.suggest_categorical("degree", [2, 3])
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    elif model == "svr_rbf":
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    elif model == "svr_rbf_wide":
        params["gamma"] = trial.suggest_float(
            "gamma", *tuning["rbf_wide_gamma"], log=True
        )
    elif model == "qsvr_angle":
        params["layers"] = trial.suggest_categorical("layers", qsvr["layers"])
        params["topology"] = trial.suggest_categorical("topology", qsvr["topologies"])
    elif model == "qsvr_amplitude":
        params["beta"] = trial.suggest_categorical("beta", qsvr["amplitude_betas"])
    elif model == "photonic_kernel":
        photonic = section["photonic"]
        params["n_photons"] = trial.suggest_categorical(
            "n_photons", photonic["n_photons"]
        )
        params["layers"] = trial.suggest_categorical("layers", photonic["layers"])
    return params


class KernelCache:
    """Quantum Gram matrices of one refit block, keyed by feature-map settings."""

    def __init__(self, scaled_features: np.ndarray, section: dict):
        self.features = scaled_features
        self.qsvr = section["qsvr"]
        self.section = section
        self.states: dict = {}

    def angle_kernel(
        self, rows_a: np.ndarray, rows_b: np.ndarray, layers: int, topology: str
    ) -> np.ndarray:
        key = (layers, topology)
        if key not in self.states:
            self.states[key] = angle_feature_states(
                self.features,
                layers,
                topology,
                float(self.qsvr["angle_scale"]),
                int(self.qsvr["rotation_seed"]),
            )
        states = self.states[key]
        return fidelity_kernel(states[rows_a], states[rows_b])

    def amplitude_kernel(
        self, rows_a: np.ndarray, rows_b: np.ndarray, beta: int
    ) -> np.ndarray:
        return amplitude_kernel(self.features[rows_a], self.features[rows_b], beta)

    def photonic_kernel(
        self, rows_a: np.ndarray, rows_b: np.ndarray, n_photons: int, layers: int
    ) -> np.ndarray:
        """Multi-photon fidelity kernel of the MerLin re-uploading feature map (Phase 4)."""
        key = ("photonic", n_photons, layers)
        if key not in self.states:
            photonic = self.section["photonic"]
            self.states[key] = photonic_feature_states(
                self.features,
                n_photons,
                layers,
                float(photonic["angle_scale"]),
                int(photonic["mesh_seed"]),
            )
        states = self.states[key]
        return fidelity_kernel(states[rows_a], states[rows_b])


def fit_predict_svr(
    model: str,
    params: dict,
    fit_rows: np.ndarray,
    predict_rows: np.ndarray,
    scaled_features: np.ndarray,
    scaled_target: np.ndarray,
    kernels: KernelCache,
) -> np.ndarray:
    """Fit one SVR/QSVR on ``fit_rows`` and predict the scaled target on ``predict_rows``."""
    svr_kwargs = {"C": params["C"], "epsilon": params["epsilon"]}
    if model.startswith("qsvr") or model == "photonic_kernel":
        if model == "photonic_kernel":
            gram = kernels.photonic_kernel(
                fit_rows, fit_rows, params["n_photons"], params["layers"]
            )
            cross = kernels.photonic_kernel(
                predict_rows, fit_rows, params["n_photons"], params["layers"]
            )
        elif model == "qsvr_angle":
            gram = kernels.angle_kernel(
                fit_rows, fit_rows, params["layers"], params["topology"]
            )
            cross = kernels.angle_kernel(
                predict_rows, fit_rows, params["layers"], params["topology"]
            )
        else:
            gram = kernels.amplitude_kernel(fit_rows, fit_rows, params["beta"])
            cross = kernels.amplitude_kernel(predict_rows, fit_rows, params["beta"])
        regressor = SVR(kernel="precomputed", **svr_kwargs).fit(
            gram, scaled_target[fit_rows]
        )
        return regressor.predict(cross)
    if model == "svr_linear":
        regressor = SVR(kernel="linear", **svr_kwargs)
    elif model == "svr_poly":
        regressor = SVR(
            kernel="poly", degree=params["degree"], gamma=params["gamma"], **svr_kwargs
        )
    else:
        regressor = SVR(kernel="rbf", gamma=params["gamma"], **svr_kwargs)
    regressor.fit(scaled_features[fit_rows], scaled_target[fit_rows])
    return regressor.predict(scaled_features[predict_rows])


def tune_svr(
    model: str,
    section: dict,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    scaled_features: np.ndarray,
    scaled_target: np.ndarray,
    target: np.ndarray,
    target_scale: float,
    floor: float,
    kernels: KernelCache,
    seed: int,
) -> optuna.Study:
    """Optuna TPE search minimizing validation QLIKE in RV units (Table VIII space)."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model, section)
        validation = fit_predict_svr(
            model,
            params,
            fit_rows,
            validation_rows,
            scaled_features,
            scaled_target,
            kernels,
        )
        forecast = np.maximum(validation * target_scale, floor)
        return float(np.mean(qlike(target[validation_rows], forecast)))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=int(section["tuning"]["n_trials"]))
    return study


def garch_forecasts(
    returns: np.ndarray,
    known_sum4: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    complete_with_known: bool = True,
) -> np.ndarray:
    """GARCH(1,1) forecasts of ``RV_{t+1}`` for every test row ``t`` (Eq. 9).

    Parameters are fitted on the training-window returns (in percent). The
    variance recursion then runs through every row up to the last test row
    with fixed parameters, giving the one-step-ahead ``sigma^2_{t+1}``. The
    5-day target combines it with the four squared returns already known at
    ``t``: ``(sum_{i=0..3} r_{t-i}^2 + sigma^2_{t+1}) / 5`` (decision D20).
    With ``complete_with_known=False`` the forecast is ``sigma^2_{t+1}``
    alone, the likely reading of the paper (ambiguity A1). Default value is True.
    """
    scaled = 100.0 * returns
    fitted = arch_model(
        scaled[train_rows],
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    ).fit(disp="off")
    mu, omega, alpha, beta = (
        float(fitted.params[name]) for name in ("mu", "omega", "alpha[1]", "beta[1]")
    )
    variance = float(np.var(scaled[train_rows]))
    next_variance = np.empty(test_rows[-1] + 1)
    for row in range(test_rows[-1] + 1):
        variance = omega + alpha * (scaled[row] - mu) ** 2 + beta * variance
        next_variance[row] = variance
    sigma2_next = next_variance[test_rows] / 1e4
    if not complete_with_known:
        return sigma2_next
    return (known_sum4[test_rows] + sigma2_next) / 5.0


def run_volatility(
    cfg: dict, section: dict, run_dir: Path, status: dict, report_progress
) -> dict:
    """Expanding-window forecasts for every configured model, metrics and DM tests.

    Artifacts written to ``run_dir``: ``test_predictions.csv`` (one column per
    model) and ``hyperparameters.json`` (tuned settings per refit block).

    Returns
    -------
    dict
        Metrics for ``metrics.json``.
    """
    dataset_cfg, tuning_cfg = section["dataset"], section["tuning"]
    models = list(section["models"])
    unknown = set(models) - SVR_MODELS - BASELINE_MODELS
    if unknown:
        raise ValueError(f"Unknown volatility models {sorted(unknown)}")

    frame = data.build_volatility_frame(dataset_cfg, cfg["data_root"])
    feature_columns = data.volatility_feature_columns(dataset_cfg)
    features = frame[feature_columns].to_numpy(dtype=np.float64)
    target = frame["target"].to_numpy()
    blocks = refit_blocks(len(frame), dataset_cfg)
    test_rows_all = np.concatenate([test for _, test in blocks])
    status["dataset"] = {
        "name": f"yahoo_daily_prices:{dataset_cfg['ticker']}:rv5_p{dataset_cfg['return_lags']}_q{dataset_cfg['rv_lags']}",
        "splits": {
            "first_train": [
                str(frame.index[0].date()),
                str(frame.index[blocks[0][0][-1]].date()),
            ],
            "test": [
                str(frame.index[test_rows_all[0]].date()),
                str(frame.index[test_rows_all[-1]].date()),
                int(len(test_rows_all)),
            ],
            "refit_blocks": len(blocks),
        },
        "preprocessing": "min-max features and mean-RV target scaling fitted per training window (SVR/QSVR only)",
        "subset": "full",
    }
    logger.info(
        "DATASET_READY | ticker=%s | rows=%d | features=%d | test_rows=%d | refit_blocks=%d | first=%s | last=%s",
        dataset_cfg["ticker"],
        len(frame),
        len(feature_columns),
        len(test_rows_all),
        len(blocks),
        frame.index[0].date(),
        frame.index[-1].date(),
    )

    forecasts = {model: np.empty(len(frame)) for model in models}
    floored = dict.fromkeys(models, 0)
    tuned: dict = {model: [] for model in models if model in SVR_MODELS}
    best_params: dict = {}
    logger.info(
        "TRAINING_STARTED | models=%s | refit_blocks=%d | optuna_trials=%d | objective=validation_qlike",
        ",".join(models),
        len(blocks),
        tuning_cfg["n_trials"],
    )
    for block_index, (train_rows, test_rows) in enumerate(blocks):
        scaler = data.MinMaxScaler().fit(features[train_rows])
        scaled_features = scaler.transform(features)
        target_scale = float(target[train_rows].mean())
        scaled_target = target / target_scale
        floor = float(target[train_rows][target[train_rows] > 0].min())
        kernels = KernelCache(scaled_features, section)
        n_validation = int(
            round(float(tuning_cfg["validation_fraction"]) * len(train_rows))
        )
        fit_rows, validation_rows = (
            train_rows[:-n_validation],
            train_rows[-n_validation:],
        )
        retune = block_index % int(tuning_cfg["tune_every_n_refits"]) == 0

        for model in models:
            if model == "persistence":
                prediction = frame["rv_now"].to_numpy()[test_rows]
            elif model == "har":
                regressors = np.column_stack(
                    [
                        np.ones(len(frame)),
                        frame["return_now"].to_numpy() ** 2,
                        frame["rv_now"].to_numpy(),
                        frame["rv_month"].to_numpy(),
                    ]
                )
                coefficients, *_ = np.linalg.lstsq(
                    regressors[train_rows], target[train_rows], rcond=None
                )
                prediction = regressors[test_rows] @ coefficients
            elif model in ("garch", "garch_sigma2"):
                prediction = garch_forecasts(
                    frame["return_now"].to_numpy(),
                    frame["squared_return_sum4"].to_numpy(),
                    train_rows,
                    test_rows,
                    complete_with_known=model == "garch",
                )
            else:
                if retune or model not in best_params:
                    study = tune_svr(
                        model,
                        section,
                        fit_rows,
                        validation_rows,
                        scaled_features,
                        scaled_target,
                        target,
                        target_scale,
                        floor,
                        kernels,
                        seed=int(tuning_cfg["sampler_seed"]) + block_index,
                    )
                    best_params[model] = {**study.best_trial.params}
                    tuned[model].append(
                        {
                            "block": block_index,
                            "validation_qlike": study.best_value,
                            **best_params[model],
                        }
                    )
                params = {**best_params[model]}
                prediction = (
                    fit_predict_svr(
                        model,
                        params,
                        train_rows,
                        test_rows,
                        scaled_features,
                        scaled_target,
                        kernels,
                    )
                    * target_scale
                )
            floored[model] += int(np.sum(prediction < floor))
            forecasts[model][test_rows] = np.maximum(prediction, floor)
            logger.info(
                "TRAIN_REFIT_COMPLETED | block=%d/%d | model=%s | train_rows=%d | test_rows=%d | train_end=%s",
                block_index + 1,
                len(blocks),
                model,
                len(train_rows),
                len(test_rows),
                frame.index[train_rows[-1]].date(),
            )
            report_progress(
                block_index * len(models) + models.index(model) + 1,
                len(blocks) * len(models),
                "model_refits",
            )

    realized = target[test_rows_all]
    previous = frame["rv_now"].to_numpy()[test_rows_all]
    results: dict = {
        "ticker": dataset_cfg["ticker"],
        "test": {},
        "diebold_mariano": {},
        "floored_forecasts": floored,
    }
    predictions = {
        "date": frame.index[test_rows_all].strftime("%Y-%m-%d"),
        "realized": realized,
    }
    for model in models:
        logger.info("EVALUATION_STARTED | split=test | model=%s", model)
        forecast = forecasts[model][test_rows_all]
        results["test"][model] = forecast_metrics(realized, forecast, previous)
        predictions[model] = forecast
        logger.info(
            "EVALUATION_COMPLETED | split=test | model=%s | qlike=%.6f | mse=%.6e | r2=%.6f | dir_acc=%.6f",
            model,
            results["test"][model]["qlike"],
            results["test"][model]["mse"],
            results["test"][model]["r2"],
            results["test"][model]["directional_accuracy"],
        )
    for reference in [model for model in models if model.startswith("qsvr")]:
        reference_forecast = forecasts[reference][test_rows_all]
        for model in models:
            if model == reference:
                continue
            forecast = forecasts[model][test_rows_all]
            # Positive statistic: `model` has the larger loss, i.e. the QSVR forecasts better.
            qlike_stat, qlike_p = diebold_mariano(
                qlike(realized, forecast), qlike(realized, reference_forecast)
            )
            mse_stat, mse_p = diebold_mariano(
                (realized - forecast) ** 2, (realized - reference_forecast) ** 2
            )
            results["diebold_mariano"][f"{model}_vs_{reference}"] = {
                "qlike_statistic": qlike_stat,
                "qlike_p_value": qlike_p,
                "mse_statistic": mse_stat,
                "mse_p_value": mse_p,
            }
    pd.DataFrame(predictions).to_csv(run_dir / "test_predictions.csv", index=False)
    logger.info("ARTIFACT_WRITTEN | path=test_predictions.csv | rows=%d", len(realized))
    pd.Series(tuned).to_json(run_dir / "hyperparameters.json", indent=2)
    logger.info("ARTIFACT_WRITTEN | path=hyperparameters.json")
    results["test_window"] = [
        str(frame.index[test_rows_all[0]].date()),
        str(frame.index[test_rows_all[-1]].date()),
    ]
    status["selection"] = {
        "name": "validation_qlike",
        "direction": "minimize",
        "split": f"last {tuning_cfg['validation_fraction']} of each training window",
        "checkpoint": "Optuna best trial, refit on the full training window",
    }
    return results
