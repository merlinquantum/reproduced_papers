"""Shared-runtime entry point for the QRC realized-volatility reproduction.

Experiments (selected with ``experiment`` in the config or ``--experiment``):

``paper_table2``
    Reproduce paper Table II / Table III: QR1 and QR2 on the authors' saved
    reservoir instances and optimal feature sets, the nine classical baselines,
    the extra fair controls, MCS p-values and Diebold-Mariano tests, at horizons
    ``S = 1`` and ``S = 5``.
``reservoir_instance_sweep``
    Evaluate every saved reservoir instance to quantify how much of the reported
    quantum advantage comes from selecting the best of 100 draws, and compare
    test-set selection (the paper's protocol) against leakage-free
    validation-window selection.
``feature_selection_sweep``
    Wrapper forward selection over the feature pool (paper Fig. 5 / Fig. 6),
    scored both on the out-of-sample window (paper protocol) and on a held-out
    validation window inside the training sample.
``photonic``
    MerLin photonic adaptation; see :mod:`lib.photonic`.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    QRC_FEATURE_POOL,
    build_lagged_inputs,
    build_regressor_frame,
    denormalise_log_rv,
    load_coupling_instances,
    load_normalised_table,
    rolling_windows,
    sample_coupling_instances,
)
from .experiment_logging import configure_logging, git_state, sha256, utc_now, write_json

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "qrc_volatility"


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #
PAPER_NAME = "qrc_volatility"


def _data_dir(cfg) -> Path:
    """Resolve the directory holding ``Data.CSV`` and ``coeff_10.jld2``.

    An explicit ``dataset.root`` always wins. Otherwise the shared data root is
    tried first, falling back to ``<repo>/data/qrc_volatility``; the fallback is
    needed because this paper folder carries its own ``implementation.py``, which
    makes the runtime's repo-root heuristic resolve ``data_root`` inside the
    paper directory.
    """
    explicit = cfg.get("dataset", {}).get("root")
    if explicit:
        return Path(explicit)
    shared = cfg.get("data_root")
    if shared:
        for candidate in (Path(shared) / PAPER_NAME, Path(shared)):
            if (candidate / "Data.CSV").exists():
                return candidate
    return DEFAULT_DATA_DIR


def _json_safe(value):
    """Make a resolved config snapshot serialisable.

    The shared runtime replaces ``dtype`` with a real NumPy/torch dtype object
    and paths with ``Path`` instances, neither of which ``json`` can encode.
    """
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _status_skeleton(run_id, cfg, run_dir, *, sweep_id=None, candidate=None, seed=None):
    absolute_dir = Path(run_dir).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sweep_id": sweep_id,
        "candidate": candidate,
        "seed": seed if seed is not None else cfg.get("seed"),
        "repetition": 0,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": {
            "name": cfg["dataset"]["name"],
            "splits": {
                "train_window_months": 816 - cfg["evaluation"]["n_out_of_sample"],
                "out_of_sample_months": cfg["evaluation"]["n_out_of_sample"],
                "validation_months": cfg["evaluation"]["n_validation"],
            },
            "preprocessing": (
                "authors' min-max normalised Data.CSV; log RV recovered with the "
                "reference code's Min_RV/Max_RV constants; exogenous econometric "
                "regressors ADF-differenced at the 5 % level"
            ),
            "subset": "full sample 1950-01..2017-12 (816 monthly observations)",
        },
        "selection": {
            "name": cfg["evaluation"]["selection_metric"],
            "direction": "minimize",
            "split": cfg["evaluation"]["selection_split"],
            "checkpoint": "final (closed-form ridge readout; no checkpointing)",
        },
        # Absolute: the logging contract's validator resolves a relative path
        # against the run directory, which would double a relative prefix.
        "log_path": str(absolute_dir / "run.log"),
        "config_path": str(absolute_dir / "config_snapshot.json"),
        "config_sha256": None,
        "metrics_path": str(absolute_dir / "metrics.json"),
        "metrics_sha256": None,
        "error": None,
    }


def _finalise(status, run_dir, metrics):
    write_json(run_dir / "metrics.json", metrics)
    logger.info("METRICS_WRITTEN | path=%s", run_dir / "metrics.json")
    status["metrics_sha256"] = sha256(run_dir / "metrics.json")
    status["status"] = "COMPLETED"
    status["completed_at"] = utc_now()
    write_json(run_dir / "run_status.json", status)
    logger.info("RUN_COMPLETED | run_id=%s", status["run_id"])


def _prepare(cfg, run_dir, *, log_name="run.log"):
    """Configure logging, snapshot the config, and emit ``RUN_STARTED``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        run_dir / log_name,
        level=getattr(logging, str(cfg.get("logging", {}).get("level", "info")).upper()),
    )
    run_id = f"{cfg['experiment']}-{uuid.uuid4().hex[:8]}"
    write_json(run_dir / "config_snapshot.json", _json_safe(cfg))
    status = _status_skeleton(run_id, cfg, run_dir)
    status["config_sha256"] = sha256(run_dir / "config_snapshot.json")
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | experiment=%s | seed=%s",
        run_id, cfg["experiment"], cfg.get("seed"),
    )
    return run_dir, status


def _write_candidate_start(candidate_dir, cfg, name, sweep_id, candidate, seed=None,
                           dataset_note=""):
    """Create the per-candidate evidence files a sweep run must own.

    Parameters
    ----------
    candidate_dir : pathlib.Path
        Directory that will hold this candidate's run evidence.
    cfg : dict
        Resolved configuration, snapshotted alongside the candidate values.
    name : str
        Candidate run ID.
    sweep_id : str
        Owning sweep's ID.
    candidate : dict
        The candidate's varied parameter values.
    seed : int or None
        Seed recorded for this candidate; must match the coordinator's ledger
        entry. Falls back to ``cfg["seed"]``. Default value is None.
    dataset_note : str
        Extra detail appended to the candidate's ``DATASET_READY`` record.
        Default value is "".

    Returns
    -------
    dict
        The candidate's initial run status.
    """
    candidate_dir.mkdir(parents=True, exist_ok=True)
    write_json(candidate_dir / "config_snapshot.json",
               _json_safe({**cfg, "_candidate": candidate}))
    status = _status_skeleton(
        name, cfg, candidate_dir, sweep_id=sweep_id, candidate=candidate, seed=seed
    )
    status["config_sha256"] = sha256(candidate_dir / "config_snapshot.json")
    write_json(candidate_dir / "run_status.json", status)
    # A candidate's run.log has to stand on its own under the logging contract,
    # so it repeats the dataset record the coordinator logged once.
    (candidate_dir / "run.log").write_text(
        f"{utc_now()} [INFO] RUN_STARTED | run_id={name} | sweep_id={sweep_id} "
        f"| candidate={json.dumps(candidate)}\n"
        f"{utc_now()} [INFO] DATASET_READY | name={cfg['dataset']['name']} "
        f"| out_of_sample={cfg['evaluation']['n_out_of_sample']} "
        f"| validation={cfg['evaluation']['n_validation']}{dataset_note}\n",
        encoding="utf-8",
    )
    return status


def _write_candidate_end(candidate_dir, status, metrics, extra_line):
    """Finish a candidate run without touching the coordinator's log handler."""
    write_json(candidate_dir / "metrics.json", metrics)
    status["metrics_sha256"] = sha256(candidate_dir / "metrics.json")
    status["status"] = "COMPLETED"
    status["completed_at"] = utc_now()
    write_json(candidate_dir / "run_status.json", status)
    with (candidate_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} [INFO] METRICS_WRITTEN | path=metrics.json\n")
        handle.write(f"{utc_now()} [INFO] {extra_line}\n")
        handle.write(f"{utc_now()} [INFO] RUN_COMPLETED | run_id={status['run_id']}\n")


class Sample:
    """Everything the experiments need from the dataset.

    Attributes
    ----------
    normalised : pandas.DataFrame
        Authors' min-max normalised feature table (the quantum reservoir's input).
    frame : pandas.DataFrame
        Raw ``log RV`` plus ADF-differenced exogenous regressors (the econometric
        baselines' input).
    target : numpy.ndarray
        Normalised realized volatility, i.e. the ridge readout's regression target.
    actual : dict of int to numpy.ndarray
        Raw ``log RV`` truth aligned with each horizon's forecasts.
    train_slices, origin_index : list
        Rolling one-step-ahead re-estimation schedule.
    validation_slices, validation_index : list
        Same scheme applied strictly inside the initial training sample, used for
        leakage-free model selection.
    """

    def __init__(self, cfg):
        self.n_lags = cfg["dataset"]["n_lags"]
        data_dir = _data_dir(cfg)
        self.normalised = load_normalised_table(data_dir)
        self.frame = build_regressor_frame(self.normalised)
        self.target = self.normalised["RV"].to_numpy()
        self.raw_rv = self.frame["RV"].to_numpy()
        self.n_total = len(self.normalised)
        self.n_out = cfg["evaluation"]["n_out_of_sample"]
        self.train_slices, self.origin_index = rolling_windows(self.n_total, self.n_out)
        self.horizons = list(cfg["evaluation"]["horizons"])

        self.actual, self.valid = {}, {}
        for horizon in self.horizons:
            rows = np.asarray(self.origin_index) + horizon - 1
            keep = rows < self.n_total
            self.valid[horizon] = keep
            self.actual[horizon] = self.raw_rv[rows[keep]]
        self.dates = self.normalised.index[np.asarray(self.origin_index)]

        self.n_validation = cfg["evaluation"]["n_validation"]
        first_train_end = self.train_slices[0][1]
        self.validation_slices, self.validation_index = rolling_windows(
            first_train_end, self.n_validation
        )
        self.validation_actual = self.raw_rv[np.asarray(self.validation_index)]
        logger.info(
            "DATASET_READY | rows=%d | forecasts=%d | first_forecast=%s | "
            "last_forecast=%s | validation_months=%d",
            self.n_total, self.n_out, self.dates[0].date(), self.dates[-1].date(),
            self.n_validation,
        )


def _forecast_metrics(sample, forecasts_normalised, horizon):
    """Denormalise forecasts to ``log RV`` and score them."""
    from . import metrics as metric_lib

    keep = sample.valid[horizon]
    predicted = denormalise_log_rv(np.asarray(forecasts_normalised)[keep])
    return predicted, metric_lib.summarise(predicted, sample.actual[horizon])


def _renormalise(raw_forecasts):
    """Map raw ``log RV`` forecasts back onto the normalised ``[-1, 0]`` scale."""
    from .data import LOG_RV_RANGE, MIN_LOG_RV

    return {
        horizon: (np.asarray(values) - MIN_LOG_RV) / LOG_RV_RANGE - 1.0
        for horizon, values in raw_forecasts.items()
    }


# --------------------------------------------------------------------------- #
# Quantum reservoir evaluation
# --------------------------------------------------------------------------- #
def _run_qrc_variant(sample, cfg, variant, couplings, features):
    """Forecasts of one quantum reservoir, per horizon plus the validation window."""
    from .qrc import QuantumReservoir, closed_loop_forecast, rolling_ridge_forecast

    quantum = cfg["quantum_reservoir"]
    virtual_nodes = quantum["virtual_nodes"][variant]
    features = list(features)
    windows = build_lagged_inputs(sample.normalised, features, sample.n_lags)
    reservoir = QuantumReservoir(
        couplings,
        len(features),
        n_qubits=quantum["n_qubits_total"],
        tau=quantum["evolution_time_tau"],
        virtual_nodes=virtual_nodes,
        field=quantum["transverse_field_v"],
    )
    readout = np.zeros((sample.n_total, reservoir.n_readout))
    readout[sample.n_lags:] = reservoir.evaluate(windows[sample.n_lags:])

    out = {
        1: rolling_ridge_forecast(
            readout, sample.target, sample.train_slices, sample.origin_index,
            delta=quantum["ridge_delta"],
        ),
        "validation": rolling_ridge_forecast(
            readout, sample.target, sample.validation_slices, sample.validation_index,
            delta=quantum["ridge_delta"],
        ),
    }
    for horizon in sample.horizons:
        if horizon == 1:
            continue
        out[horizon] = closed_loop_forecast(
            windows, sample.target, couplings, sample.train_slices, sample.origin_index,
            rv_column=features.index("RV"), horizon=horizon,
            n_qubits=quantum["n_qubits_total"], tau=quantum["evolution_time_tau"],
            virtual_nodes=virtual_nodes, delta=quantum["ridge_delta"], readout=readout,
        )
    return out


# --------------------------------------------------------------------------- #
# Classical baselines and fair controls
# --------------------------------------------------------------------------- #
def _run_classical(sample, cfg):
    """Every enabled classical baseline, keyed by model name then horizon."""
    from . import baselines as base

    horizons = tuple(sample.horizons)
    slices, origins = sample.train_slices, sample.origin_index
    settings = cfg["baselines"]
    enabled = set(settings["enabled"])
    out: dict[str, dict] = {}

    if "har" in enabled:
        out["HAR"] = _renormalise(base.har_forecasts(
            sample.frame, slices, origins, horizons, reference_misalignment=True))
        out["HAR-aligned"] = _renormalise(base.har_forecasts(
            sample.frame, slices, origins, horizons))
    if "harx" in enabled:
        out["HARX"] = _renormalise(base.har_forecasts(
            sample.frame, slices, origins, horizons, exogenous=base.HARX_EXOGENOUS,
            window_start=settings["harx_window_start"], reference_misalignment=True))
        out["HARX-aligned"] = _renormalise(base.har_forecasts(
            sample.frame, slices, origins, horizons, exogenous=base.HARX_EXOGENOUS,
            window_start=settings["harx_window_start"]))
    if "ar1" in enabled:
        out["AR1"] = _renormalise(
            base.ar_forecasts(sample.frame, slices, origins, 1, horizons))
    if "ar3" in enabled:
        out["AR3"] = _renormalise(
            base.ar_forecasts(sample.frame, slices, origins, 3, horizons))
    if "armax" in enabled:
        # Match the authors' exogenous block, which is built after the HAR and
        # HARX helper columns have been appended to their `dff` frame.
        armax_frame = sample.frame.copy()
        for name in base.HARX_EXOGENOUS:
            armax_frame[f"{name}_lag1"] = armax_frame[name].shift(1)
        out["ARMAX"] = _renormalise(base.armax_forecasts(
            armax_frame.fillna(0.0), slices, origins, horizons))

    # The neural and reservoir baselines are driven by the normalised table so
    # their input scale matches the paper's `dff` frame up to an affine map; the
    # authors' raw `dff.csv` is not published (see LOG.md deviation D3).
    neural = [c for c in base.NEURAL_FEATURES if c in sample.normalised.columns]
    windows_by_model = {
        "rc": build_lagged_inputs(sample.normalised, ["RV"], sample.n_lags),
        "rcx": build_lagged_inputs(sample.normalised, neural, sample.n_lags),
        "lstm": build_lagged_inputs(sample.normalised, ["RV"], sample.n_lags),
        "lstmx": build_lagged_inputs(sample.normalised, neural, sample.n_lags),
    }
    for name in ("rc", "rcx"):
        if name not in enabled:
            continue
        opts = settings[name]
        out[name.upper()] = base.esn_forecasts(
            windows_by_model[name], sample.target, slices, origins,
            n_units=opts["n_units"], horizons=horizons, rv_column=0, ridge=opts["ridge"],
            leak_rate=opts["leak_rate"], spectral_radius=opts["spectral_radius"],
            input_scaling=opts["input_scaling"], seed=cfg["seed"],
        )
        logger.info("EVALUATION_COMPLETED | model=%s | n_units=%d",
                    name.upper(), opts["n_units"])
    for name in ("lstm", "lstmx"):
        if name not in enabled:
            continue
        opts = settings[name]
        logger.info("TRAINING_STARTED | model=%s | rolling_refits=%d | epochs=%d",
                    name.upper(), len(origins), opts["epochs"])
        out[name.upper()] = base.lstm_forecasts(
            windows_by_model[name], sample.target, slices, origins,
            hidden_size=opts["hidden_size"], horizons=horizons, rv_column=0,
            n_layers=opts["layers"], epochs=opts["epochs"],
            batch_size=opts["batch_size"], lr=opts["lr"], seed=cfg["seed"],
        )
    return out


def _run_iso_controls(sample, cfg, features):
    """Controls matched to the quantum reservoir's inputs, readout size and protocol.

    Two selection protocols are reported for the echo state controls so the
    quantum numbers can be compared like for like:

    ``ESN-iso-<d>-besttest``
        Lowest out-of-sample MSE over ``n_instances`` random draws, mirroring the
        paper's "best of 100 reservoirs" protocol (and its test-set leakage).
    ``ESN-iso-<d>-valsel``
        The draw with the lowest MSE on the leakage-free validation window.
    """
    from . import baselines as base
    from . import metrics as metric_lib

    horizons = tuple(sample.horizons)
    features = list(features)
    windows = build_lagged_inputs(sample.normalised, features, sample.n_lags)
    rv_column = features.index("RV")
    quantum = cfg["quantum_reservoir"]
    iso = cfg["controls"]["esn_iso"]
    controls = {
        "Linear-lag": base.linear_lag_forecasts(
            windows, sample.target, sample.train_slices, sample.origin_index,
            horizons=horizons, rv_column=rv_column, ridge=quantum["ridge_delta"],
        )
    }
    logger.info("EVALUATION_COMPLETED | model=Linear-lag | regressors=%d",
                windows.shape[1] * windows.shape[2] + 1)

    for readout_dim in iso["readout_dims"]:
        draws = []
        for seed in range(iso["n_instances"]):
            shared = dict(
                n_units=readout_dim, rv_column=rv_column, ridge=quantum["ridge_delta"],
                leak_rate=iso["leak_rate"], spectral_radius=iso["spectral_radius"],
                input_scaling=iso["input_scaling"], seed=seed,
            )
            forecasts = base.esn_forecasts(
                windows, sample.target, sample.train_slices, sample.origin_index,
                horizons=horizons, **shared)
            validation = base.esn_forecasts(
                windows, sample.target, sample.validation_slices,
                sample.validation_index, horizons=(1,), **shared)
            _, test_scores = _forecast_metrics(sample, forecasts[1], 1)
            validation_scores = metric_lib.summarise(
                denormalise_log_rv(validation[1]), sample.validation_actual)
            draws.append((seed, test_scores["mse"], validation_scores["mse"], forecasts))

        best_test = min(draws, key=lambda draw: draw[1])
        best_validation = min(draws, key=lambda draw: draw[2])
        controls[f"ESN-iso-{readout_dim}-besttest"] = best_test[3]
        controls[f"ESN-iso-{readout_dim}-valsel"] = best_validation[3]
        logger.info(
            "EVALUATION_COMPLETED | model=ESN-iso-%d | draws=%d | "
            "best_test_mse=%.6f (seed=%d) | val_selected_seed=%d test_mse=%.6f | "
            "test_mse_mean=%.6f | test_mse_std=%.6f",
            readout_dim, len(draws), best_test[1], best_test[0], best_validation[0],
            best_validation[1], float(np.mean([d[1] for d in draws])),
            float(np.std([d[1] for d in draws], ddof=1)),
        )
    return controls


# --------------------------------------------------------------------------- #
# Experiment: paper_table2
# --------------------------------------------------------------------------- #
def _paper_table2(cfg, run_dir):
    from . import metrics as metric_lib

    run_dir, status = _prepare(cfg, run_dir)
    sample = Sample(cfg)
    quantum = cfg["quantum_reservoir"]
    couplings = load_coupling_instances(_data_dir(cfg))

    forecasts: dict[str, dict] = {}
    for variant in cfg["quantum_reservoir"]["variants"]:
        features = cfg["feature_selection"][f"paper_optimal_{variant.lower()}"]
        instance = quantum["paper_instance"][variant]
        logger.info("EVALUATION_STARTED | model=%s | instance=%d | features=%s",
                    variant, instance, ",".join(features))
        forecasts[variant] = _run_qrc_variant(
            sample, cfg, variant, couplings[instance], features
        )
    forecasts.update(_run_classical(sample, cfg))
    if cfg["controls"]["enabled"]:
        forecasts.update(
            _run_iso_controls(sample, cfg, cfg["feature_selection"]["paper_optimal_qr1"])
        )

    results = {"experiment": "paper_table2", "horizons": {}}
    for horizon in sample.horizons:
        losses_mse, losses_qlike, rows = {}, {}, {}
        columns = {"date": sample.dates[sample.valid[horizon]],
                   "actual": sample.actual[horizon]}
        for name, per_horizon in forecasts.items():
            if horizon not in per_horizon:
                continue
            predicted, scores = _forecast_metrics(sample, per_horizon[horizon], horizon)
            rows[name] = scores
            columns[name] = predicted
            losses_mse[name] = (predicted - sample.actual[horizon]) ** 2
            losses_qlike[name] = metric_lib.qlike_losses(predicted, sample.actual[horizon])
            logger.info("EVALUATION_COMPLETED | model=%s | S=%d | mse=%.6f | qlike=%.4f",
                        name, horizon, scores["mse"], scores["qlike"])

        mcs_settings = cfg["evaluation"]["mcs"]
        for label, losses in (("mse", losses_mse), ("qlike", losses_qlike)):
            p_values = metric_lib.model_confidence_set(
                losses, size=mcs_settings["size"], reps=mcs_settings["reps"],
                seed=cfg["seed"],
                min_observations=mcs_settings.get("min_observations", 60),
            )
            for name, value in p_values.items():
                rows[name][f"mcs_p_{label}"] = value

        names = list(rows)
        dm_stat = pd.DataFrame(0.0, index=names, columns=names)
        dm_p = pd.DataFrame(np.nan, index=names, columns=names)
        for position, first in enumerate(names):
            for second in names[position + 1:]:
                stat, p_value = metric_lib.diebold_mariano(
                    sample.actual[horizon], columns[first], columns[second])
                dm_stat.loc[first, second] = stat
                dm_p.loc[first, second] = p_value
        results["horizons"][str(horizon)] = {
            "n_observations": int(sample.valid[horizon].sum()),
            "models": rows,
            "diebold_mariano_statistic": dm_stat.to_dict(),
            "diebold_mariano_pvalue": dm_p.to_dict(),
        }
        path = run_dir / f"predictions_S{horizon}.csv"
        pd.DataFrame(columns).set_index("date").to_csv(path)
        logger.info("ARTIFACT_WRITTEN | path=%s", path)

    _finalise(status, run_dir, results)
    return results


# --------------------------------------------------------------------------- #
# Experiment: reservoir_instance_sweep
# --------------------------------------------------------------------------- #
def _reservoir_instance_sweep(cfg, run_dir):
    from . import metrics as metric_lib

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir / "sweep.log")
    sweep_id = f"instances-{uuid.uuid4().hex[:8]}"
    sweep = cfg["instance_sweep"]
    variants = list(sweep["variants"])
    n_instances = sweep["n_instances"]
    expected = len(variants) * n_instances

    sweep_status = {
        "schema_version": SCHEMA_VERSION, "sweep_id": sweep_id, "status": "RUNNING",
        "started_at": utc_now(), "completed_at": None, "code": git_state(REPO_ROOT),
        "expected_runs": expected, "runs": [],
        "summary_path": str((run_dir / "sweep_summary.json").resolve()),
        "selected_candidates": None, "error": None,
    }
    write_json(run_dir / "sweep_status.json", sweep_status)
    logger.info("SWEEP_STARTED | sweep_id=%s | expected_runs=%d | selection_metric=%s "
                "| selection_split=%s | direction=minimize",
                sweep_id, expected, cfg["evaluation"]["selection_metric"],
                cfg["evaluation"]["selection_split"])

    sample = Sample(cfg)
    if sweep["coupling_source"] == "authors":
        couplings = load_coupling_instances(_data_dir(cfg))
    else:
        couplings = sample_coupling_instances(
            n_instances, cfg["quantum_reservoir"]["n_qubits_total"], cfg["seed"])
    if len(couplings) < n_instances:
        raise ValueError(f"only {len(couplings)} coupling instances available")

    records = []
    for variant in variants:
        features = cfg["feature_selection"][f"paper_optimal_{variant.lower()}"]
        for instance in range(n_instances):
            candidate = {"variant": variant, "instance": instance}
            name = f"{variant}-i{instance:03d}"
            candidate_dir = run_dir / "candidates" / name
            candidate_status = _write_candidate_start(
                candidate_dir, cfg, name, sweep_id, candidate, seed=cfg["seed"],
                dataset_note=f" | features={','.join(features)}")
            logger.info("CANDIDATE_STARTED | run_id=%s | candidate=%s",
                        name, json.dumps(candidate))
            try:
                out = _run_qrc_variant(sample, cfg, variant, couplings[instance], features)
                _, test_scores = _forecast_metrics(sample, out[1], 1)
                validation_scores = metric_lib.summarise(
                    denormalise_log_rv(out["validation"]), sample.validation_actual)
                _write_candidate_end(
                    candidate_dir, candidate_status,
                    {"candidate": candidate, "test": test_scores,
                     "validation": validation_scores},
                    f"EVALUATION_COMPLETED | test_mse={test_scores['mse']:.6f} | "
                    f"validation_mse={validation_scores['mse']:.6f}",
                )
                np.save(candidate_dir / "forecast_S1.npy", out[1])
                records.append({
                    "variant": variant, "instance": instance, "status": "DONE",
                    "metrics_path": str((candidate_dir / "metrics.json").resolve()),
                    "run_id": name, "run_dir": str(candidate_dir.resolve()),
                })
                logger.info("CANDIDATE_COMPLETED | run_id=%s | metrics=%s | "
                            "test_mse=%.6f | validation_mse=%.6f",
                            name, candidate_dir / "metrics.json", test_scores["mse"],
                            validation_scores["mse"])
            except Exception as exc:  # noqa: BLE001 - recorded; the sweep continues
                candidate_status.update(
                    status="FAILED", completed_at=utc_now(),
                    error={"type": type(exc).__name__, "message": str(exc)})
                write_json(candidate_dir / "run_status.json", candidate_status)
                records.append({
                    "variant": variant, "instance": instance, "status": "FAILED",
                    "metrics_path": None, "run_id": name,
                    "run_dir": str(candidate_dir.resolve()),
                })
                logger.exception("CANDIDATE_FAILED | run_id=%s", name)
            sweep_status["runs"].append({
                "run_id": name, "candidate": candidate, "seed": cfg["seed"],
                "repetition": 0, "status": candidate_status["status"],
                "run_dir": str(candidate_dir.resolve()),
                "run_log": str((candidate_dir / "run.log").resolve()),
                "run_status": str((candidate_dir / "run_status.json").resolve()),
                "config_path": str((candidate_dir / "config_snapshot.json").resolve()),
                "metrics_path": records[-1]["metrics_path"],
            })
            write_json(run_dir / "sweep_status.json", sweep_status)

    summary = _summarise_instance_sweep(run_dir, records, cfg)
    write_json(run_dir / "sweep_summary.json", summary)
    completed = sum(1 for record in records if record["status"] == "DONE")
    sweep_status["selected_candidates"] = summary["selected"]
    sweep_status["status"] = "COMPLETED" if completed == expected else "PARTIAL"
    sweep_status["completed_at"] = utc_now()
    write_json(run_dir / "sweep_status.json", sweep_status)
    logger.info("%s | summary=%s | selected=%s | completed=%d/%d",
                "SWEEP_COMPLETED" if completed == expected else "SWEEP_PARTIAL",
                run_dir / "sweep_summary.json", json.dumps(summary["selected"]),
                completed, expected)
    return summary


def _summarise_instance_sweep(run_dir, records, cfg):
    """Build the sweep table programmatically from each candidate's metrics.json."""
    rows = []
    for record in records:
        row = {"variant": record["variant"], "instance": record["instance"],
               "status": record["status"], "run_id": record["run_id"]}
        if record["metrics_path"] is not None:
            with open(record["metrics_path"], encoding="utf-8") as handle:
                payload = json.load(handle)
            row.update(
                test_mse=payload["test"]["mse"], test_qlike=payload["test"]["qlike"],
                validation_mse=payload["validation"]["mse"],
                validation_qlike=payload["validation"]["qlike"],
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "instance_sweep.csv", index=False)

    summary = {"experiment": "reservoir_instance_sweep",
               "selection_rule": {
                   "metric": cfg["evaluation"]["selection_metric"],
                   "direction": "minimize",
                   "tie_tolerance": cfg["evaluation"]["tie_tolerance"],
               },
               "per_variant": {}, "selected": {}}
    done = frame[frame["status"] == "DONE"]
    finite = done[np.isfinite(done["test_mse"]) & np.isfinite(done["validation_mse"])]
    for variant, group in finite.groupby("variant"):
        best_test = group.loc[group["test_mse"].idxmin()]
        best_validation = group.loc[group["validation_mse"].idxmin()]
        tolerance = cfg["evaluation"]["tie_tolerance"]
        tied = group[group["validation_mse"] <= best_validation["validation_mse"] + tolerance]
        summary["per_variant"][variant] = {
            "n_completed": int(len(group)),
            "n_expected": int(cfg["instance_sweep"]["n_instances"]),
            "test_mse": {
                "mean": float(group["test_mse"].mean()),
                "std": float(group["test_mse"].std(ddof=1)),
                "median": float(group["test_mse"].median()),
                "min": float(group["test_mse"].min()),
                "max": float(group["test_mse"].max()),
                "q05": float(group["test_mse"].quantile(0.05)),
                "q95": float(group["test_mse"].quantile(0.95)),
            },
            "test_qlike": {
                "mean": float(group["test_qlike"].mean()),
                "std": float(group["test_qlike"].std(ddof=1)),
                "median": float(group["test_qlike"].median()),
                "min": float(group["test_qlike"].min()),
            },
            "best_on_test": {
                "instance": int(best_test["instance"]),
                "test_mse": float(best_test["test_mse"]),
                "test_qlike": float(best_test["test_qlike"]),
            },
            "selected_on_validation": {
                "instance": int(best_validation["instance"]),
                "validation_mse": float(best_validation["validation_mse"]),
                "test_mse": float(best_validation["test_mse"]),
                "test_qlike": float(best_validation["test_qlike"]),
                "n_tied_within_tolerance": int(len(tied)),
            },
        }
        summary["selected"][variant] = int(best_validation["instance"])
    return summary


# --------------------------------------------------------------------------- #
# Experiment: feature_selection_sweep
# --------------------------------------------------------------------------- #
def _feature_selection_sweep(cfg, run_dir):
    from . import metrics as metric_lib

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir / "sweep.log")
    sweep_id = f"forward-{uuid.uuid4().hex[:8]}"
    settings = cfg["feature_selection"]
    pool = list(settings.get("pool") or QRC_FEATURE_POOL)
    max_features = min(settings["max_features"], len(pool))
    variants = list(settings["variants"])
    expected = sum(len(pool) - step for step in range(max_features)) * len(variants)
    selection_key = f"{settings['selection_split']}_mse"

    sweep_status = {
        "schema_version": SCHEMA_VERSION, "sweep_id": sweep_id, "status": "RUNNING",
        "started_at": utc_now(), "completed_at": None, "code": git_state(REPO_ROOT),
        "expected_runs": expected, "runs": [],
        "summary_path": str((run_dir / "sweep_summary.json").resolve()),
        "selected_candidates": None, "error": None,
    }
    write_json(run_dir / "sweep_status.json", sweep_status)
    logger.info("SWEEP_STARTED | sweep_id=%s | expected_runs=%d | pool=%s | "
                "selection_metric=mse | selection_split=%s | direction=minimize",
                sweep_id, expected, ",".join(pool), settings["selection_split"])

    sample = Sample(cfg)
    couplings = load_coupling_instances(_data_dir(cfg))
    records = []
    summary = {"experiment": "feature_selection_sweep",
               "selection_rule": {"metric": "mse", "direction": "minimize",
                                  "split": settings["selection_split"]},
               "paths": {}, "selected": {}}

    for variant in variants:
        instance = cfg["quantum_reservoir"]["paper_instance"][variant]
        selected: list[str] = []
        remaining = list(pool)
        path = []
        for _ in range(max_features):
            step_rows = []
            for feature in list(remaining):
                trial = selected + [feature]
                candidate = {"variant": variant, "features": trial}
                name = f"{variant}-k{len(trial):02d}-{feature}"
                candidate_dir = run_dir / "candidates" / name
                candidate_status = _write_candidate_start(
                    candidate_dir, cfg, name, sweep_id, candidate, seed=cfg["seed"],
                    dataset_note=f" | features={','.join(trial)}")
                logger.info("CANDIDATE_STARTED | run_id=%s | candidate=%s",
                            name, json.dumps(candidate))
                out = _run_qrc_variant(
                    sample, cfg, variant, couplings[instance], trial)
                _, test_scores = _forecast_metrics(sample, out[1], 1)
                validation_scores = metric_lib.summarise(
                    denormalise_log_rv(out["validation"]), sample.validation_actual)
                _write_candidate_end(
                    candidate_dir, candidate_status,
                    {"candidate": candidate, "test": test_scores,
                     "validation": validation_scores},
                    f"EVALUATION_COMPLETED | test_mse={test_scores['mse']:.6f} | "
                    f"validation_mse={validation_scores['mse']:.6f}",
                )
                row = {"variant": variant, "step": len(trial), "feature": feature,
                       "features": ",".join(trial), "status": "DONE",
                       "test_mse": test_scores["mse"],
                       "test_qlike": test_scores["qlike"],
                       "validation_mse": validation_scores["mse"],
                       "metrics_path": str((candidate_dir / "metrics.json").resolve())}
                step_rows.append(row)
                records.append(row)
                sweep_status["runs"].append({
                    "run_id": name, "candidate": candidate, "seed": cfg["seed"],
                    "repetition": 0, "status": "COMPLETED",
                    "run_dir": str(candidate_dir.resolve()),
                    "run_log": str((candidate_dir / "run.log").resolve()),
                    "run_status": str((candidate_dir / "run_status.json").resolve()),
                    "config_path": str((candidate_dir / "config_snapshot.json").resolve()),
                    "metrics_path": str((candidate_dir / "metrics.json").resolve())})
                logger.info("CANDIDATE_COMPLETED | run_id=%s | test_mse=%.6f | "
                            "validation_mse=%.6f", name, test_scores["mse"],
                            validation_scores["mse"])
            winner = min(step_rows, key=lambda row: row[selection_key])
            selected.append(winner["feature"])
            remaining.remove(winner["feature"])
            path.append({"step": len(selected), "added": winner["feature"],
                         "features": list(selected), "test_mse": winner["test_mse"],
                         "test_qlike": winner["test_qlike"],
                         "validation_mse": winner["validation_mse"]})
            logger.info("SWEEP_STEP | variant=%s | k=%d | added=%s | %s=%.6f",
                        variant, len(selected), winner["feature"], selection_key,
                        winner[selection_key])
            write_json(run_dir / "sweep_status.json", sweep_status)
        summary["paths"][variant] = path
        summary["selected"][variant] = min(
            path, key=lambda row: row[selection_key])["features"]

    pd.DataFrame(records).to_csv(run_dir / "feature_selection.csv", index=False)
    write_json(run_dir / "sweep_summary.json", summary)
    completed = len(records)
    sweep_status["selected_candidates"] = summary["selected"]
    sweep_status["status"] = "COMPLETED" if completed == expected else "PARTIAL"
    sweep_status["completed_at"] = utc_now()
    write_json(run_dir / "sweep_status.json", sweep_status)
    logger.info("%s | summary=%s | selected=%s | completed=%d/%d",
                "SWEEP_COMPLETED" if completed == expected else "SWEEP_PARTIAL",
                run_dir / "sweep_summary.json", json.dumps(summary["selected"]),
                completed, expected)
    return summary


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
EXPERIMENTS = {
    "paper_table2": _paper_table2,
    "reservoir_instance_sweep": _reservoir_instance_sweep,
    "feature_selection_sweep": _feature_selection_sweep,
}


def train_and_evaluate(cfg, run_dir) -> None:
    """Shared-runtime entry point.

    Parameters
    ----------
    cfg : dict
        Resolved configuration; ``cfg["experiment"]`` selects the experiment.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Raises
    ------
    KeyError
        If ``cfg["experiment"]`` names an unknown experiment.
    """
    name = cfg.get("experiment")
    experiments = dict(EXPERIMENTS)
    if name == "photonic":
        from .photonic import run_photonic

        experiments["photonic"] = run_photonic
    if name not in experiments:
        raise KeyError(
            f"unknown experiment {name!r}; expected one of {sorted(experiments)}"
        )
    run_dir = Path(run_dir)
    try:
        experiments[name](cfg, run_dir)
    except Exception as exc:
        status_path = run_dir / "run_status.json"
        if status_path.exists():
            with status_path.open(encoding="utf-8") as handle:
                status = json.load(handle)
            status.update(status="FAILED", completed_at=utc_now(),
                          error={"type": type(exc).__name__, "message": str(exc)})
            write_json(status_path, status)
        logger.exception("RUN_FAILED | experiment=%s", name)
        raise
