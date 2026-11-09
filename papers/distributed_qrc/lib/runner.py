"""Runtime entry point: one architecture grid (Tables IV-VII, C1-C12) for one seed.

For every configuration (row) of the experiment, the runner builds the reservoir
features of each requested reservoir type (classical, quantum ideal, quantum
noisy), fits each requested readout type (linear ridge, ideal or noisy fidelity
kernel ridge) on every readout instance, merges the instance predictions and
evaluates the cell on the evaluation and test splits. This is the paper's
3 x 3 cell grid per configuration; every cell is reported, and the paper's
"best of the quantum/hybrid cells" statistic is computed afterwards from
``metrics.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .architectures import CellSpec, build_cell_spec
from .baselines_classical import (
    esn_settings,
    fit_persistence,
    fit_rbf_kernel_ridge,
    select_first_within_tolerance,
)
from .data import ForecastData, build_forecast_data
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .metrics import forecast_metrics, rmse
from .noise import IDEAL, device_from_config, devices_for_components
from .photonic_merlin import (
    PhotonicKernelMap,
    PhotonicReservoir,
    fit_photonic_kernel_ridge,
)
from .readouts import (
    FittedReadout,
    fit_kernel_ridge,
    fit_linear_ridge,
    merge_predictions,
    partition_features,
)
from .reservoirs import (
    ClassicalReservoir,
    QuantumReservoir,
    noise_for,
    sample_quantum_reservoir,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESERVOIR_TYPES = (
    # From the paper:
    "classical",  # ReservoirPy reservoir, library defaults (paper baseline)
    "quantum_ideal",  # 4-qubit quantum neurons, ideal simulator
    "quantum_noisy",  # same circuits with the hardware-informed noise model
    # MerLin adaptation:
    "photonic",  # one frozen 4-mode, 2-photon chip per neuron
    # Additional classical controls (not the paper's method):
    "esn_tuned",  # ReservoirPy ESN with the paper baseline's matrices, hyperparameters tuned on eval
    "raw_lags",  # no reservoir: the readout sees the 4-lag window directly
)
# Seed stream that separates the photonic chips from the gate-model randomness.
PHOTONIC_STREAM = 7
READOUT_TYPES = (
    # From the paper:
    "linear",  # classical ridge (Eq. 5; intercept added by default, see --linear-intercept)
    "quantum_kernel_ideal",  # "quantum ridge": fidelity-kernel ridge, ideal simulator
    "quantum_kernel_noisy",  # "quantum ridge" with the noise model
    # MerLin adaptation:
    "photonic_kernel",  # 2-photon fidelity-kernel ridge
    # Additional classical controls (not the paper's method):
    "rbf_kernel",  # classical RBF kernel ridge, gamma and lambda tuned on eval
    "persistence",  # forecast = newest observed hour (raw_lags only)
)
SPLITS = ("train", "eval", "test")
_LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in cfg:
        raise KeyError(f"Missing required config section '{name}'")
    return cfg[name]


def _child_seed(seed: int, *path: int) -> np.random.SeedSequence:
    # Reservoir structure depends only on (seed, configuration, sub-reservoir), so
    # the ideal and noisy versions of a quantum reservoir share the same circuit.
    return np.random.SeedSequence([seed, *path])


def sub_reservoir_features(
    cfg: dict[str, Any],
    data: ForecastData,
    n_neurons: int,
    cell_index: int,
    sub: int,
    reservoir_type: str,
    device_name: str | None,
    classical_params: dict[str, Any] | None = None,
) -> list[np.ndarray]:
    """Features of one sub-reservoir for every split, in ``SPLITS`` order.

    One sub-reservoir is one branch of a distributed architecture: it depends only
    on the seed, the configuration index and its own index, so it can be computed
    on its own process, node or QPU (see :mod:`lib.distributed`).
    """

    seed = int(cfg["seed"])
    lags = int(cfg["dataset"]["lags"])
    sequence = _child_seed(seed, cell_index, sub)
    if reservoir_type == "classical":
        params = classical_params or _section(cfg, "classical_reservoir")
        node_seed = int(sequence.generate_state(1)[0])
        outputs = ClassicalReservoir(n_neurons, params, node_seed).transform_sequence(
            [data.inputs[split] for split in SPLITS]
        )
    elif reservoir_type in ("quantum_ideal", "quantum_noisy"):
        qcfg = _section(cfg, "quantum_reservoir")
        structure_rng, shot_rng = (np.random.default_rng(s) for s in sequence.spawn(2))
        reservoir_spec = sample_quantum_reservoir(n_neurons, lags, structure_rng)
        device = device_from_config(
            device_name, cfg["noise"]["devices"], float(cfg["noise"]["scale"])
        )
        # Finite shots model the noisy (hardware-informed) simulator only; the
        # paper's ideal "Sim." columns match exact expectations.
        shots = qcfg["shots"] if reservoir_type == "quantum_noisy" else None
        reservoir = QuantumReservoir(
            reservoir_spec,
            noise_for(reservoir_type, device),
            observable=qcfg["observable"],
            passes=int(qcfg["passes"]),
            update=qcfg["update"],
            shots=shots,
            rng=shot_rng if shots is not None else None,
        )
        outputs = [reservoir.transform(data.inputs[split]) for split in SPLITS]
    elif reservoir_type == "photonic":
        # MerLin adaptation: same wiring as the gate reservoir of
        # this seed, photonic neuron chips with their own fixed interferometers.
        pcfg = _section(cfg, "photonic")
        structure_rng = np.random.default_rng(sequence.spawn(2)[0])
        reservoir_spec = sample_quantum_reservoir(n_neurons, lags, structure_rng)
        reservoir = PhotonicReservoir(
            reservoir_spec,
            scale=float(pcfg["neuron_scale"]),
            passes=int(_section(cfg, "quantum_reservoir")["passes"]),
            sequence=_child_seed(seed, cell_index, sub, PHOTONIC_STREAM),
        )
        outputs = [reservoir.transform(data.inputs[split]) for split in SPLITS]
    else:
        raise ValueError(
            f"reservoir type must be one of {RESERVOIR_TYPES}, got {reservoir_type!r}"
        )
    return outputs


def _reservoir_features(
    cfg: dict[str, Any],
    data: ForecastData,
    spec: CellSpec,
    cell_index: int,
    reservoir_type: str,
    devices: list[str],
    classical_params: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    if reservoir_type == "raw_lags":
        # No-reservoir control: the readout sees the 4-lag window itself.
        return {split: data.inputs[split] for split in SPLITS}
    per_split: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    for sub, n_neurons in enumerate(spec.reservoir_sizes):
        outputs = sub_reservoir_features(
            cfg,
            data,
            n_neurons,
            cell_index,
            sub,
            reservoir_type,
            # Only quantum reservoirs carry a device noise profile.
            devices[sub]
            if reservoir_type in ("quantum_ideal", "quantum_noisy")
            else None,
            classical_params,
        )
        for split, values in zip(SPLITS, outputs):
            per_split[split].append(values)
    return {split: np.concatenate(parts, axis=1) for split, parts in per_split.items()}


def fit_readout_instance(
    cfg: dict[str, Any],
    data: ForecastData,
    sub_features: dict[str, np.ndarray],
    readout_type: str,
    kernel_qubits: int,
    device_name: str,
    cell_tag: str,
    index: int,
) -> FittedReadout:
    """Fit one readout instance on its share of the reservoir features.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    data : ForecastData
        Windowed, normalized splits (targets are used).
    sub_features : dict
        Per-split features seen by this instance.
    readout_type : str
        One of :data:`READOUT_TYPES`.
    kernel_qubits : int
        Width of the kernel feature map (5 or 10 in the paper).
    device_name : str
        Simulated device of this instance, for the noisy kernel.
    cell_tag : str
        ``"<arch>:<config>|<reservoir>|<readout>"``, part of the shot and chip seeds.
    index : int
        Index of the instance in the cell, part of the same seeds.

    Returns
    -------
    FittedReadout
        Fitted readout with its predictions for every split.
    """

    rcfg = _section(cfg, "readout")
    if readout_type == "linear":
        return fit_linear_ridge(
            sub_features,
            data.targets,
            rcfg["lambda_grid"],
            intercept=bool(rcfg["linear_intercept"]),
        )
    elif readout_type in ("quantum_kernel_ideal", "quantum_kernel_noisy"):
        noise = (
            IDEAL
            if readout_type == "quantum_kernel_ideal"
            else device_from_config(
                device_name,
                cfg["noise"]["devices"],
                float(cfg["noise"]["scale"]),
            )
        )
        return fit_kernel_ridge(
            sub_features,
            data.targets,
            rcfg["lambda_grid"],
            n_qubits=kernel_qubits,
            noise=noise,
            device=str(cfg["device"]),
            selection_subsample=rcfg["kernel_selection_subsample"],
            fit_subsample=rcfg["kernel_fit_subsample"],
            shots=(
                _section(cfg, "kernel")["shots"]
                if readout_type == "quantum_kernel_noisy"
                else None
            ),
            shot_seed=int(
                np.random.SeedSequence(
                    [int(cfg["seed"]), zlib.crc32(cell_tag.encode()), index]
                ).generate_state(1)[0]
            ),
        )
    elif readout_type == "rbf_kernel":
        return fit_rbf_kernel_ridge(
            sub_features,
            data.targets,
            rcfg["lambda_grid"],
            _section(cfg, "classical_baselines")["rbf_gamma_factors"],
            device=str(cfg["device"]),
            selection_subsample=rcfg["kernel_selection_subsample"],
            fit_subsample=rcfg["kernel_fit_subsample"],
        )
    elif readout_type == "persistence":
        # The cell tag is "<arch>:<config>|<reservoir>|<readout>". Persistence
        # (no-reservoir control) predicts the newest lag, so it needs the raw window (raw_lags).
        if cell_tag.split("|")[1] != "raw_lags":
            raise ValueError(
                "persistence needs the raw_lags reservoir and one readout instance"
            )
        return fit_persistence(sub_features, data.targets)
    elif readout_type == "photonic_kernel":
        pcfg = _section(cfg, "photonic")
        kernel_map = PhotonicKernelMap(
            n_features=sub_features["train"].shape[1],
            n_modes=kernel_qubits,
            n_photons=int(pcfg["kernel_photons"]),
            scale=float(pcfg["kernel_scale"]),
            blocks=int(pcfg["kernel_blocks"]),
            sequence=np.random.SeedSequence(
                [
                    int(cfg["seed"]),
                    zlib.crc32(cell_tag.encode()),
                    index,
                    PHOTONIC_STREAM,
                ]
            ),
        )
        return fit_photonic_kernel_ridge(
            sub_features,
            data.targets,
            rcfg["lambda_grid"],
            kernel_map,
            device=str(cfg["device"]),
            selection_subsample=rcfg["kernel_selection_subsample"],
            fit_subsample=rcfg["kernel_fit_subsample"],
        )
    else:
        raise ValueError(
            f"readout type must be one of {READOUT_TYPES}, got {readout_type!r}"
        )


def _fit_cell(
    cfg: dict[str, Any],
    data: ForecastData,
    spec: CellSpec,
    features: dict[str, np.ndarray],
    readout_type: str,
    devices: list[str],
    cell_tag: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rcfg = _section(cfg, "readout")
    chunks = partition_features(features["train"].shape[1], spec.n_instances)
    if readout_type == "persistence" and len(chunks) != 1:
        raise ValueError(
            "persistence needs the raw_lags reservoir and one readout instance"
        )
    parts, instances = [], []
    for index, chunk in enumerate(chunks):
        sub_features = {split: values[:, chunk] for split, values in features.items()}
        started = time.perf_counter()
        fitted = fit_readout_instance(
            cfg,
            data,
            sub_features,
            readout_type,
            spec.kernel_qubits,
            devices[index],
            cell_tag,
            index,
        )
        seconds = time.perf_counter() - started
        logger.info(
            "TRAIN_READOUT_COMPLETED | cell=%s | instance=%d/%d | features=%d | lambda=%.3g | eval_rmse=%.6f | seconds=%.1f",
            cell_tag,
            index + 1,
            len(chunks),
            chunk.size,
            fitted.lambda_selected,
            fitted.selection_scores[fitted.lambda_selected],
            seconds,
        )
        parts.append(fitted.predictions)
        instances.append(
            {
                "features": [int(chunk[0]), int(chunk[-1]) + 1],
                "lambda_selected": fitted.lambda_selected,
                "selection_eval_rmse": {
                    f"{k:.3g}": v for k, v in fitted.selection_scores.items()
                },
                "seconds": round(seconds, 3),
                **fitted.details,
            }
        )
    merged = {
        split: merge_predictions([p[split] for p in parts], rcfg["merge"])
        for split in SPLITS
    }
    return merged, instances


def _tuned_esn_features(
    cfg: dict[str, Any],
    data: ForecastData,
    spec: CellSpec,
    cell_index: int,
    readout_devices: list[str],
    tag: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Tuned-ESN baseline: ReservoirPy reservoir tuned on the evaluation split.

    Every setting of ``classical_baselines.esn_grid`` reuses the paper baseline's random matrices
    (same node seeds) and is scored by the evaluation RMSE of the merged linear
    readout of this configuration; ties within 1 % go to the first setting.
    """

    base = _section(cfg, "classical_reservoir")
    settings = esn_settings(_section(cfg, "classical_baselines")["esn_grid"])
    scores, candidates = [], []
    for setting in settings:
        features = _reservoir_features(
            cfg, data, spec, cell_index, "classical", [], {**base, **setting}
        )
        merged, _ = _fit_cell(
            cfg, data, spec, features, "linear", readout_devices, f"{tag}|esn-candidate"
        )
        scores.append(rmse(data.targets["eval"], merged["eval"]))
        candidates.append(features)
        logger.info(
            "ESN_CANDIDATE | config=%s | sr=%g | lr=%g | input_scaling=%g | eval_rmse=%.6f",
            spec.label,
            setting["sr"],
            setting["lr"],
            setting["input_scaling"],
            scores[-1],
        )
    chosen = select_first_within_tolerance(scores)
    logger.info(
        "ESN_SELECTED | config=%s | setting=%s | eval_rmse=%.6f",
        spec.label,
        settings[chosen],
        scores[chosen],
    )
    return candidates[chosen], {
        "selected": settings[chosen],
        "candidates": [
            {**setting, "eval_rmse_linear": score}
            for setting, score in zip(settings, scores)
        ],
        "rule": "min eval RMSE of the linear readout; ties within 1 % -> first in grid order",
    }


def lib_sha256() -> str:
    """SHA-256 over this paper's ``lib/*.py`` sources.

    The paper folder is untracked, so the Git commit alone cannot tell code
    versions apart; this hash does.
    """

    digest = hashlib.sha256()
    for path in sorted((PROJECT_DIR / "lib").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _initial_status(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    experiment = cfg["experiment"]
    rcfg = cfg["readout"]
    identity = {
        "experiment": experiment["id"],
        "architecture": experiment["architecture"],
        "variant": experiment.get("variant", "original"),
    }
    # Inside a sweep the candidate is the coordinator's flag dict, so that the
    # run and sweep evidence agree; a standalone run uses the experiment identity.
    sweep_candidate = os.environ.get("DISTRIBUTED_QRC_SWEEP_CANDIDATE")
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "sweep_id": os.environ.get("DISTRIBUTED_QRC_SWEEP_ID"),
        "candidate": json.loads(sweep_candidate) if sweep_candidate else identity,
        "experiment": identity,
        "seed": int(cfg["seed"]),
        "repetition": int(cfg["seed"]),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": {**git_state(PROJECT_DIR), "lib_sha256": lib_sha256()},
        "dataset": {
            "name": cfg["dataset"]["name"],
            "splits": {
                "train": cfg["dataset"]["split_sizes"][0],
                "eval": cfg["dataset"]["split_sizes"][1],
                "test": cfg["dataset"]["split_sizes"][2],
                "window": [cfg["dataset"]["start"], cfg["dataset"]["end"]],
            },
            "preprocessing": {
                "lags": cfg["dataset"]["lags"],
                "normalization": cfg["dataset"]["normalization"],
                "artifact_handling": cfg["dataset"]["artifact_handling"],
            },
            "subset": {
                "kernel_selection_subsample": rcfg["kernel_selection_subsample"],
                "kernel_fit_subsample": rcfg["kernel_fit_subsample"],
            },
        },
        "selection": {
            "name": "ridge lambda per readout instance by eval RMSE (ties within 1% -> larger lambda)",
            "direction": "minimize",
            "split": "eval",
            "checkpoint": "closed-form refit at the selected lambda",
        },
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": None,
        "metrics_path": None,
        "metrics_sha256": None,
        "error": None,
    }


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Run every configuration and cell of ``cfg["experiment"]`` for ``cfg["seed"]``."""

    run_dir = Path(run_dir)
    level = _LOG_LEVELS.get(
        str(cfg.get("logging", {}).get("level", "info")).lower(), logging.INFO
    )
    configure_logging(run_dir / "run.log", level=level)
    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    status = _initial_status(cfg, run_dir)
    status["config_sha256"] = sha256(config_path)
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | experiment=%s | seed=%s",
        status["run_id"],
        cfg["experiment"]["id"],
        cfg["seed"],
    )

    try:
        experiment = _section(cfg, "experiment")
        architecture = experiment["architecture"]
        reservoir_types = list(experiment["reservoir_types"])
        readout_types = list(experiment["readout_types"])
        mapping = cfg["noise"]["mapping"]

        data = build_forecast_data(cfg)
        logger.info(
            "DATASET_READY | name=%s | windows train/eval/test=%d/%d/%d | normalization=%s | artifacts=%s",
            data.metadata["name"],
            *(data.inputs[s].shape[0] for s in SPLITS),
            data.metadata["preprocessing"]["normalization"],
            data.metadata["preprocessing"]["artifact_handling"],
        )

        specs = [build_cell_spec(architecture, size) for size in experiment["sizes"]]
        n_cells = len(specs) * len(reservoir_types) * len(readout_types)
        logger.info(
            "TRAINING_STARTED | architecture=%s | configurations=%d | cells=%d | device=%s",
            architecture,
            len(specs),
            n_cells,
            cfg["device"],
        )

        cells, predictions = (
            [],
            {"test_targets": data.targets["test"].astype(np.float32)},
        )
        done = 0
        for cell_index, spec in enumerate(specs):
            reservoir_devices = devices_for_components(
                len(spec.reservoir_sizes), mapping
            )
            readout_devices = devices_for_components(spec.n_instances, mapping)
            for reservoir_type in reservoir_types:
                started = time.perf_counter()
                reservoir_selection = None
                if reservoir_type == "esn_tuned":
                    # Tuned ESN: try the 12 settings, keep the best on eval (linear readout).
                    features, reservoir_selection = _tuned_esn_features(
                        cfg,
                        data,
                        spec,
                        cell_index,
                        readout_devices,
                        f"{architecture}:{spec.label}",
                    )
                else:
                    features = _reservoir_features(
                        cfg, data, spec, cell_index, reservoir_type, reservoir_devices
                    )
                logger.info(
                    "TRAIN_RESERVOIR_COMPLETED | config=%s | reservoir=%s | features=%d | seconds=%.1f",
                    spec.label,
                    reservoir_type,
                    features["train"].shape[1],
                    time.perf_counter() - started,
                )
                for readout_type in readout_types:
                    tag = f"{architecture}:{spec.label}|{reservoir_type}|{readout_type}"
                    merged, instances = _fit_cell(
                        cfg, data, spec, features, readout_type, readout_devices, tag
                    )
                    split_metrics = {
                        split: forecast_metrics(data.targets[split], merged[split])
                        for split in ("eval", "test")
                    }
                    test_mw = forecast_metrics(
                        data.denormalize("test", data.targets["test"]),
                        data.denormalize("test", merged["test"]),
                    )
                    done += 1
                    # Natural-unit progress for a sweep coordinator watching this run.
                    status["progress"] = {
                        "current": done,
                        "total": n_cells,
                        "unit": "cells",
                    }
                    write_json(run_dir / "run_status.json", status)
                    for split in ("eval", "test"):
                        m = split_metrics[split]
                        logger.info(
                            "EVALUATION_COMPLETED | cell=%s | %d/%d | split=%s | mae=%.5f | rmse=%.5f | r2=%.4f",
                            tag,
                            done,
                            n_cells,
                            split,
                            m["mae"],
                            m["rmse"],
                            m["r2"],
                        )
                    cells.append(
                        {
                            "architecture": architecture,
                            "configuration": spec.label,
                            "reservoir_sizes": list(spec.reservoir_sizes),
                            "total_neurons": spec.total_neurons,
                            "readout_instances": spec.n_instances,
                            "kernel_qubits": spec.kernel_qubits,
                            "reservoir_type": reservoir_type,
                            "readout_type": readout_type,
                            "reservoir_devices": reservoir_devices
                            if reservoir_type == "quantum_noisy"
                            else None,
                            "readout_devices": readout_devices
                            if readout_type == "quantum_kernel_noisy"
                            else None,
                            "metrics": {**split_metrics, "test_mw": test_mw},
                            "instances": instances,
                            **(
                                {"reservoir_selection": reservoir_selection}
                                if reservoir_selection is not None
                                else {}
                            ),
                        }
                    )
                    predictions[tag] = merged["test"].astype(np.float32)

        metrics_path = run_dir / "metrics.json"
        write_json(
            metrics_path,
            {
                "experiment": experiment["id"],
                "architecture": architecture,
                "variant": experiment.get("variant", "original"),
                "seed": int(cfg["seed"]),
                "metric_scale": f"normalized ({cfg['dataset']['normalization']} min-max); test_mw in MW",
                "dataset": data.metadata,
                "cells": cells,
            },
        )
        predictions_path = run_dir / "test_predictions.npz"
        np.savez_compressed(predictions_path, **predictions)
        logger.info("ARTIFACT_WRITTEN | path=%s", predictions_path.name)
        status.update(
            metrics_path=metrics_path.name, metrics_sha256=sha256(metrics_path)
        )
        write_json(run_dir / "run_status.json", status)
        logger.info(
            "METRICS_WRITTEN | path=%s | cells=%d", metrics_path.name, len(cells)
        )
        status.update(status="COMPLETED", completed_at=utc_now())
        write_json(run_dir / "run_status.json", status)
        logger.info("RUN_COMPLETED | run_id=%s", status["run_id"])
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise
