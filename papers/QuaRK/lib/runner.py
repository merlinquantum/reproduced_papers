"""QuaRK reproduction runner: one model, one seed, all tasks.

A run builds the windows dataset, computes the features of one model (QuaRK, its
photonic MerLin adaptation, or a classical baseline), then for every task:

1. tunes the Matérn kernel on the training set (kernel models only);
2. sweeps ``reg`` over the paper's grid (Fig. 2) and records train, validation and
   test MSE;
3. selects ``reg`` twice: by validation MSE (our fair protocol) and by test MSE
   (the authors' rule for Fig. 4, kept only to reproduce the figure);
4. for each selection, sweeps the training-set size on a fixed test set (Fig. 4)
   and, for QuaRK, evaluates the generalization-bound terms (Thm 2).

Artifacts written to the run directory (paths relative to it):
``metrics.json``, ``run_status.json``, ``regularization_sweep.npz``,
``training_size_sweep.npz``, ``fit_examples.npz`` and optionally ``features.npz``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .bound import generalization_bound_terms
from .classical import esn_features, raw_window_features
from .data import check_against_release, generate_dataset
from .experiment_logging import git_state, sha256, utc_now, write_json
from .kernel_ridge import (
    decay_slope,
    kernel_regularization_sweep,
    linear_regularization_sweep,
    linear_ridge_fit_predict,
    solve_dual,
    standardize,
    tune_matern,
)
from .measurement import k_local_paulis, pauli_features
from .quark_merlin import photonic_features
from .reservoir import draw_reservoir_parameters, reservoir_states

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
SELECTIONS = ("validation", "test")


def _quark_features(windows, quark_cfg, seed):
    params = draw_reservoir_parameters(
        input_dim=windows.shape[-1],
        num_qubits=int(quark_cfg["num_qubits"]),
        num_reservoirs=int(quark_cfg["num_reservoirs"]),
        first_contraction=float(quark_cfg["first_contraction"]),
        seed=seed,
    )
    states = reservoir_states(windows, params)
    features = pauli_features(
        states,
        locality=int(quark_cfg["locality"]),
        estimator=quark_cfg["estimator"],
        shots=int(quark_cfg["shots"]),
        seed=seed,
    )
    info = {
        "contractions": params.contractions.tolist(),
        "num_observables": len(
            k_local_paulis(int(quark_cfg["locality"]), params.projection.shape[1])
        ),
    }
    return features, info


def _select_esn(windows, labels, train_idx, esn_cfg, reg_grid, tuning, seed, task):
    """Pick the ESN configuration with the lowest validation MSE under linear ridge."""
    best = None
    for spectral_radius in esn_cfg["spectral_radius"]:
        for leak in esn_cfg["leak"]:
            for input_scale in esn_cfg["input_scale"]:
                candidate = {
                    "spectral_radius": float(spectral_radius),
                    "leak": float(leak),
                    "input_scale": float(input_scale),
                }
                features = esn_features(
                    windows, units=int(esn_cfg["units"]), seed=seed, **candidate
                )
                (train,) = standardize(features[train_idx])
                sweep = linear_regularization_sweep(
                    train,
                    labels[train_idx],
                    train[:1],
                    labels[train_idx][:1],
                    reg_grid=reg_grid,
                    tuning=tuning,
                )
                candidate["val_mse"] = float(np.min(sweep["mse_val"]))
                logger.info(
                    "TRAIN_ESN_CANDIDATE_COMPLETED | task=%s | candidate=%s",
                    task,
                    candidate,
                )
                if best is None or candidate["val_mse"] < best["val_mse"]:
                    best = candidate
    return best


def build_features(cfg, dataset, task_index):
    """Features ``(N, D)`` of the configured model for one task, plus model information.

    Only the tuned ESN depends on the task (its configuration is selected per task).
    """
    model = cfg["model"]
    seed = int(cfg["seed"])
    kind = model["kind"]
    if kind == "quark":
        return _quark_features(dataset.windows, model["quark"], seed)
    if kind == "quark_merlin":
        return photonic_features(dataset.windows, model["merlin"], model["quark"], seed)
    if kind in ("raw_window_kernel", "raw_window_linear"):
        return raw_window_features(dataset.windows), {}
    if kind == "esn_kernel":
        esn = model["esn"]
        if esn["mode"] == "default":
            chosen = {
                k: float(esn["default"][k])
                for k in ("spectral_radius", "leak", "input_scale")
            }
        elif esn["mode"] == "tuned":
            chosen = _select_esn(
                dataset.windows,
                dataset.labels[task_index],
                dataset.train_idx,
                esn,
                cfg["readout"]["reg_grid"],
                cfg["readout"]["tuning"],
                seed,
                dataset.task_names[task_index],
            )
        else:
            raise ValueError(f"Unknown ESN mode {esn['mode']!r}")
        params = {k: chosen[k] for k in ("spectral_radius", "leak", "input_scale")}
        features = esn_features(
            dataset.windows, units=int(esn["units"]), seed=seed, **params
        )
        return features, {"esn": chosen}
    raise ValueError(f"Unknown model kind {kind!r}")


def _evaluate_task(cfg, dataset, features, task_index, model_info, artifacts):
    task = dataset.task_names[task_index]
    readout = cfg["readout"]
    reg_grid = np.asarray(readout["reg_grid"], dtype=float)
    tuning = readout["tuning"]
    labels = dataset.labels[task_index]
    # Training rows stay in the authors' order (their random permutation): the tuning
    # split and the regularization sweep index into it, as in their code. Fig. 4 uses
    # nested prefixes in time order, taken through ``chronological``.
    train_idx = dataset.train_idx
    chronological = np.argsort(train_idx)
    y_train, y_test = labels[train_idx], labels[dataset.test_idx]
    train, test = standardize(features[train_idx], features[dataset.test_idx])
    is_kernel = cfg["model"]["kind"] != "raw_window_linear"
    result = {"label_variance_test": float(np.var(y_test)), "selections": {}}

    if is_kernel:
        kernel = tune_matern(train, y_train, tuning, task=task)
        result["kernel"] = kernel
        sweep, gram, cross_test = kernel_regularization_sweep(
            train,
            y_train,
            test,
            y_test,
            xi=kernel["xi"],
            nu=kernel["nu"],
            reg_grid=reg_grid,
            tuning=tuning,
        )
        # Fig. 3: fit at the tuning ridge on the full training set, first windows shown.
        fit_train = gram @ solve_dual(gram, y_train, float(tuning["reg"]))
        shown = int(cfg["outputs"]["fit_examples"])
        artifacts[f"fit_examples/{task}/prediction"] = fit_train[:shown]
        artifacts[f"fit_examples/{task}/truth"] = y_train[:shown]
        result["fit_examples_mse"] = float(
            np.mean((fit_train[:shown] - y_train[:shown]) ** 2)
        )
    else:
        sweep = linear_regularization_sweep(
            train, y_train, test, y_test, reg_grid=reg_grid, tuning=tuning
        )
    for key, values in sweep.items():
        artifacts[f"regularization_sweep/{task}/{key}"] = values
    result["mse_train_at_smallest_reg"] = float(sweep["mse_train"][0])

    sizes = [n for n in readout["train_sizes"] if n <= len(train_idx)]
    for selection in SELECTIONS:
        best = int(
            np.argmin(sweep["mse_val" if selection == "validation" else "mse_test"])
        )
        reg = float(reg_grid[best])
        size_test, size_train = [], []
        for size in sizes:
            rows = chronological[:size]
            if is_kernel:
                block = gram[np.ix_(rows, rows)]
                weights = solve_dual(block, y_train[rows], reg)
                pred_train = block @ weights
                pred_test = cross_test[:, rows] @ weights
                del block
            else:
                pred_train, pred_test = linear_ridge_fit_predict(
                    train[rows], y_train[rows], test, reg
                )
            size_train.append(float(np.mean((pred_train - y_train[rows]) ** 2)))
            size_test.append(float(np.mean((pred_test - y_test) ** 2)))
        artifacts[f"training_size_sweep/{task}/{selection}/mse_test"] = np.array(
            size_test
        )
        artifacts[f"training_size_sweep/{task}/{selection}/mse_train"] = np.array(
            size_train
        )
        selected = {
            "reg": reg,
            "mse_train": float(sweep["mse_train"][best]),
            "mse_val": float(sweep["mse_val"][best]),
            "mse_test": float(sweep["mse_test"][best]),
            "normalized_mse_test": float(sweep["mse_test"][best] / np.var(y_test)),
            "training_sizes": sizes,
            "training_size_mse_test": size_test,
            "training_size_mse_train": size_train,
            "decay_slope": decay_slope(
                sizes, size_test, min_size=int(readout["slope_min_size"])
            )
            if sum(n >= int(readout["slope_min_size"]) for n in sizes) >= 2
            else None,
        }
        if cfg["model"]["kind"] == "quark":
            selected["bound"] = generalization_bound_terms(
                rkhs_norm=float(sweep["rkhs_norm"][best]),
                label_bound=float(np.max(np.abs(labels))),
                num_train=len(train_idx),
                delta=float(readout["bound_delta"]),
                nu=kernel["nu"],
                xi=kernel["xi"],
                num_features=features.shape[1],
                max_contraction=float(max(model_info["contractions"])),
                window=dataset.windows.shape[1],
            )
            selected["bound"]["measured_gap"] = (
                selected["mse_test"] - selected["mse_train"]
            )
        result["selections"][selection] = selected
        logger.info(
            "EVALUATION_COMPLETED | task=%s | selection=%s | reg=%g | mse_train=%.6g | "
            "mse_test=%.6g | normalized_mse_test=%.6g",
            task,
            selection,
            reg,
            selected["mse_train"],
            selected["mse_test"],
            selected["normalized_mse_test"],
        )
    artifacts["training_size_sweep/sizes"] = np.array(sizes)
    artifacts["regularization_sweep/reg_grid"] = reg_grid
    return result


def _write_artifacts(run_dir, artifacts):
    groups = {}
    for key, value in artifacts.items():
        group, _, name = key.partition("/")
        groups.setdefault(group, {})[name.replace("/", "__")] = value
    paths = {}
    for group, arrays in groups.items():
        path = run_dir / f"{group}.npz"
        np.savez_compressed(path, **arrays)
        paths[group] = path.name
        logger.info("ARTIFACT_WRITTEN | path=%s", path.name)
    return paths


def train_and_evaluate(cfg, run_dir: Path) -> None:
    """Run one configured model on all configured tasks and write the evidence files."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config_snapshot.json"
    status = {
        "schema_version": 1,
        "run_id": cfg.get("run_id", run_dir.name),
        "sweep_id": cfg.get("sweep_id"),
        "candidate": cfg.get("candidate", {"model": cfg["model"]["kind"]}),
        "seed": int(cfg["seed"]),
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(PROJECT_DIR),
        "dataset": {
            "name": cfg["data"]["name"],
            "splits": {
                "train": int(cfg["data"]["num_windows"])
                - int(
                    float(cfg["data"]["test_ratio"]) * int(cfg["data"]["num_windows"])
                ),
                "test": int(
                    float(cfg["data"]["test_ratio"]) * int(cfg["data"]["num_windows"])
                ),
                "validation": "inner 20% of train (tuning and reg selection)",
            },
            "preprocessing": "features standardized with training mean and std",
            "subset": f"num_windows={cfg['data']['num_windows']}",
        },
        "selection": {
            "name": "reg (ridge) and Matérn (xi, nu)",
            "direction": "minimize",
            "split": "validation (primary); test (paper protocol, reported separately)",
            "checkpoint": "closed-form kernel ridge (no checkpoints)",
        },
        "log_path": "run.log",
        "config_path": config_path.name,
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(run_dir / "run_status.json", status)
    try:
        logger.info(
            "RUN_STARTED | run_id=%s | model=%s | seed=%s",
            status["run_id"],
            cfg["model"]["kind"],
            cfg["seed"],
        )
        if cfg["device"] != "cpu":
            raise ValueError(
                f"This implementation is numpy-only; device must be 'cpu', got {cfg['device']!r}"
            )
        start = time.perf_counter()
        dataset = generate_dataset(cfg["data"], int(cfg["seed"]))
        data_check = None
        # The release holds one dataset: seed 12345 with 10,000 windows.
        if (
            cfg["data"]["check_release"]
            and int(cfg["seed"]) == int(cfg["data"]["release_seed"])
            and int(cfg["data"]["num_windows"])
            == int(cfg["data"]["release_num_windows"])
        ):
            data_check = check_against_release(
                dataset, cfg["data"], cfg.get("data_root")
            )
        logger.info(
            "DATASET_READY | windows=%s | train=%d | test=%d | release_check=%s",
            dataset.windows.shape,
            len(dataset.train_idx),
            len(dataset.test_idx),
            data_check,
        )
        tasks = [dataset.task_names.index(name) for name in cfg["data"]["tasks"]]
        logger.info(
            "TRAINING_STARTED | model=%s | tasks=%s",
            cfg["model"]["kind"],
            cfg["data"]["tasks"],
        )
        metrics = {
            "model": cfg["model"],
            "data_check": data_check,
            "dataset_meta": dataset.meta,
            "tasks": {},
        }
        artifacts = {}
        shared = None
        for task_index in tasks:
            task = dataset.task_names[task_index]
            task_start = time.perf_counter()
            per_task_features = (
                cfg["model"]["kind"] == "esn_kernel"
                and cfg["model"]["esn"]["mode"] == "tuned"
            )
            if shared is None or per_task_features:
                shared = build_features(cfg, dataset, task_index)
                logger.info(
                    "FEATURES_READY | task=%s | shape=%s | info=%s",
                    task,
                    shared[0].shape,
                    shared[1],
                )
                if cfg["outputs"].get("save_features"):
                    artifacts[f"features/{task}"] = shared[0].astype(np.float32)
            features, model_info = shared
            result = _evaluate_task(
                cfg, dataset, features, task_index, model_info, artifacts
            )
            result["model_info"] = model_info
            result["feature_dim"] = int(features.shape[1])
            result["seconds"] = time.perf_counter() - task_start
            metrics["tasks"][task] = result
            logger.info(
                "TRAIN_TASK_COMPLETED | task=%s | seconds=%.1f", task, result["seconds"]
            )
        metrics["artifacts"] = _write_artifacts(run_dir, artifacts)
        metrics["seconds"] = time.perf_counter() - start
        write_json(run_dir / "metrics.json", metrics)
        logger.info("METRICS_WRITTEN | path=metrics.json")
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(run_dir / "metrics.json"),
        )
        write_json(run_dir / "run_status.json", status)
        logger.info(
            "RUN_COMPLETED | run_id=%s | seconds=%.1f",
            status["run_id"],
            metrics["seconds"],
        )
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise
