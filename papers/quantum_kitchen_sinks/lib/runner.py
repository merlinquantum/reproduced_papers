"""Runtime entry point for the Quantum Kitchen Sinks reproduction."""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .classifiers import ClassifierResult, train_classifier
from .data import load_dataset


_LOGGER = logging.getLogger(__name__)


def _resolve_data_root(cfg: Dict[str, Any]) -> Path:
    data_root = cfg.get("data_root") or "data"
    return Path(data_root)


def _featurize(cfg: Dict[str, Any], X_train: np.ndarray, X_test: np.ndarray, seed: int):
    qks_cfg = cfg.get("qks", {})
    backend = qks_cfg.get("backend", "gate")
    if backend == "gate":
        from .qks_model import QKSFeaturizer

        feat = QKSFeaturizer(
            circuit=qks_cfg["circuit"],
            n_qubits=int(qks_cfg["n_qubits"]),
            n_episodes=int(qks_cfg["n_episodes"]),
            sigma=float(qks_cfg["sigma"]),
            encoding=qks_cfg.get("encoding", "split"),
            n_layers=int(qks_cfg.get("n_layers", 1)),
            shots_per_episode=int(qks_cfg.get("shots_per_episode", 1)),
        )
        feat.fit_episodes(input_dim=X_train.shape[1], seed=seed)
        t0 = time.perf_counter()
        Xf_train = feat.transform(X_train, seed=seed)
        Xf_test = feat.transform(X_test, seed=seed + 1)
        dt = time.perf_counter() - t0
        meta = {
            "backend": "gate",
            "n_features": int(Xf_train.shape[1]),
            "transform_seconds": float(dt),
        }
        return Xf_train, Xf_test, meta
    if backend == "photonic_merlin":
        from .photonic_qks import PhotonicQKSFeaturizer

        feat = PhotonicQKSFeaturizer(
            n_modes=int(qks_cfg.get("n_modes", 4)),
            n_photons=int(qks_cfg.get("n_photons", 2)),
            n_episodes=int(qks_cfg["n_episodes"]),
            sigma=float(qks_cfg["sigma"]),
            encoding=qks_cfg.get("encoding", "split"),
            n_layers=int(qks_cfg.get("n_layers", 1)),
            shots_per_episode=int(qks_cfg.get("shots_per_episode", 1)),
            input_modes=qks_cfg.get("input_modes"),
            angle_scale=float(qks_cfg.get("angle_scale", 1.0)),
            computation_space=qks_cfg.get("computation_space", "UNBUNCHED"),
        )
        feat.fit_episodes(input_dim=X_train.shape[1], seed=seed)
        t0 = time.perf_counter()
        Xf_train = feat.transform(X_train, seed=seed)
        Xf_test = feat.transform(X_test, seed=seed + 1)
        dt = time.perf_counter() - t0
        meta = {
            "backend": "photonic_merlin",
            "n_features": int(Xf_train.shape[1]),
            "transform_seconds": float(dt),
            "n_modes": feat.n_modes,
            "n_photons": feat.n_photons,
            "input_state": list(feat.input_state),
            "computation_space": feat.computation_space.name,
            "detector_model": "threshold",
            "measurement_strategy": "PROBABILITIES",
            "postselection": "none",
            "simulator": "MerLin CPU simulator (analytic)",
        }
        return Xf_train, Xf_test, meta
    if backend == "none":
        return X_train, X_test, {"backend": "none", "n_features": int(X_train.shape[1])}
    raise ValueError(f"Unknown qks.backend: {backend}")


def _classifier_run(
    cfg: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> ClassifierResult:
    clf_cfg = cfg.get("classifier", {})
    return train_classifier(
        name=clf_cfg.get("kind", "logistic_regression"),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cfg=clf_cfg,
        seed=seed,
    )


def _resolve_seed(cfg: Dict[str, Any]) -> int:
    seed = cfg.get("seed", 42)
    return int(seed) if seed is not None else 0


def _maybe_sigma_sweep(cfg: Dict[str, Any]) -> list[float] | None:
    sweep = cfg.get("sigma_sweep")
    if not sweep:
        return None
    if isinstance(sweep, (list, tuple)):
        return [float(s) for s in sweep]
    raise ValueError("sigma_sweep must be a list of floats")


def _maybe_e_sweep(cfg: Dict[str, Any]) -> list[int] | None:
    sweep = cfg.get("episodes_sweep")
    if not sweep:
        return None
    if isinstance(sweep, (list, tuple)):
        return [int(e) for e in sweep]
    raise ValueError("episodes_sweep must be a list of ints")


def _seed_list(cfg: Dict[str, Any]) -> list[int]:
    seeds_field = cfg.get("seeds")
    if seeds_field:
        return [int(s) for s in seeds_field]
    return [_resolve_seed(cfg)]


def train_and_evaluate(cfg: Dict[str, Any], run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = _resolve_data_root(cfg)

    seeds = _seed_list(cfg)
    sigma_sweep = _maybe_sigma_sweep(cfg)
    e_sweep = _maybe_e_sweep(cfg)
    base_qks = cfg.get("qks", {})

    X_train, y_train, X_test, y_test = load_dataset(cfg, data_root)
    _LOGGER.info(
        "Loaded dataset %s: X_train %s, X_test %s",
        cfg["dataset"]["name"],
        X_train.shape,
        X_test.shape,
    )

    sigma_values = sigma_sweep or [float(base_qks.get("sigma", 1.0))]
    e_values = e_sweep or [int(base_qks.get("n_episodes", 100))]

    results = []
    for sigma in sigma_values:
        for n_ep in e_values:
            for seed in seeds:
                run_cfg = copy.deepcopy(cfg)
                run_cfg["qks"]["sigma"] = sigma
                run_cfg["qks"]["n_episodes"] = n_ep
                run_cfg["seed"] = seed
                Xf_train, Xf_test, meta = _featurize(
                    run_cfg, X_train, X_test, seed=seed
                )
                clf_result = _classifier_run(
                    run_cfg, Xf_train, y_train, Xf_test, y_test, seed=seed
                )
                entry = {
                    "sigma": sigma,
                    "n_episodes": n_ep,
                    "seed": seed,
                    **asdict(clf_result),
                    **meta,
                }
                _LOGGER.info(
                    "sigma=%.3f E=%d seed=%d: train_acc=%.4f test_acc=%.4f",
                    sigma,
                    n_ep,
                    seed,
                    clf_result.train_accuracy,
                    clf_result.test_accuracy,
                )
                results.append(entry)

    metrics_path = run_dir / "metrics.json"
    summary = _summarise(results)
    metrics_path.write_text(
        json.dumps({"results": results, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    _LOGGER.info("Wrote %s", metrics_path)

    csv_path = run_dir / "results.csv"
    headers = sorted({k for r in results for k in r})
    lines = [",".join(headers)]
    for r in results:
        lines.append(",".join(str(r.get(h, "")) for h in headers))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _LOGGER.info("Wrote %s", csv_path)


def _summarise(results: list[dict]) -> dict:
    if not results:
        return {}
    best = max(results, key=lambda r: r["test_accuracy"])
    accs = np.array([r["test_accuracy"] for r in results])
    return {
        "best_test_accuracy": float(best["test_accuracy"]),
        "best_test_error": float(1.0 - best["test_accuracy"]),
        "best_config": {
            "sigma": best.get("sigma"),
            "n_episodes": best.get("n_episodes"),
            "seed": best.get("seed"),
        },
        "mean_test_accuracy": float(accs.mean()),
        "std_test_accuracy": float(accs.std()),
        "n_runs": int(len(results)),
    }


__all__ = ["train_and_evaluate"]
