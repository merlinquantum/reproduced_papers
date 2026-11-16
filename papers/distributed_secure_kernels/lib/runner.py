"""Runtime entry point for the distributed secure kernel reproduction.

``cfg["experiment"]["kind"]`` selects one experiment:

- ``accuracy``: Table 1 rows, the noise sweep (Fig. 3) and the shots sweep
  (Fig. 4). Every method in ``cfg["methods"]`` is evaluated on the same
  samples and folds, under ``cfg["dataset"]["protocol"]`` (``paper`` or ``corrected``).
- ``protocol``: per-shot Aer runs of the authors' circuits, compared with the
  exact probabilities used by ``accuracy`` (App. A; security check on client bits).

Method kernels (``cfg["methods"][i]["kernel"]``):

- From the paper: ``linear`` (the paper's "centralised classical" column, a
  linear SVM on standardised features) and ``swap_test`` (finite-shot
  swap-test estimate; ``circuit`` = ``centralised`` or ``distributed``,
  ``noise_level`` 0/1/2, ``shots``).
- Additional references (not the paper's method): ``squared_overlap`` (exact
  swap-test kernel, infinite shots, no noise), ``cosine`` (the true "linear"
  kernel of amplitude-encoded data), ``rbf`` and ``poly`` (tuned classical
  kernels), ``majority`` (most-frequent-class predictor).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

from .data import build_dataset
from .encoding import amplitude_encode
from .evaluation import (
    _fold_scores,
    corrected_cv,
    paper_kernel_pca,
    paper_kernel_svm,
    stratified_folds,
    summarize,
)
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .kernels import (
    cosine_kernel,
    estimate_from_zero_probability,
    squared_overlap_kernel,
)
from .noise import noisy_zero_probability, rate_for_level

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parents[1]
EXPERIMENT_KINDS = ("accuracy", "protocol")
KERNELS = (
    "linear",
    "swap_test",
    "photonic_swap_test",
    "squared_overlap",
    "cosine",
    "rbf",
    "poly",
    "majority",
)
_LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in cfg:
        raise KeyError(f"Missing required config section '{name}'")
    return cfg[name]


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #


def swap_test_kernel_fn(method: dict[str, Any], rng: np.random.Generator):
    """Finite-shot swap-test kernel: exact ``p0`` (noisy or not) then binomial shots."""

    shots = int(method["shots"])
    rate = rate_for_level(int(method["noise_level"]))
    circuit = method["circuit"]
    if circuit not in ("centralised", "distributed"):
        raise ValueError(
            f"circuit must be 'centralised' or 'distributed', got {circuit!r}"
        )

    def kernel(left: np.ndarray, right: np.ndarray, symmetric: bool) -> np.ndarray:
        # Rows play Alice (first argument of the authors' teleport_and_inner_product).
        left_states, right_states = amplitude_encode(left), amplitude_encode(right)
        if rate == 0:
            probability = 0.5 + 0.5 * (left_states @ right_states.T) ** 2
        else:
            probability = noisy_zero_probability(
                left_states,
                right_states,
                rate,
                teleport=circuit == "distributed",
                # Diagnostic variant: state preparation decomposed and noisy.
                noisy_initialize=bool(method.get("noisy_initialize", False)),
            )
        return estimate_from_zero_probability(probability, shots, rng, symmetric)

    return kernel


def photonic_swap_test_kernel_fn(method: dict[str, Any], rng: np.random.Generator):
    """MerLin adaptation: P(even) from the photonic swap test, then binomial shots.

    Every call checks the photonic probability against 1/2 + 1/2 |<psi|phi>|^2
    (the adaptation must reproduce the swap-test law exactly) and logs the
    largest deviation.
    """

    from .photonic_merlin import PhotonicSwapTest, encode_registers, register_modes

    shots = int(method["shots"])
    tolerance = float(method["exactness_tolerance"])
    tests: dict[int, PhotonicSwapTest] = {}

    def kernel(left: np.ndarray, right: np.ndarray, symmetric: bool) -> np.ndarray:
        modes = register_modes(left.shape[1])
        if modes not in tests:
            tests[modes] = PhotonicSwapTest(modes, device=method["device"])
        left_states = encode_registers(left, modes)
        right_states = encode_registers(right, modes)
        probability = tests[modes].zero_probability_matrix(
            left_states, right_states, symmetric
        )
        deviation = float(
            np.max(
                np.abs(probability - (0.5 + 0.5 * (left_states @ right_states.T) ** 2))
            )
        )
        logger.info(
            "PHOTONIC_KERNEL_CHECK | modes_per_register=%d | photons=4 | pairs=%d | max_abs_deviation=%.3e",
            modes,
            probability.size,
            deviation,
        )
        if deviation > tolerance:
            raise RuntimeError(
                f"Photonic swap test deviates from the swap-test law by {deviation:.3e}"
            )
        return estimate_from_zero_probability(probability, shots, rng, symmetric)

    return kernel


def kernel_fn_for(
    method: dict[str, Any], point: dict[str, Any], rng: np.random.Generator
):
    """Return ``kernel(left, right, symmetric)`` for one method and hyperparameter point."""

    kind = method["kernel"]
    if kind == "linear":
        return lambda left, right, symmetric: left @ right.T
    if kind == "squared_overlap":
        return lambda left, right, symmetric: squared_overlap_kernel(left, right)
    if kind == "cosine":
        return lambda left, right, symmetric: cosine_kernel(left, right)
    if kind == "swap_test":
        return swap_test_kernel_fn(method, rng)
    if kind == "photonic_swap_test":
        return photonic_swap_test_kernel_fn(method, rng)
    if kind == "rbf":
        # gamma = multiplier / (n_features * Var(X_train)), i.e. sklearn's "scale" times a factor.
        def rbf(left, right, symmetric, _m=point["gamma_multiplier"]):
            return rbf_kernel(left, right, gamma=_m / (right.shape[1] * right.var()))

        return rbf
    if kind == "poly":

        def poly(left, right, symmetric, _p=point):
            gamma = 1.0 / (right.shape[1] * right.var())
            return polynomial_kernel(
                left, right, degree=_p["degree"], gamma=gamma, coef0=_p["coef0"]
            )

        return poly
    raise ValueError(f"kernel must be one of {KERNELS}, got {kind!r}")


def kernel_grid(method: dict[str, Any]) -> list[dict[str, Any]]:
    """Kernel hyperparameter points of a tuned method (``[{}]`` otherwise)."""

    if method["kernel"] == "rbf":
        return [{"gamma_multiplier": float(m)} for m in method["gamma_multipliers"]]
    if method["kernel"] == "poly":
        return [
            {"degree": int(d), "coef0": float(c)}
            for d in method["degrees"]
            for c in method["coef0s"]
        ]
    return [{}]


def _kernel_statistics(kernel: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    off_diagonal = ~np.eye(kernel.shape[0], dtype=bool)
    return {
        "mean_off_diagonal": float(kernel[off_diagonal].mean()),
        "variance": float(
            kernel.var()
        ),  # the authors print np.var of the kernel matrix
        "mean_abs_error_vs_exact": float(
            np.abs(kernel - reference)[off_diagonal].mean()
        ),
        "mean_diagonal": float(np.diag(kernel).mean()),
    }


# --------------------------------------------------------------------------- #
# accuracy experiment
# --------------------------------------------------------------------------- #


def _majority_paper(labels, folds):
    results = []
    for train, test in folds:
        model = DummyClassifier(strategy="most_frequent").fit(
            np.zeros((len(train), 1)), labels[train]
        )
        results.append(
            _fold_scores(labels[test], model.predict(np.zeros((len(test), 1))))
        )
    return {"kernel_svm": summarize(results)}


def _method_rng(cfg: dict[str, Any], index: int) -> np.random.Generator:
    # Stream 0 is data oversampling (lib.data); methods use streams (1, index).
    return np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), 1, index]))


def run_accuracy(
    cfg: dict[str, Any], status: dict[str, Any], run_dir: Path
) -> dict[str, Any]:
    """Evaluate every method of ``cfg["methods"]`` on one dataset and protocol."""

    data = build_dataset(cfg, "train")
    status["dataset"] = {
        "name": data.metadata["name"],
        "splits": {"train": data.metadata["shape"][0]},
        "preprocessing": data.metadata["preprocessing"],
        "subset": f"{data.metadata['protocol']} protocol, class counts {data.metadata['class_counts']}",
    }
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | name=%s | shape=%s | protocol=%s | classes=%s",
        data.metadata["name"],
        data.metadata["shape"],
        data.metadata["protocol"],
        data.metadata["class_counts"],
    )

    evaluation = _section(cfg, "evaluation")
    methods = _section(cfg, "methods")
    protocol = data.metadata["protocol"]
    folds = stratified_folds(
        data.labels, int(evaluation["n_splits"]), int(evaluation["cv_random_state"])
    )
    logger.info(
        "TRAINING_STARTED | methods=%d | protocol=%s | folds=%d | kernel_pca=%s",
        len(methods),
        protocol,
        len(folds),
        evaluation["kernel_pca"],
    )
    results: dict[str, Any] = {}
    kernel_stats: dict[str, Any] = {}
    exact = (
        squared_overlap_kernel(data.features, data.features)
        if protocol == "paper"
        else None
    )
    for index, method in enumerate(methods):
        if method["kernel"] == "photonic_swap_test":
            # The MerLin simulation runs on the configured torch device (--device).
            method = {**method, "device": cfg["device"]}
        name = method["name"]
        started = time.perf_counter()
        rng = _method_rng(cfg, index)
        if method["kernel"] == "majority":
            outcome = _majority_paper(data.labels, folds)
        elif protocol == "paper":
            if method["kernel"] in ("rbf", "poly"):
                raise ValueError(
                    "Tuned kernels need the corrected protocol (no tuning in the paper protocol)"
                )
            kernel = kernel_fn_for(method, {}, rng)(data.features, data.features, True)
            kernel_stats[name] = _kernel_statistics(kernel, exact)
            outcome = {"kernel_svm": paper_kernel_svm(kernel, data.labels, folds, name)}
            if evaluation["kernel_pca"]:
                outcome["kernel_pca"] = paper_kernel_pca(
                    kernel,
                    data.labels,
                    folds,
                    [int(c) for c in evaluation["kpca_components"]],
                    name,
                )
        else:
            outcome = corrected_cv(
                data.features,
                data.labels,
                kernel_grid(method),
                lambda point, generator, _m=method: kernel_fn_for(_m, point, generator),
                evaluation,
                cfg["dataset"]["oversample_ratio"],
                cfg["dataset"]["scaling"] == "standard",
                rng,
                name,
            )
        outcome["method"] = method
        outcome["seconds"] = time.perf_counter() - started
        results[name] = outcome
        svm = outcome["kernel_svm"]["accuracy"]
        message = f"kernel_svm_accuracy={svm['mean']:.4f}+-{svm['std']:.4f}"
        if "kernel_pca" in outcome:
            pca = outcome["kernel_pca"]
            pca_acc = (
                pca["best_on_test_folds"] if "best_on_test_folds" in pca else pca
            )["accuracy"]
            message += (
                f" | kernel_pca_accuracy={pca_acc['mean']:.4f}+-{pca_acc['std']:.4f}"
            )
        logger.info(
            "EVALUATION_COMPLETED | method=%s | %d/%d | %s | seconds=%.1f",
            name,
            index + 1,
            len(methods),
            message,
            outcome["seconds"],
        )
        status["progress"] = {
            "current": index + 1,
            "total": len(methods),
            "unit": "methods",
        }
        write_json(run_dir / "run_status.json", status)
    return {
        "dataset": data.metadata,
        "protocol": protocol,
        "methods": results,
        "kernel_statistics": kernel_stats,
    }


# --------------------------------------------------------------------------- #
# Other experiments
# --------------------------------------------------------------------------- #


def _no_dataset(
    status: dict[str, Any], run_dir: Path, name: str, description: str
) -> None:
    status["dataset"] = {
        "name": name,
        "splits": {},
        "preprocessing": [description],
        "subset": "none",
    }
    write_json(run_dir / "run_status.json", status)
    logger.info("DATASET_READY | name=%s | %s", name, description)


def run_protocol(cfg, status, run_dir):
    from .protocol_checks import check_protocol

    _no_dataset(
        status,
        run_dir,
        "random_states",
        "random amplitude vectors drawn from cfg['seed']",
    )
    return check_protocol(cfg)


RUNNERS = {
    "accuracy": run_accuracy,
    "protocol": run_protocol,
}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def paper_tree_sha256() -> str:
    """SHA-256 over the paper's ``lib/``, ``utils/``, ``configs/`` files and ``cli.json``."""

    import hashlib

    digest = hashlib.sha256()
    files = [PROJECT_DIR / "cli.json"]
    for folder in ("lib", "utils", "configs"):
        files += sorted(
            p for p in (PROJECT_DIR / folder).rglob("*") if p.suffix in (".py", ".json")
        )
    for path in files:
        digest.update(path.relative_to(PROJECT_DIR).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _initial_status(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    experiment = cfg["experiment"]
    return {
        "schema_version": 1,
        # Inside a sweep, utils/run_sweep.py passes the identity it records in
        # sweep_status.json (--run-id, --sweep-id, --candidate-config, --repetition).
        "run_id": cfg.get("run_id") or run_dir.name,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": cfg.get("candidate") or {"experiment": experiment["id"]},
        "seed": int(cfg["seed"]),
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        # The commit alone does not identify an uncommitted tree: also hash the
        # paper's code and configs (lib/, utils/, configs/, cli.json).
        "code": {**git_state(REPO_DIR), "paper_tree_sha256": paper_tree_sha256()},
        "dataset": {
            "name": cfg.get("dataset", {}).get("name", "none"),
            "splits": {},
            "preprocessing": [],
            "subset": "pending",
        },
        "selection": {
            "name": experiment.get("selection_metric", "cross-validated accuracy"),
            "direction": "maximize",
            "split": "stratified 5-fold CV of the evaluated subset",
            "checkpoint": "not applicable (closed-form kernel methods)",
        },
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": None,
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Run ``cfg["experiment"]`` and write ``metrics.json`` and ``run_status.json`` in ``run_dir``."""

    run_dir = Path(run_dir)
    level = _LOG_LEVELS.get(
        str(cfg.get("logging", {}).get("level", "info")).lower(), logging.INFO
    )
    configure_logging(run_dir / "run.log", level=level)
    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        import json

        config_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    status = _initial_status(cfg, run_dir)
    status["config_sha256"] = sha256(config_path)
    write_json(run_dir / "run_status.json", status)
    experiment = _section(cfg, "experiment")
    logger.info(
        "RUN_STARTED | run_id=%s | experiment=%s | kind=%s | seed=%s | device=%s",
        status["run_id"],
        experiment["id"],
        experiment["kind"],
        cfg["seed"],
        cfg["device"],
    )
    try:
        kind = experiment["kind"]
        if kind not in RUNNERS:
            raise ValueError(
                f"experiment.kind must be one of {EXPERIMENT_KINDS}, got {kind!r}"
            )
        metrics = RUNNERS[kind](cfg, status, run_dir)
        metrics = {"experiment": experiment, "seed": int(cfg["seed"]), **metrics}
        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, metrics)
        status["metrics_sha256"] = sha256(metrics_path)
        logger.info("METRICS_WRITTEN | path=%s", status["metrics_path"])
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
