"""Runtime entry point for the QNN hyperparameter reproduction.

``cfg["mode"]`` selects what one invocation does:

- ``qnn``: train and evaluate one QNN configuration (one run directory);
- ``baseline``: one classical baseline (E-F);
- ``reanalysis``: E-A on the author's released results;
- ``photonic``: the Phase 4 MerLin adaptation of one ``qnn`` run
  (``lib/photonic.py``);
- ``sweep``: a detached, resumable coordinator over many ``qnn`` or
  ``baseline`` candidates (``lib/sweep.py``).

Every run writes ``run.log``, ``run_status.json``, ``config_snapshot.json``
and ``metrics.json`` with the logging-contract events (``RUN_STARTED``,
``DATASET_READY``, ``METRICS_WRITTEN``, ``RUN_COMPLETED`` / ``RUN_FAILED``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
MODES = ("qnn", "photonic", "baseline", "reanalysis", "sweep")


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Dispatch on ``cfg["mode"]`` (called by the shared runtime).

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    run_dir : pathlib.Path
        Timestamped run directory created by the shared runtime.
    """

    mode = cfg["mode"]
    run_dir = Path(run_dir)
    if mode == "sweep":
        from .sweep import run_sweep

        run_sweep(cfg, run_dir)
    elif mode in ("qnn", "photonic", "baseline", "reanalysis"):
        run_single(cfg, run_dir)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def _selection(cfg: dict) -> dict:
    return {
        "name": "accuracy",
        "direction": "maximize",
        "split": "val (250-row author validation sample)",
        "checkpoint": "final optimizer point"
        if cfg["mode"] in ("qnn", "photonic")
        else "n/a",
    }


def run_single(
    cfg: dict,
    run_dir: Path,
    *,
    sweep_id: str | None = None,
    candidate: dict | None = None,
    repetition: int = 0,
    console: bool = True,
) -> dict:
    """Run one ``qnn``, ``baseline`` or ``reanalysis`` job with full evidence.

    Parameters
    ----------
    cfg : dict
        Resolved configuration for this run.
    run_dir : pathlib.Path
        Directory for this run's evidence.
    sweep_id : str | None
        Owning sweep, if any. Default value is None.
    candidate : dict | None
        Candidate values when run inside a sweep. Default value is None.
    repetition : int
        Repetition index inside the sweep. Default value is 0.
    console : bool
        Mirror logs to stderr; sweep workers turn this off. Default value is True.

    Returns
    -------
    dict
        The metrics written to ``metrics.json``.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir / "run.log", console=console)
    config_path = run_dir / "config_snapshot.json"
    # The shared runtime resolves ``dtype`` to a dtype object; store its name.
    write_json(config_path, json.loads(json.dumps(cfg, default=str)))
    status = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "sweep_id": sweep_id,
        "candidate": candidate or {},
        "seed": cfg["seed"],
        "repetition": repetition,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": {
            "name": cfg.get("dataset", {}).get("name"),
            "splits": {},
            "preprocessing": None,
            "subset": None,
        },
        "selection": _selection(cfg),
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(config_path),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | mode=%s | seed=%s | sweep_id=%s",
        run_dir.name,
        cfg["mode"],
        cfg["seed"],
        sweep_id,
    )
    try:
        if cfg["mode"] == "qnn":
            metrics = _run_qnn(cfg, run_dir, status)
        elif cfg["mode"] == "photonic":
            metrics = _run_photonic(cfg, run_dir, status)
        elif cfg["mode"] == "baseline":
            metrics = _run_baseline(cfg, run_dir, status)
        elif cfg["mode"] == "reanalysis":
            metrics = _run_reanalysis(cfg, run_dir, status)
        else:
            raise ValueError(f"run_single cannot run mode {cfg['mode']!r}")
        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, metrics)
        logger.info("METRICS_WRITTEN | path=metrics.json")
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(metrics_path),
        )
        write_json(run_dir / "run_status.json", status)
        logger.info("RUN_COMPLETED | run_id=%s", run_dir.name)
        return metrics
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", run_dir.name)
        raise


def _dataset_ready(status: dict, run_dir: Path, data) -> None:
    status["dataset"] = data.identity
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | dataset=%s | preprocessing=%s | splits=%s",
        data.identity["name"],
        data.identity["preprocessing"],
        data.identity["splits"],
    )


def expected_evaluations(training_cfg: dict) -> int:
    """Upper bound on loss evaluations, used as the progress denominator.

    COBYLA: ``maxiter``. SPSA: 50 calibration evaluations, two per iteration,
    one final (2 * maxiter + 51; the released Rice runs show 351 for 150).
    Nelder-Mead: ``maxfev``. Adam (photonic runs): ``steps``.
    """

    name = training_cfg["optimizer"]
    if name == "Adam":
        return int(training_cfg["steps"])
    if name == "COBYLA":
        return int(training_cfg["maxiter"])
    if name == "SPSA":
        return 2 * int(training_cfg["maxiter"]) + 51
    return int(training_cfg["maxfev"])


def _run_qnn(cfg: dict, run_dir: Path, status: dict) -> dict:
    from qiskit_machine_learning.utils import algorithm_globals

    from .data import prepare_data
    from .qnn import QNNClassifier

    if not str(cfg["device"]).startswith("cpu"):
        logger.info(
            "DEVICE_NOTE | requested=%s | the numpy/Aer simulators run on CPU",
            cfg["device"],
        )
    data = prepare_data(cfg)
    _dataset_ready(status, run_dir, data)

    seed = int(cfg["seed"])
    # SPSA draws its perturbations from qiskit-ml's global generator.
    algorithm_globals.random_seed = seed
    rng = np.random.default_rng(seed)
    noise_model = cfg["noise"]["model"] if cfg["noise"]["enabled"] else None
    model = QNNClassifier(
        cfg["model"],
        data.n_features,
        cfg["training"]["shots"],
        rng,
        noise_model=noise_model,
    )
    total = expected_evaluations(cfg["training"])
    progress_path = run_dir / "progress.json"

    def report_progress(evaluations: int) -> None:
        if evaluations % 10 == 0 or evaluations == 1:
            write_json(
                progress_path,
                {"current": evaluations, "total": total, "unit": "loss_evaluations"},
            )

    model.progress_hook = report_progress
    logger.info(
        "TRAINING_STARTED | optimizer=%s | maxiter=%s | init=%s | qubits=%d | weights=%d | shots=%s | noise=%s",
        cfg["training"]["optimizer"],
        cfg["training"]["maxiter"],
        cfg["model"]["init"],
        data.n_features,
        model.num_weights,
        cfg["training"]["shots"],
        noise_model,
    )
    result = model.fit(data.X["train"], data.y["train"], cfg["training"])
    write_json(
        progress_path,
        {"current": result.evaluations, "total": total, "unit": "loss_evaluations"},
    )
    logger.info(
        "TRAINING_COMPLETED | evaluations=%d | final_loss=%.6f | seconds=%.1f",
        result.evaluations,
        result.losses[-1],
        result.fit_seconds,
    )
    (run_dir / "losses.json").write_text(json.dumps(result.losses), encoding="utf-8")

    metrics = {
        "n_qubits": data.n_features,
        "num_weights": model.num_weights,
        "evaluations": result.evaluations,
        "fit_seconds": result.fit_seconds,
        "final_loss": result.losses[-1],
        "initial_point": result.initial_point_details,
        "classes": [str(c) for c in result.classes],
        "losses_path": "losses.json",
    }
    for split in ("train", "val", "test"):
        logger.info("EVALUATION_STARTED | split=%s", split)
        scores = model.evaluate(data.X[split], data.y[split])
        metrics[split] = scores
        logger.info(
            "EVALUATION_COMPLETED | split=%s | accuracy=%.6f | f1_weighted=%.6f",
            split,
            scores["accuracy"],
            scores["f1_weighted"],
        )
    return metrics


def _run_photonic(cfg: dict, run_dir: Path, status: dict) -> dict:
    import merlin
    import torch

    from .data import prepare_data
    from .photonic import ENCODING_SCALE, PhotonicClassifier

    data = prepare_data(cfg)
    _dataset_ready(status, run_dir, data)

    seed = int(cfg["seed"])
    # Seeds MerLin's default phase draw and the readout head's init.
    torch.manual_seed(seed)
    model = PhotonicClassifier(
        cfg["model"],
        cfg["photonic"],
        data.n_features,
        cfg["device"],
        np.random.default_rng(seed),
    )
    total = expected_evaluations(cfg["training"])
    progress_path = run_dir / "progress.json"

    def report_progress(steps: int) -> None:
        if steps % 10 == 0 or steps == 1:
            write_json(
                progress_path, {"current": steps, "total": total, "unit": "steps"}
            )

    model.progress_hook = report_progress
    # Hardware-aware fields required for every MerLin result (Phase 4, step 11).
    hardware = {
        "computation_space": "UNBUNCHED",
        "detector": "threshold",
        "n_photons": model.n_photons,
        "n_modes": model.n_modes,
        "input_state": model.input_state,
        "encoding": f"angle, phase {ENCODING_SCALE:.6f} * x_i on mode 2i, "
        f"reuploaded {model.depth} time(s) between trainable meshes",
        "measurement": "MeasurementStrategy.mode_expectations(UNBUNCHED) + linear head",
        "postselection": "none",
        "simulator": f"MerLin {merlin.__version__} analytic simulator on {cfg['device']}",
        "shots": "none (exact expectations)",
        "gradients": "exact autograd",
    }
    logger.info(
        "TRAINING_STARTED | optimizer=%s | steps=%s | lr=%s | init=%s | modes=%d | photons=%d | input_state=%s | depth=%d | phases=%d | device=%s",
        cfg["training"]["optimizer"],
        cfg["training"]["steps"],
        cfg["training"]["lr"],
        cfg["model"]["init"],
        model.n_modes,
        model.n_photons,
        model.input_state,
        model.depth,
        model.num_phases,
        cfg["device"],
    )
    result = model.fit(data.X["train"], data.y["train"], cfg["training"])
    write_json(
        progress_path, {"current": result.evaluations, "total": total, "unit": "steps"}
    )
    logger.info(
        "TRAINING_COMPLETED | evaluations=%d | final_loss=%.6f | seconds=%.1f",
        result.evaluations,
        result.losses[-1],
        result.fit_seconds,
    )
    (run_dir / "losses.json").write_text(json.dumps(result.losses), encoding="utf-8")

    metrics = {
        "n_qubits": data.n_features,
        "num_weights": model.num_weights,
        "num_phases": model.num_phases,
        "evaluations": result.evaluations,
        "fit_seconds": result.fit_seconds,
        "final_loss": result.losses[-1],
        "initial_point": result.initial_point_details,
        "classes": [str(c) for c in result.classes],
        "losses_path": "losses.json",
        "hardware": hardware,
    }
    for split in ("train", "val", "test"):
        logger.info("EVALUATION_STARTED | split=%s", split)
        scores = model.evaluate(data.X[split], data.y[split])
        metrics[split] = scores
        logger.info(
            "EVALUATION_COMPLETED | split=%s | accuracy=%.6f | f1_weighted=%.6f",
            split,
            scores["accuracy"],
            scores["f1_weighted"],
        )
    return metrics


def _run_baseline(cfg: dict, run_dir: Path, status: dict) -> dict:
    from .baselines import fit_and_evaluate
    from .data import prepare_data

    data = prepare_data(cfg)
    _dataset_ready(status, run_dir, data)
    baseline = cfg["baseline"]
    logger.info(
        "TRAINING_STARTED | baseline=%s | cv_folds=%d",
        baseline["model"],
        baseline["cv_folds"],
    )
    scores = fit_and_evaluate(
        baseline["model"],
        data.X,
        data.y,
        int(baseline["mlp_parameter_target"]),
        int(cfg["seed"]),
        int(baseline["cv_folds"]),
    )
    logger.info(
        "TRAINING_COMPLETED | baseline=%s | parameters=%d",
        baseline["model"],
        scores["parameters"],
    )
    return {"model": baseline["model"], "n_features": data.n_features, **scores}


def _run_reanalysis(cfg: dict, run_dir: Path, status: dict) -> dict:
    from . import reanalysis
    from .data import resolve_paper_data_dir

    results_dir = resolve_paper_data_dir(cfg) / cfg["reanalysis"]["results_subdir"]
    results = reanalysis.load_released_results(results_dir)
    status["dataset"] = {
        "name": "author released results",
        "splits": {"rows": int(len(results))},
        "preprocessing": "none",
        "subset": f"{results['source_file'].nunique()} CSV files from {cfg['reanalysis']['results_subdir']}",
    }
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | files=%d | rows=%d",
        results["source_file"].nunique(),
        len(results),
    )
    results.to_csv(run_dir / "released_results_tidy.csv", index=False)

    logger.info("EVALUATION_STARTED | analysis=E-A1 tables")
    top = reanalysis.top_configurations(results)
    top.to_csv(run_dir / "top5_by_f1.csv", index=False)
    ranges = reanalysis.accuracy_ranges(results)
    ranges.to_csv(run_dir / "accuracy_ranges.csv", index=False)
    noise = reanalysis.noise_differences(results)
    noise.to_csv(run_dir / "noise_differences.csv", index=False)
    logger.info("EVALUATION_COMPLETED | analysis=E-A1 tables | top5_rows=%d", len(top))

    logger.info("EVALUATION_STARTED | analysis=E-A2 significance tests")
    main = results[results["init"].isin(["uniform", "normal", "beta"])]
    tests = {
        "init": reanalysis.paired_factor_tests(
            main, "init", ["beta", "uniform", "normal"], better="beta"
        ),
        "optimizer": reanalysis.paired_factor_tests(
            main, "optimizer", ["cobyla", "spsa", "neldermead"], better=None
        ),
        "ansatz": reanalysis.unpaired_factor_tests(results, "ansatz"),
        "entanglement": reanalysis.unpaired_factor_tests(results, "entanglement"),
        "entanglement_featuremap": reanalysis.unpaired_factor_tests(
            results, "entanglement_featuremap"
        ),
        "featuremap": reanalysis.unpaired_factor_tests(results, "featuremap"),
        "preprocessing": reanalysis.unpaired_factor_tests(results, "preprocessing"),
    }
    (run_dir / "significance_tests.json").write_text(
        json.dumps(tests, indent=2), encoding="utf-8"
    )
    logger.info(
        "EVALUATION_COMPLETED | analysis=E-A2 significance tests | families=%d",
        len(tests),
    )

    logger.info("EVALUATION_STARTED | analysis=E-A3 init equivalence")
    equivalence = reanalysis.init_equivalence(
        results, margin=float(cfg["reanalysis"]["equivalence_margin"])
    )
    (run_dir / "init_equivalence.json").write_text(
        json.dumps(equivalence, indent=2), encoding="utf-8"
    )
    true_files = reanalysis.resolve_true_files(results)
    (run_dir / "true_files_resolution.json").write_text(
        json.dumps(true_files, indent=2), encoding="utf-8"
    )
    spread = reanalysis.repetition_spread(results)
    (run_dir / "repetition_spread.json").write_text(
        json.dumps(spread, indent=2), encoding="utf-8"
    )
    within = [row for row in equivalence if row["same_input_scaling"]]
    across = [row for row in equivalence if not row["same_input_scaling"]]
    logger.info(
        "EVALUATION_COMPLETED | analysis=E-A3 | within_class_equivalent=%d/%d | across_class_equivalent=%d/%d",
        sum(r["tost_equivalent"] for r in within),
        len(within),
        sum(r["tost_equivalent"] for r in across),
        len(across),
    )
    return {
        "released_rows": int(len(results)),
        "released_files": int(results["source_file"].nunique()),
        "artifacts": [
            "released_results_tidy.csv",
            "top5_by_f1.csv",
            "accuracy_ranges.csv",
            "noise_differences.csv",
            "significance_tests.json",
            "init_equivalence.json",
            "true_files_resolution.json",
            "repetition_spread.json",
        ],
        "init_equivalence_summary": {
            "within_class_pairs": len(within),
            "within_class_tost_equivalent": int(
                sum(r["tost_equivalent"] for r in within)
            ),
            "within_class_mean_abs_diff": float(
                np.mean([abs(r["mean_diff"]) for r in within])
            )
            if within
            else None,
            "across_class_pairs": len(across),
            "across_class_tost_equivalent": int(
                sum(r["tost_equivalent"] for r in across)
            ),
            "across_class_mean_abs_diff": float(
                np.mean([abs(r["mean_diff"]) for r in across])
            )
            if across
            else None,
        },
    }
