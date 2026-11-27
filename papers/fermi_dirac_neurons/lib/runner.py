"""Shared-runtime entry point for the Fermi-Dirac neurons reproduction (arXiv:2605.24386).

One call of :func:`train_and_evaluate` runs one task and writes, in
``run_dir``::

    run.log               lifecycle, one line per epoch, evaluations
    run_status.json       structured status (experiment-logging contract)
    config_snapshot.json  resolved configuration (written by the runtime)
    metrics.json          final metrics of the run

plus, for ``experiment.task = "train"``::

    history.csv           training loss of every model before each update

Tasks:

* ``train``: Figs. 7-9 and Table II. Every model in ``cfg["models"]`` is
  trained on the same seeded data with full-batch gradient descent.
* ``theory_checks``: experiments E0a-E0d (identities, integral formulas,
  convolution identities, Monte Carlo of Algorithms 1, 2, 4, 5, 7).
* ``gradient_variance``, ``contextuality_scan``, ``pauli_compressibility``:
  extensions E10, E12, E15/E16 (``lib/extension_runs.py``).
* ``photonic_train``, ``photonic_gradient_variance``: Phase 4 MerLin
  adaptation; ``kerr_gradient_variance``: its non-MerLin Kerr ablation
  (``lib/photonic_runs.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .data import (
    STREAM_INITIALIZATION,
    STREAM_SHOTS,
    build_split,
    build_target,
    describe_dataset,
    random_stream,
    selection_split,
)
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .extension_runs import (
    run_contextuality_scan,
    run_gradient_variance,
    run_pauli_compressibility,
)
from .models import Dataset, build_model
from .photonic_runs import (
    run_kerr_gradient_variance,
    run_photonic_gradient_variance,
    run_photonic_training,
)
from .training import train_models

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[3]
TASKS = (
    "train",
    "theory_checks",
    "gradient_variance",
    "contextuality_scan",
    "pauli_compressibility",
    "photonic_train",
    "photonic_gradient_variance",
    "kerr_gradient_variance",
    "linear_reference",
)


def _check_device(cfg: dict) -> None:
    # All models are dense NumPy linear algebra in complex128; there is no GPU
    # path, so any other device request is an error rather than a silent CPU run.
    if cfg["device"] != "cpu":
        raise ValueError(
            f"device={cfg['device']!r} is not supported: this reproduction runs NumPy on the CPU only"
        )
    # The shared runtime resolves "dtype" into a DtypeSpec with a string label.
    dtype_label = getattr(cfg["dtype"], "label", cfg["dtype"])
    if dtype_label != "float64":
        raise ValueError(
            f"dtype must be float64 (complex128 matrices), got {dtype_label!r}"
        )


def describe_candidate(cfg: dict) -> dict:
    """Return the settings that identify one run's candidate.

    ``utils/run_sweep.py`` declares the same dictionary in ``sweep_status.json``
    so the sweep ledger and each ``run_status.json`` can be matched.
    """
    task = cfg["experiment"]["task"]
    if task == "theory_checks":
        return {"task": task, "experiment": cfg["experiment"]["id"]}
    if task == "gradient_variance":
        block = cfg["gradient_variance"]
        return {
            "task": task,
            "experiment": cfg["experiment"]["id"],
            "term_set": block["term_set"],
            "n_qubits": block["n_qubits"],
            "init_range": block["init_range"],
            "draws": block["draws"],
            "inputs": block["inputs"],
        }
    if task == "contextuality_scan":
        return {"task": task, "experiment": cfg["experiment"]["id"]}
    if task == "linear_reference":
        return {
            "task": task,
            "experiment": cfg["experiment"]["id"],
            "n_qubits": cfg["data"]["n_qubits"],
            "label_rule": cfg["data"]["label_rule"],
            "max_weights": cfg["linear_reference"]["max_weights"],
            "regularization": cfg["linear_reference"]["regularization"],
        }
    if task == "photonic_train":
        block = cfg["photonic"]
        return {
            "task": task,
            "experiment": cfg["experiment"]["id"],
            "n_modes": block["n_modes"],
            "n_photons": block["n_photons"],
            "train_states": block["train_states"]["set"],
            "models": [model["name"] for model in cfg["models"]],
            "loss": cfg["training"]["loss"],
            "learning_rate": cfg["training"]["learning_rate"],
            "epochs": cfg["training"]["epochs"],
            "temperature": cfg["physics"]["temperature"],
        }
    if task in ("photonic_gradient_variance", "kerr_gradient_variance"):
        block = cfg[task]
        return {
            "task": task,
            "experiment": cfg["experiment"]["id"],
            "n_modes": block["n_modes"],
            "n_photons": block["n_photons"],
            "draws": block["draws"],
            "inputs": block["inputs"],
        }
    if task == "pauli_compressibility":
        block = cfg["pauli_compressibility"]
        return {
            "task": task,
            "experiment": cfg["experiment"]["id"],
            "term_set": block["term_set"],
            "n_qubits": block["n_qubits"],
            "temperature": block["temperature"],
        }
    return {
        "task": task,
        "experiment": cfg["experiment"]["id"],
        "n_qubits": cfg["data"]["n_qubits"],
        "label_rule": cfg["data"]["label_rule"],
        "target_term_set": cfg["data"]["target"]["term_set"],
        "target_range": [cfg["data"]["target"]["low"], cfg["data"]["target"]["high"]],
        "train_states": cfg["data"]["train_states"]["set"],
        "models": [model["name"] for model in cfg["models"]],
        "loss": cfg["training"]["loss"],
        "learning_rate": cfg["training"]["learning_rate"],
        "epochs": cfg["training"]["epochs"],
        "temperature": cfg["physics"]["temperature"],
    }


def _selection(cfg: dict) -> dict:
    if cfg["experiment"]["task"] == "gradient_variance":
        return {
            "name": "variance of d_j f and d_j L over random parameters (E10; larger = more trainable; reported, no selection)",
            "direction": "maximize",
            "split": "n/a",
            "checkpoint": "n/a",
        }
    if cfg["experiment"]["task"] in (
        "photonic_gradient_variance",
        "kerr_gradient_variance",
    ):
        return {
            "name": "variance of d f (and d L) over random parameters (E13; larger = more trainable; reported, no selection)",
            "direction": "maximize",
            "split": "n/a",
            "checkpoint": "n/a",
        }
    if cfg["experiment"]["task"] == "linear_reference":
        return {
            "name": "B4 reference: selection-split metric per regularization strength (selected later by utils/select_p5.py)",
            "direction": "minimize",
            "split": "selection",
            "checkpoint": "n/a (closed-form / L-BFGS fit)",
        }
    if cfg["experiment"]["task"] == "photonic_train":
        return {
            "name": "final training squared loss (Fig. 7 metric, photonic counterpart)",
            "direction": "minimize",
            "split": "train",
            "checkpoint": "final epoch, no early stopping (as in the paper)",
        }
    if cfg["experiment"]["task"] == "pauli_compressibility":
        return {
            "name": "Pauli strings needed for error epsilon on every input (E15; fewer = cheaper; reported, no selection)",
            "direction": "minimize",
            "split": "n/a",
            "checkpoint": "n/a",
        }
    if cfg["experiment"]["task"] == "contextuality_scan":
        return {
            "name": "greedy contextual fraction vs T (E12; reported, no selection)",
            "direction": "minimize",
            "split": "n/a",
            "checkpoint": "n/a",
        }
    if cfg["experiment"]["task"] == "theory_checks":
        return {
            "name": "max_abs_error per identity",
            "direction": "minimize",
            "split": "n/a",
            "checkpoint": "n/a",
        }
    if "selection_fraction" in cfg["data"]:
        return {
            "name": "Phase 5 learning-rate selection: selection-split log10 MSE (regression) or own loss (classification), median over seeds 0-2 (utils/select_p5.py)",
            "direction": "minimize",
            "split": "selection",
            "checkpoint": "final epoch, no early stopping",
        }
    if cfg["data"]["label_rule"] == "regression":
        return {
            "name": "final training squared loss (paper metric, Figs. 7 and 9)",
            "direction": "minimize",
            "split": "train",
            "checkpoint": "final epoch, no early stopping (as in the paper)",
        }
    return {
        "name": "validation accuracy (Table II)",
        "direction": "maximize",
        "split": "validation",
        "checkpoint": "final epoch, no early stopping (as in the paper)",
    }


def run_training(cfg: dict, run_dir: Path) -> tuple[dict, dict]:
    """Train every configured model on one seeded dataset (experiments E1-E7)."""
    temperature = float(cfg["physics"]["temperature"])
    n_qubits = int(cfg["data"]["n_qubits"])
    target = build_target(cfg)
    splits = _qubit_splits(cfg, target)
    dataset_description = describe_dataset(cfg, target, splits)
    logger.info(
        "DATASET_READY | n_qubits=%d | train=%s n=%d | validation=%s n=%d | label_rule=%s | target=%s",
        n_qubits,
        cfg["data"]["train_states"]["set"],
        len(splits["train"].targets),
        cfg["data"]["validation_states"]["set"],
        len(splits["validation"].targets),
        cfg["data"]["label_rule"],
        cfg["data"]["target"]["term_set"],
    )
    models, initial = {}, {}
    for index, model_cfg in enumerate(cfg["models"]):
        name = model_cfg["name"]
        if name in models:
            raise ValueError(f"Duplicate model name {name!r}")
        model = build_model(model_cfg, n_qubits, temperature)
        rng = random_stream(cfg["seed"], STREAM_INITIALIZATION + index)
        low, high = cfg["training"]["init_low"], cfg["training"]["init_high"]
        initial[name] = rng.uniform(float(low), float(high), size=model.n_parameters)
        models[name] = model
        logger.info(
            "MODEL_READY | name=%s | kind=%s | term_set=%s | parameters=%d | activation=%s | commuting_terms=%s",
            name,
            model_cfg["kind"],
            model_cfg["term_set"],
            model.n_parameters,
            model_cfg["activation"],
            getattr(model, "commuting", "n/a"),
        )
    task_kind = (
        "regression" if cfg["data"]["label_rule"] == "regression" else "classification"
    )
    shot_streams = None
    if "finite_shots" in cfg["training"]:
        if cfg["training"]["loss"] != "squared":
            raise ValueError("finite_shots is implemented for the squared loss only")
        shot_streams = {
            name: random_stream(cfg["seed"], STREAM_SHOTS + index)
            for index, name in enumerate(models)
        }
    result = train_models(
        models,
        initial,
        splits["train"],
        splits,
        cfg["training"],
        task_kind,
        run_dir / "history.csv",
        shot_streams,
        learning_rates=_model_learning_rates(cfg),
    )
    metrics = {
        "task": "train",
        "experiment": cfg["experiment"]["id"],
        "task_kind": task_kind,
        "seed": cfg["seed"],
        "n_qubits": n_qubits,
        "temperature": temperature,
        "target": {"labels": target.labels, "zeta": target.zeta.tolist()},
        "models": {
            name: {
                "n_parameters": model.n_parameters,
                "term_labels": model.labels,
                "initial_parameters": initial[name].tolist(),
                "final_parameters": result["final_parameters"][name],
                "last_logged_train_loss": result["final_train_loss_before_last_update"][
                    name
                ],
                "final": result["final_metrics"][name],
                **training_record(result, name),
            }
            for name, model in models.items()
        },
        "optimizer": result["optimizer"],
        "evaluation_history": result["evaluations"],
        "artifacts": {"history": "history.csv"},
    }
    return metrics, dataset_description


def _qubit_splits(cfg: dict, target) -> dict:
    """Return train / validation splits, plus the Phase 5 selection split when configured."""
    splits = {
        split: build_split(cfg, split, target) for split in ("train", "validation")
    }
    if "selection_fraction" not in cfg["data"]:
        return splits
    # Phase 5: learning rates are selected on states held out of training.
    train_index, selection_index = selection_split(
        len(splits["train"].targets),
        float(cfg["data"]["selection_fraction"]),
        cfg["seed"],
    )
    full = splits["train"]
    return {
        "train": Dataset(full.states[train_index], full.targets[train_index]),
        "selection": Dataset(
            full.states[selection_index], full.targets[selection_index]
        ),
        "validation": splits["validation"],
    }


def run_linear_reference(cfg: dict, run_dir: Path) -> tuple[dict, dict]:
    """Baseline B4: ridge / logistic regression on Pauli expectations up to weight w.

    Raises
    ------
    ValueError
        If the config has no ``data.selection_fraction`` (B4 needs the
        selection split to choose its regularization).
    """
    from .linear_reference import evaluate_linear_reference, pauli_labels_up_to_weight
    from .pauli import pauli_string

    if "selection_fraction" not in cfg["data"]:
        raise ValueError("linear_reference needs data.selection_fraction")
    block = cfg["linear_reference"]
    n_qubits = int(cfg["data"]["n_qubits"])
    target = build_target(cfg)
    splits = _qubit_splits(cfg, target)
    task_kind = (
        "regression" if cfg["data"]["label_rule"] == "regression" else "classification"
    )
    logger.info(
        "DATASET_READY | n_qubits=%d | train=%d | selection=%d | test=%d | task=%s",
        n_qubits,
        len(splits["train"].targets),
        len(splits["selection"].targets),
        len(splits["validation"].targets),
        task_kind,
    )
    variants = {}
    for max_weight in block["max_weights"]:
        labels = pauli_labels_up_to_weight(n_qubits, int(max_weight))
        terms = np.array([pauli_string(label) for label in labels])
        name = f"b4_w{max_weight}"
        logger.info(
            "EVALUATION_STARTED | model=%s | features=%d | strengths=%s",
            name,
            len(labels),
            block["regularization"],
        )
        variants[name] = evaluate_linear_reference(
            splits,
            labels,
            terms,
            task_kind,
            [float(value) for value in block["regularization"]],
        )
        for record in variants[name]:
            logger.info(
                "EVALUATION_COMPLETED | model=%s | strength=%g | %s",
                name,
                record["strength"],
                " | ".join(
                    f"{key}={value:.10g}"
                    for key, value in record.items()
                    if key not in ("strength", "n_features")
                ),
            )
    metrics = {
        "task": "linear_reference",
        "experiment": cfg["experiment"]["id"],
        "task_kind": task_kind,
        "seed": cfg["seed"],
        "n_qubits": n_qubits,
        "variants": variants,
    }
    return metrics, describe_dataset(cfg, target, splits)


def _model_learning_rates(cfg: dict) -> dict:
    """Per-model learning rates set in ``cfg["models"]`` (Phase 5 tuned runs)."""
    return {
        model["name"]: float(model["learning_rate"])
        for model in cfg["models"]
        if "learning_rate" in model
    }


def training_record(result: dict, name: str) -> dict:
    """Per-model optimizer settings, divergence and timing from :func:`train_models`."""
    return {
        "learning_rate": result["learning_rates"][name],
        "diverged_at_epoch": result["diverged_at_epoch"][name],
        "training_seconds": result["training_seconds"][name],
    }


def run_theory_checks(cfg: dict, run_dir: Path) -> tuple[dict, dict]:
    """Run experiments E0a-E0d and return their error records."""
    from . import theory_checks as tc

    checks_cfg = cfg["checks"]
    temperature = float(cfg["physics"]["temperature"])
    rng = np.random.default_rng([int(cfg["seed"]), 100])
    logger.info(
        "DATASET_READY | random problems from seed %s (no dataset)", cfg["seed"]
    )
    records: dict = {
        "densities": {},
        "classical_reduction": {},
        "integral_formulas": {},
    }
    for name in ("mu", "gamma", "xi"):
        records["densities"][name] = tc.density_normalization(name)
        logger.info(
            "EVALUATION_COMPLETED | check=normalization_%s | value=%.15f",
            name,
            records["densities"][name],
        )
    for n_qubits in checks_cfg["classical_reduction_qubits"]:
        for activation in checks_cfg["activations"]:
            record = tc.check_classical_reduction(
                int(n_qubits), activation, temperature, rng
            )
            records["classical_reduction"][f"n{n_qubits}_{activation}"] = record
            logger.info(
                "EVALUATION_COMPLETED | check=classical_reduction | n=%s | activation=%s | max_abs_error=%.3e",
                n_qubits,
                activation,
                record["max_abs_error"],
            )
    for problem in checks_cfg["problems"]:
        theta, terms, rho = tc.random_problem(
            int(problem["n_qubits"]),
            problem["term_set"],
            bool(problem["include_identity"]),
            rng,
        )
        key = f"n{problem['n_qubits']}_{problem['term_set']}"
        logger.info("EVALUATION_STARTED | check=integral_formulas | problem=%s", key)
        problem_records = {
            "theorem1": tc.check_theorem1(theta, terms, rho, temperature),
            "theorem2": tc.check_theorem2(theta, terms, rho, temperature),
            "theorem5_y+1": tc.check_theorem5(theta, terms, rho, temperature, 1.0),
            "theorem5_y-1": tc.check_theorem5(theta, terms, rho, temperature, -1.0),
            "theorem6": tc.check_theorem6(theta, terms, rho, temperature),
            "theorem8_as_printed": tc.check_theorem8_as_printed(
                theta, terms, rho, temperature
            ),
            "corollary1": tc.check_corollary1(theta, terms),
        }
        for label in (1.0, -1.0):
            for variant, record in tc.check_theorem13(
                theta, terms, rho, temperature, label
            ).items():
                problem_records[f"theorem13_{variant}_y{label:+.0f}"] = record
        records["integral_formulas"][key] = problem_records
        for name, record in problem_records.items():
            logger.info(
                "EVALUATION_COMPLETED | check=%s | problem=%s | max_abs_error=%.3e",
                name,
                key,
                record["max_abs_error"],
            )
    conv_cfg = checks_cfg["convolution"]
    records["convolution"] = tc.check_convolution_identities(
        float(conv_cfg["t1"]),
        float(conv_cfg["t2"]),
        np.array(conv_cfg["points"], dtype=float),
    )
    for name, record in records["convolution"].items():
        logger.info(
            "EVALUATION_COMPLETED | check=%s | max_abs_error=%.3e",
            name,
            record["max_abs_error"],
        )
    monte_carlo_cfg = checks_cfg["monte_carlo"]
    theta, terms, rho = tc.random_problem(
        int(monte_carlo_cfg["n_qubits"]), monte_carlo_cfg["term_set"], True, rng
    )
    hamiltonian = np.einsum("j,jab->ab", theta, terms)
    records["single_shot"] = tc.simulate_single_shot_neurons(
        hamiltonian,
        rho,
        float(conv_cfg["t1"]),
        float(conv_cfg["t2"]),
        int(monte_carlo_cfg["single_shot_shots"]),
        rng,
    )
    for name, record in records["single_shot"].items():
        logger.info(
            "EVALUATION_COMPLETED | check=%s | estimate=%.6f | exact=%.6f | standard_error=%.2e",
            name,
            record["estimate"],
            record["exact"],
            record["standard_error"],
        )
    sampler = tc.TabulatedDensity("mu")
    exact_gradient = tc.exact_trace_gradient("tanh", temperature, theta, terms, rho)
    exact_objective = tc.exact_trace("tanh", temperature, hamiltonian, rho)
    records["sampling_algorithms"] = {
        "mu_table_normalization": sampler.normalization,
        "algorithm1": [],
        "algorithm2": [],
    }
    for shots in monte_carlo_cfg["shot_counts"]:
        samples = tc.simulate_algorithm1(
            theta, terms, rho, temperature, 0, int(shots), rng, sampler
        )
        records["sampling_algorithms"]["algorithm1"].append(
            {
                "shots": int(shots),
                "estimate": float(samples.mean()),
                "standard_error": float(samples.std(ddof=1) / np.sqrt(shots)),
                "exact": float(exact_gradient[0]),
            }
        )
        samples = tc.simulate_algorithm2(
            theta, terms, rho, temperature, int(shots), rng, sampler
        )
        records["sampling_algorithms"]["algorithm2"].append(
            {
                "shots": int(shots),
                "estimate": float(samples.mean()),
                "standard_error": float(samples.std(ddof=1) / np.sqrt(shots)),
                "exact": exact_objective,
            }
        )
        logger.info(
            "EVALUATION_COMPLETED | check=sampling_algorithms | shots=%d | alg1=%.5f (exact %.5f) | alg2=%.5f (exact %.5f)",
            shots,
            records["sampling_algorithms"]["algorithm1"][-1]["estimate"],
            exact_gradient[0],
            records["sampling_algorithms"]["algorithm2"][-1]["estimate"],
            exact_objective,
        )
    metrics = {
        "task": "theory_checks",
        "experiment": cfg["experiment"]["id"],
        "seed": cfg["seed"],
        **records,
    }
    dataset = {
        "name": "none (random test problems)",
        "splits": "n/a",
        "preprocessing": "none",
        "subset": "n/a",
    }
    return metrics, dataset


TASK_FUNCTIONS = {
    "train": run_training,
    "theory_checks": run_theory_checks,
    "gradient_variance": run_gradient_variance,
    "contextuality_scan": run_contextuality_scan,
    "pauli_compressibility": run_pauli_compressibility,
    "photonic_train": run_photonic_training,
    "photonic_gradient_variance": run_photonic_gradient_variance,
    "kerr_gradient_variance": run_kerr_gradient_variance,
    "linear_reference": run_linear_reference,
}


def _ensure_config_snapshot(cfg: dict, run_dir: Path) -> Path:
    """Return the runtime's snapshot, writing one when called outside the runtime."""
    path = run_dir / "config_snapshot.json"
    if not path.exists():
        path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    return path


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run the task selected by ``cfg["experiment"]["task"]``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration after defaults, overlays and CLI flags.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Raises
    ------
    ValueError
        If ``experiment.task`` is unknown or the device/dtype is unsupported.
    """
    run_dir = Path(run_dir)
    task = cfg["experiment"]["task"]
    if task not in TASKS:
        raise ValueError(
            f"Unknown experiment.task {task!r}; expected one of {sorted(TASKS)}"
        )
    _check_device(cfg)
    level = getattr(logging, str(cfg["logging"]["level"]).upper())
    configure_logging(run_dir / "run.log", level=level)
    config_path = _ensure_config_snapshot(cfg, run_dir)
    status_path = run_dir / "run_status.json"
    run_id = cfg["experiment"]["run_id"] or run_dir.name
    status = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sweep_id": cfg["experiment"]["sweep_id"],
        "candidate": describe_candidate(cfg),
        "seed": cfg["seed"],
        "repetition": cfg["experiment"]["repetition"],
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(_REPO_ROOT),
        "dataset": {
            "name": "pending",
            "splits": "pending",
            "preprocessing": "pending",
            "subset": "pending",
        },
        "selection": _selection(cfg),
        "log_path": "run.log",
        "config_path": config_path.name,
        "config_sha256": sha256(config_path),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(status_path, status)
    logger.info(
        "RUN_STARTED | run_id=%s | task=%s | experiment=%s | seed=%s",
        run_id,
        task,
        cfg["experiment"]["id"],
        cfg["seed"],
    )
    try:
        metrics, dataset = TASK_FUNCTIONS[task](cfg, run_dir)
        status["dataset"] = dataset
        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, metrics)
        logger.info("METRICS_WRITTEN | path=%s", metrics_path.name)
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(metrics_path),
        )
        write_json(status_path, status)
        logger.info("RUN_COMPLETED | run_id=%s", run_id)
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", run_id)
        raise
