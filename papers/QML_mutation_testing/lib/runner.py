"""Runtime entry point for the photonic mutation-testing reproduction.

Pipeline per run:

1. train the original (unmutated) photonic QNN, once per seed;
2. derive the paper's test suite from the correctly classified samples;
3. generate and evaluate the seven new mutation operators;
4. generate and evaluate the prior (Muskit / QMutPy style) operators on the
   same suite, for the comparison of Table I against Table II;
5. train fair classical baselines for context;
6. aggregate everything over seeds and write structured evidence.

Steps 3 and 4 are skipped when ``mutation_testing.enabled`` is false, which is
the Iteration-1 configuration.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
import time
import uuid
from pathlib import Path

import torch
from lib.classical_baseline import run_classical_baselines
from lib.data import load_image_splits, load_tabular_splits
from lib.experiment_logging import git_state, sha256, utc_now, write_json
from lib.mutation_testing import (
    SuiteSpec,
    evaluate_mutants,
    outcomes_to_records,
    summarise,
)
from lib.mutations import (
    generate_baseline_mutants,
    generate_control_mutants,
    generate_new_operator_mutants,
    suppressed_redundant_counts,
)
from lib.photonic_qnn import (
    MutatableQCNN,
    PhotonicQnnSpec,
    gate_model_parameter_count,
)
from lib.training import (
    TrainingConfig,
    build_mutation_test_suite,
    predict_with_shots,
    train_photonic_qnn,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[3]
NEW_OPERATOR_ORDER = ("APC", "DFC", "APGC", "LS", "ILS", "ALA", "ALD")
PRIOR_OPERATOR_ORDER = ("ADD", "DELETE", "CHANGE")
IMAGE_DATASETS = ("mnist", "fashion_mnist", "kmnist")


def _require(cfg: dict, *path: str):
    """Return a nested config value, failing loudly when it is absent."""
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing required config key: {'.'.join(path)}")
        node = node[key]
    return node


def _aggregate(values: list) -> dict:
    """Return mean, standard deviation, and range, ignoring ``None`` entries."""
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
        "n": len(clean),
    }


def _write_outcome_csv(path: Path, records: list[dict]) -> None:
    """Write per-mutant outcome records as CSV."""
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _infer_image_hw(x: torch.Tensor) -> tuple[int, int]:
    """Déduit (hauteur, largeur) depuis un tenseur d'images."""
    if x.dim() == 4:  # (N, C, H, W)
        return int(x.shape[-2]), int(x.shape[-1])
    if x.dim() == 3:  # (N, H, W)
        return int(x.shape[-2]), int(x.shape[-1])
    if x.dim() == 2:  # (N, H*W) -> image carrée supposée
        n = int(x.shape[-1])
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(
                f"Impossible d'inférer une image carrée depuis {n} features."
            )
        return side, side
    raise ValueError(f"Format d'image non supporté: {tuple(x.shape)}")


def _aggregate_family(summaries: list[dict], operators: tuple[str, ...]) -> dict:
    """Aggregate one mutant family's per-seed summaries."""
    per_operator = {}
    for operator in operators:
        rows = [
            summary["per_operator"][operator]
            for summary in summaries
            if operator in summary["per_operator"]
        ]
        if not rows:
            continue
        per_operator[operator] = {
            "generated": rows[0]["generated"],
            "killed": _aggregate([row["killed"] for row in rows]),
            "incompetent": _aggregate([row["incompetent"] for row in rows]),
            "mutation_score": _aggregate([row["mutation_score"] for row in rows]),
            "mutation_score_analytic": _aggregate(
                [row["mutation_score_analytic"] for row in rows]
            ),
            "distinct_circuits": rows[0]["distinct_circuits"],
            "distinct_behaviours": _aggregate(
                [row["distinct_behaviours"] for row in rows]
            ),
            "suppressed_redundant": rows[0]["suppressed_redundant"],
            "generation_time_per_mutant_s": _aggregate(
                [row["generation_time_per_mutant_s"] for row in rows]
            ),
            "evaluation_time_per_mutant_s": _aggregate(
                [row["evaluation_time_per_mutant_s"] for row in rows]
            ),
        }

    return {
        "per_operator": per_operator,
        "total_generated": summaries[0]["total_generated"],
        "mutation_score": _aggregate(
            [summary["mutation_score"] for summary in summaries]
        ),
        "mutation_score_analytic": _aggregate(
            [summary["mutation_score_analytic"] for summary in summaries]
        ),
        "total_incompetent": _aggregate(
            [summary["total_incompetent"] for summary in summaries]
        ),
        "distinct_behaviours_total": _aggregate(
            [summary["distinct_behaviours_total"] for summary in summaries]
        ),
        "behavioural_redundancy_rate": _aggregate(
            [summary["behavioural_redundancy_rate"] for summary in summaries]
        ),
        "suppressed_redundant_total": summaries[0]["suppressed_redundant_total"],
        "generation_time_per_mutant_s": _aggregate(
            [
                summary["generation_time_total_s"] / summary["total_generated"]
                for summary in summaries
            ]
        ),
        "evaluation_time_per_mutant_s": _aggregate(
            [
                summary["evaluation_time_total_s"] / summary["total_generated"]
                for summary in summaries
            ]
        ),
    }


def _run_mutation_testing(
    *,
    spec: PhotonicQnnSpec,
    model,
    splits,
    shots: int,
    mutation_cfg: dict,
    seed: int,
    suite: dict,
    run_dir: Path,
) -> dict:
    """Generate and evaluate both mutant families for one trained model."""

    trained_ir = model.export_ir()
    suite_spec = SuiteSpec.from_suite(splits.x_suite, splits.y_suite, suite)
    logger.info(
        "MUTATION_SUITE_READY | seed=%d | suite_shots=%d | suite_analytic=%d | candidates=%d",
        seed,
        len(suite_spec.shot_positions),
        len(suite_spec.analytic_positions),
        int(splits.y_suite.shape[0]),
    )

    generation_started = time.perf_counter()
    new_mutants = generate_new_operator_mutants(
        trained_ir,
        n_features=spec.n_features if spec is not None else splits.x_train[0].numel(),
        apc_granularity=str(mutation_cfg.get("apc_granularity", "gate")),
        ala_seed=seed,
    )
    new_generation_s = time.perf_counter() - generation_started

    generation_started = time.perf_counter()
    include_baseline = bool(mutation_cfg.get("include_baseline", True))
    baseline_mutants = generate_baseline_mutants(trained_ir) if include_baseline else []
    baseline_generation_s = time.perf_counter() - generation_started

    new_outcomes = evaluate_mutants(
        new_mutants,
        spec,
        model,
        suite_spec,
        shots,
        family="new",
        log_every=int(mutation_cfg.get("log_every", 100)),
    )
    new_summary = summarise(
        new_outcomes,
        family="new",
        suppressed=suppressed_redundant_counts(trained_ir),
    )
    new_summary["generation_time_total_s"] = new_generation_s
    _write_outcome_csv(
        run_dir / f"mutants_new_seed{seed}.csv", outcomes_to_records(new_outcomes)
    )

    control_mutants = generate_control_mutants(
        trained_ir, count=int(mutation_cfg.get("control_replicates", 12))
    )
    control_outcomes = evaluate_mutants(
        control_mutants,
        spec,
        model,
        suite_spec,
        shots,
        family="control",
        log_every=max(1, len(control_mutants)),
    )
    control_summary = summarise(control_outcomes, family="control")
    control_summary["generation_time_total_s"] = 0.0

    result = {
        "new": new_summary,
        "control": control_summary,
        "suite_size": len(suite_spec.shot_positions),
        "suite_size_analytic": len(suite_spec.analytic_positions),
    }

    if baseline_mutants:
        baseline_outcomes = evaluate_mutants(
            baseline_mutants,
            spec,
            model,
            suite_spec,
            shots,
            family="baseline",
            log_every=int(mutation_cfg.get("log_every", 100)),
        )
        baseline_summary = summarise(baseline_outcomes, family="baseline")
        baseline_summary["generation_time_total_s"] = baseline_generation_s
        _write_outcome_csv(
            run_dir / f"mutants_baseline_seed{seed}.csv",
            outcomes_to_records(baseline_outcomes),
        )
        result["baseline"] = baseline_summary

    return result


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run the reproduction pipeline and record evidence."""

    context = cfg.get("run_context") or {}

    dataset_cfg = _require(cfg, "dataset")
    model_cfg = _require(cfg, "model", "params")
    training_cfg = _require(cfg, "training")
    evaluation_cfg = _require(cfg, "evaluation")
    mutation_cfg = _require(cfg, "mutation_testing")
    seeds = [int(seed) for seed in _require(cfg, "experiment", "seeds")]

    run_id = (
        context.get("run_id")
        or f"{dataset_cfg['name']}-photonic-{uuid.uuid4().hex[:8]}"
    )

    started_at = utc_now()
    config_path = run_dir / "config_snapshot.json"
    metrics_path = run_dir / "metrics.json"
    status_path = run_dir / "run_status.json"

    status = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sweep_id": context.get("sweep_id"),
        "candidate": context.get("candidate")
        or {
            "n_layers": model_cfg["n_layers"],
            "lr": training_cfg["lr"],
            "epochs": training_cfg["epochs"],
            "apc_granularity": mutation_cfg.get("apc_granularity", "gate"),
        },
        "seed": seeds[0] if len(seeds) == 1 else seeds,
        "repetition": context.get("repetition", len(seeds)),
        "status": "RUNNING",
        "started_at": started_at,
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": {
            "name": dataset_cfg["name"],
            "splits": {
                "test_size": dataset_cfg["test_size"],
                "val_size": dataset_cfg["val_size"],
                "split_seed": cfg.get("seed"),
            },
            "preprocessing": (
                f"stratified split, MinMaxScaler[0,{dataset_cfg['feature_max']}] "
                "fitted on train"
            ),
            "subset": f"{dataset_cfg['n_suite_samples']} test samples for the suite",
        },
        "selection": {
            "name": "val_accuracy",
            "direction": "maximize",
            "split": "validation",
            "checkpoint": "final epoch",
        },
        "log_path": "run.log",
        "config_path": config_path.name,
        "config_sha256": sha256(config_path),
        "metrics_path": metrics_path.name,
        "metrics_sha256": None,
        "error": None,
    }
    write_json(status_path, status)
    logger.info(
        "RUN_STARTED | run_id=%s | seeds=%s | mutation_testing=%s",
        run_id,
        seeds,
        bool(mutation_cfg.get("enabled", False)),
    )

    try:
        is_image_task = dataset_cfg["name"] in IMAGE_DATASETS

        if is_image_task:
            splits = load_image_splits(
                name=dataset_cfg["name"],
                seed=int(cfg["seed"]),
                test_size=float(dataset_cfg["test_size"]),
                val_size=float(dataset_cfg["val_size"]),
                n_suite_samples=int(dataset_cfg["n_suite_samples"]),
            )

            n_features_data = splits.x_train.view(splits.x_train.size(0), -1).size(1)
            n_classes_data = len(torch.unique(splits.y_train))
            n_photons_cfg = int(model_cfg["n_photons"])

            dynamic_input_state = [1] * n_photons_cfg + [0] * (
                n_features_data - n_photons_cfg
            )

            # ⚠️ Taille RÉELLE des images (28x28 pour MNIST), pas sqrt(n_modes)
            input_h, input_w = _infer_image_hw(splits.x_train)

            model = MutatableQCNN(
                input_shape=(input_h, input_w),
                num_classes=int(model_cfg.get("n_classes", n_classes_data)),
            )

            modes_cfg = int(model_cfg.get("n_modes", 16))

            spec = PhotonicQnnSpec(
                n_features=n_features_data,
                n_classes=int(model_cfg.get("n_classes", n_classes_data)),
                n_modes=modes_cfg,
                n_photons=n_photons_cfg,
                n_layers=int(model_cfg["n_layers"]),
                encoding_scale=float(model_cfg["encoding_scale"]),
                computation_space=str(model_cfg["computation_space"]),
                input_state=dynamic_input_state,
            )
        else:
            splits = load_tabular_splits(
                name=dataset_cfg["name"],
                seed=int(cfg["seed"]),
                test_size=float(dataset_cfg["test_size"]),
                val_size=float(dataset_cfg["val_size"]),
                n_suite_samples=int(dataset_cfg["n_suite_samples"]),
                feature_max=float(dataset_cfg["feature_max"]),
            )
            model = None

            n_features_data = splits.x_train.shape[1]
            n_classes_data = len(torch.unique(splits.y_train))
            n_photons_cfg = int(model_cfg["n_photons"])
            dynamic_input_state = [1] * n_photons_cfg + [0] * (
                n_features_data - n_photons_cfg
            )

            spec = PhotonicQnnSpec(
                n_features=n_features_data,
                n_classes=n_classes_data,
                n_modes=n_features_data,
                n_photons=n_photons_cfg,
                n_layers=int(model_cfg["n_layers"]),
                encoding_scale=float(model_cfg["encoding_scale"]),
                computation_space=str(model_cfg["computation_space"]),
                input_state=dynamic_input_state,
            )

        train_cfg = TrainingConfig(
            epochs=int(training_cfg["epochs"]),
            lr=float(training_cfg["lr"]),
            optimizer=str(training_cfg["optimizer"]),
            log_every=int(training_cfg.get("log_every", 10)),
        )
        shots = int(evaluation_cfg["shots"])

        logger.info(
            "TRAINING_STARTED | replicates=%d | epochs=%d | optimizer=%s | lr=%g",
            len(seeds),
            train_cfg.epochs,
            train_cfg.optimizer,
            train_cfg.lr,
        )

        per_seed: list[dict] = []
        suites: list[dict] = []
        new_summaries: list[dict] = []
        baseline_summaries: list[dict] = []
        control_summaries: list[dict] = []
        started = time.time()

        for seed in seeds:
            model, seed_metrics = train_photonic_qnn(
                spec, splits, train_cfg, seed, model=model
            )
            suite = build_mutation_test_suite(model, splits, shots)

            shot_predictions = predict_with_shots(model, splits.x_test, shots)
            seed_metrics["test_accuracy_shots"] = float(
                (shot_predictions == splits.y_test).float().mean().item()
            )
            seed_metrics["test_suite"] = suite
            per_seed.append(seed_metrics)
            suites.append(suite)

            checkpoint = run_dir / f"original_model_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            logger.info("CHECKPOINT_WRITTEN | seed=%d | path=%s", seed, checkpoint.name)

            if mutation_cfg.get("enabled", False):
                mutation_result = _run_mutation_testing(
                    spec=spec,
                    model=model,
                    splits=splits,
                    shots=shots,
                    mutation_cfg=mutation_cfg,
                    seed=seed,
                    suite=suite,
                    run_dir=run_dir,
                )
                new_summaries.append(mutation_result["new"])
                control_summaries.append(mutation_result["control"])
                if "baseline" in mutation_result:
                    baseline_summaries.append(mutation_result["baseline"])

        wall_clock = time.time() - started
        reference_model = model  # last replicate, used for the hardware report

        logger.info("CLASSICAL_BASELINES_STARTED | seeds=%s", seeds)
        classical = run_classical_baselines(
            splits, seeds=seeds, epochs=train_cfg.epochs, lr=train_cfg.lr
        )
        ir_data = reference_model.export_ir()
        n_components = (
            sum(len(layer.elements) for layer in ir_data.values())
            if isinstance(ir_data, dict)
            else len(ir_data.elements)
        )

        metrics = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "scope": (
                "full-pipeline"
                if mutation_cfg.get("enabled", False)
                else "original-model-only"
            ),
            "adaptation": "photonic MerLin adaptation of a gate-model QNN",
            "dataset": dataset_cfg["name"],
            "feature_map": "photonic ZFeatureMap analogue (BS column + phase shifters)",
            "ansatz": "photonic RealAmplitudes analogue (trainable phases + fixed mesh)",
            "seeds": seeds,
            "per_seed": per_seed,
            "aggregate": {
                "train_accuracy": _aggregate([m["train_accuracy"] for m in per_seed]),
                "val_accuracy": _aggregate([m["val_accuracy"] for m in per_seed]),
                "test_accuracy": _aggregate([m["test_accuracy"] for m in per_seed]),
                "test_accuracy_shots": _aggregate(
                    [m["test_accuracy_shots"] for m in per_seed]
                ),
                "final_loss": _aggregate([m["final_loss"] for m in per_seed]),
                "test_suite_size": _aggregate([s["n_suite_samples"] for s in suites]),
            },
            "circuit": spec.as_dict(),
            "hardware_report": reference_model.hardware_report(shots=shots),
            "parameter_budget": {
                "photonic_trainable_phases": reference_model.num_trainable_parameters(),
                "equivalent_real_amplitudes_parameters": gate_model_parameter_count(
                    spec.n_modes, spec.n_layers
                ),
                "n_optical_components": n_components,
            },
            "training": {
                "epochs": train_cfg.epochs,
                "optimizer": train_cfg.optimizer,
                "lr": train_cfg.lr,
                "loss": "NLLLoss on log of grouped outcome probabilities",
                "batching": "full batch",
                "total_wall_clock_s": wall_clock,
            },
            "classical_baselines": classical,
        }

        if new_summaries:
            mutation_metrics = {
                "apc_granularity": mutation_cfg.get("apc_granularity", "gate"),
                "shots": shots,
                "suite_size": _aggregate([s["n_suite_samples"] for s in suites]),
                "per_seed": {
                    "new": new_summaries,
                    "baseline": baseline_summaries,
                    "control": control_summaries,
                },
                "new_operators": _aggregate_family(new_summaries, NEW_OPERATOR_ORDER),
                "shot_noise_control": _aggregate_family(
                    control_summaries, ("CONTROL",)
                ),
            }
            if baseline_summaries:
                mutation_metrics["prior_operators"] = _aggregate_family(
                    baseline_summaries, PRIOR_OPERATOR_ORDER
                )
                new_total = new_summaries[0]["total_generated"]
                prior_total = baseline_summaries[0]["total_generated"]

                paper_targets = {
                    "iris": {
                        "ms_new": 0.7159,
                        "ms_prior": 0.3592,
                        "mutants_new": 1584,
                        "mutants_prior": 13448,
                    },
                    "wine": {
                        "ms_new": 0.8309,
                        "ms_prior": 0.3804,
                        "mutants_new": 1440,
                        "mutants_prior": 21465,
                    },
                    "breast_cancer": {
                        "ms_new": 0.7508,
                        "ms_prior": 0.0431,
                        "mutants_new": 2560,
                        "mutants_prior": 38160,
                    },
                }
                target = paper_targets.get(dataset_cfg["name"], paper_targets["iris"])

                mutation_metrics["comparison"] = {
                    "mutants_new": new_total,
                    "mutants_prior": prior_total,
                    "mutant_reduction_factor": prior_total / new_total,
                    "mutation_score_new": mutation_metrics["new_operators"][
                        "mutation_score"
                    ],
                    "mutation_score_prior": mutation_metrics["prior_operators"][
                        "mutation_score"
                    ],
                    "paper_mutation_score_new": target["ms_new"],
                    "paper_mutation_score_prior": target["ms_prior"],
                    "paper_mutants_new": target["mutants_new"],
                    "paper_mutants_prior": target["mutants_prior"],
                    "paper_reduction_factor": (
                        target["mutants_prior"] / target["mutants_new"]
                        if target["mutants_new"]
                        else 0
                    ),
                }
            metrics["mutation_testing"] = mutation_metrics

        write_json(metrics_path, metrics)
        logger.info("METRICS_WRITTEN | path=%s", metrics_path.name)

        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(metrics_path),
        )
        write_json(status_path, status)
        logger.info(
            "RUN_COMPLETED | run_id=%s | test_acc_mean=%.4f | suite_size_mean=%.2f "
            "| wall_clock_s=%.2f",
            run_id,
            metrics["aggregate"]["test_accuracy"]["mean"],
            metrics["aggregate"]["test_suite_size"]["mean"],
            wall_clock,
        )

        summary = {"accuracy": metrics["aggregate"]}
        if "mutation_testing" in metrics:
            summary["mutation_testing"] = {
                key: metrics["mutation_testing"][key]
                for key in ("comparison", "new_operators", "prior_operators")
                if key in metrics["mutation_testing"]
            }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(status_path, status)
        logger.exception("RUN_FAILED | run_id=%s", run_id)
        raise
