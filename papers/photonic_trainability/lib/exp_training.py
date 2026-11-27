"""E8 (Fig. 7) and E9 (Fig. 9): shot-based training of photonic circuits.

Circuits (``circuit`` block):

* ``{"type": "clements"}`` - Clements mesh of phase-only MZIs, initialised at a
  Haar-random unitary through the exact decomposition (paper Sec. 4.1);
* ``{"type": "loop", "loops": 2}`` - cascaded one-bin-delay loop processor, a
  stand-in for ORCA's PT-2 (Sec. 4.2); angles start uniform in ``[0, pi)``
  (assumption, the paper does not say).

Observables (``observable`` block):

* ``{"type": "network", "hidden_layers": 3, "bias": true, "normalise":
  "uniform_std"}`` - random quadratic neural-network observable (Sec. 3.2.3),
  a PNP of order ``2^hidden_layers``, optionally rescaled to unit standard
  deviation under the maximally mixed output state;
* ``{"type": "random_pnp", "terms": 100, "min_order": 1, "max_order": 2,
  "coefficient_range": [-5, 5]}`` - Fig. 9's 100-term order-2 PNP.

Methods, all with the same initial parameters, observable and iteration count:

* ``gpsr_shots`` - paper method: GPSR (Eq. 143) with ``shots_per_shift``
  multinomial samples per shifted loss;
* ``gpsr_exact`` - the same rule with exact losses (noise-free reference);
* ``spsa_equal_shots`` - sanity baseline: simultaneous-perturbation estimate
  with the same total shots per iteration split over two evaluations.

The loss reported per iteration is always the exact loss at the current
parameters. Optional learning-rate selection (``learning_rate_grid``) runs
``gpsr_exact`` on ``selection_seed`` for each candidate and keeps the one with
the lowest final exact loss (ties within 1e-12 go to the smaller rate), as
predeclared in LOG.md.

Artifacts: ``losses.csv`` (method, seed, iteration, exact and estimated loss),
``trajectories.csv`` (15 randomly chosen parameters of the first seed, paper
Fig. 7c,d / 9b) and ``learning_rate_selection.csv`` when selection runs.
"""

from __future__ import annotations

import csv
import logging
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from . import clements
from .combinatorics import fock_dimension
from .fock import fock_basis, input_occupation
from .observables import (
    QuadraticNetworkObservable,
    evaluate_polynomial,
    random_polynomial,
    sample_distinct_monomials,
)
from .photonic_sim import haar_unitary
from .training import PhotonicExpectation, gpsr_gradient

logger = logging.getLogger(__name__)

LOSS_TABLE = "losses.csv"
TRAJECTORY_TABLE = "trajectories.csv"
SELECTION_TABLE = "learning_rate_selection.csv"
TRACKED_PARAMETERS = 15
TIE_TOLERANCE = 1e-12


def describe_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Descriptor for ``run_status.json``."""
    return {
        "name": "synthetic circuit initialisations and observables (from seed)",
        "splits": {
            "evaluation_seeds": list(cfg["seeds"]),
            "selection_seed": cfg.get("selection_seed"),
        },
        "preprocessing": "none",
        "subset": {
            "modes": cfg["modes"],
            "circuit": cfg["circuit"],
            "observable": cfg["observable"],
        },
    }


def _build_problem(cfg: dict[str, Any], seed: int):
    """Initial parameters and the expectation object for one seed."""
    modes = int(cfg["modes"])
    occupation = input_occupation(modes, cfg["input"])
    dimension = fock_dimension(sum(occupation), modes)
    if dimension > int(cfg["max_fock_dimension"]):
        raise ValueError(
            f"Fock dimension {dimension} exceeds max_fock_dimension "
            f"{cfg['max_fock_dimension']}; MerLin computes the full output "
            "distribution, so use fewer modes or photons"
        )
    circuit_rng = np.random.default_rng([seed, 1])
    observable_rng = np.random.default_rng([seed, 2])
    observable_cfg = cfg["observable"]
    if observable_cfg["type"] == "network":
        network = QuadraticNetworkObservable.random(
            modes,
            int(observable_cfg["hidden_layers"]),
            observable_rng,
            hidden_width=observable_cfg.get("hidden_width"),
            with_bias=bool(observable_cfg["bias"]),
        )
        if observable_cfg["normalise"] == "uniform_std":
            network = network.normalised(fock_basis(sum(occupation), modes))
        elif observable_cfg["normalise"] != "none":
            raise ValueError(
                f"Unknown normalise option {observable_cfg['normalise']!r}"
            )
        observable, order = network, network.polynomial_order
    elif observable_cfg["type"] == "random_pnp":
        monomials = sample_distinct_monomials(
            modes,
            int(observable_cfg["min_order"]),
            int(observable_cfg["max_order"]),
            int(observable_cfg["terms"]),
            "uniform_monomial",
            observable_rng,
        )
        low, high = observable_cfg["coefficient_range"]
        polynomial = random_polynomial(
            monomials, None, float(low), float(high), observable_rng
        )
        order = max(sum(e for _, e in m) for m in polynomial)

        def observable(patterns, polynomial=polynomial):
            return evaluate_polynomial(polynomial, patterns)

    else:
        raise ValueError(f"Unknown observable type {observable_cfg['type']!r}")
    circuit_cfg = cfg["circuit"]
    if circuit_cfg["type"] == "clements":
        mesh = clements.decompose(haar_unitary(modes, circuit_rng))
        circuit, initial, scale = (
            clements.perceval_circuit(mesh),
            mesh.parameters(),
            1.0,
        )
    elif circuit_cfg["type"] == "loop":
        blocks = clements.loop_circuit_blocks(modes, int(circuit_cfg["loops"]))
        circuit = clements.perceval_loop_circuit(modes, blocks)
        initial, scale = circuit_rng.uniform(0.0, np.pi, size=len(blocks)), 2.0
    else:
        raise ValueError(f"Unknown circuit type {circuit_cfg['type']!r}")
    # Phase 04 adaptation: `training_hardware` runs require an explicit hardware
    # condition; Phase 03 `training` runs use the ideal PNR model.
    hardware = cfg["hardware"] if cfg["experiment"] == "training_hardware" else None
    expectation = PhotonicExpectation(
        circuit,
        occupation,
        observable,
        parameter_scale=scale,
        device=str(cfg["device"]),
        hardware=hardware,
    )
    gradient_order = int(cfg.get("gpsr_order") or order)
    if gradient_order < min(order, sum(occupation)):
        raise ValueError(
            f"gpsr_order {gradient_order} is below the frequency bound "
            f"min(order={order}, photons={sum(occupation)})"
        )
    return initial, expectation, gradient_order


def _shots_per_shift(cfg: dict[str, Any], gradient_order: int) -> int:
    shots = int(cfg["shots"])
    interpretation = cfg["shot_interpretation"]
    if interpretation == "per_shift":
        return shots
    if interpretation == "per_gradient_component":
        return max(1, shots // (2 * gradient_order))
    raise ValueError(f"Unknown shot_interpretation {interpretation!r}")


def _train(
    cfg: dict[str, Any],
    method: str,
    seed: int,
    learning_rate: float,
    record_rows: list[dict[str, Any]] | None,
    trajectory_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    parameters, expectation, gradient_order = _build_problem(cfg, seed)
    iterations = int(cfg["iterations"])
    shots = _shots_per_shift(cfg, gradient_order)
    shot_rng = np.random.default_rng([seed, 3, zlib.crc32(method.encode())])
    total_shots = len(parameters) * 2 * gradient_order * shots
    tracked = np.random.default_rng([seed, 4]).choice(
        len(parameters), size=min(TRACKED_PARAMETERS, len(parameters)), replace=False
    )
    exact_losses = [float(expectation.exact_losses(parameters[None, :])[0])]
    for iteration in range(1, iterations + 1):
        if method == "gpsr_shots":

            def losses(batch):
                return expectation.shot_losses(batch, shots, shot_rng)

            gradient = gpsr_gradient(losses, parameters, gradient_order)
        elif method == "gpsr_exact":
            gradient = gpsr_gradient(
                expectation.exact_losses, parameters, gradient_order
            )
        elif method == "spsa_equal_shots":
            perturbation = float(cfg["spsa_perturbation"])
            direction = shot_rng.choice([-1.0, 1.0], size=len(parameters))
            batch = np.stack(
                [
                    parameters + perturbation * direction,
                    parameters - perturbation * direction,
                ]
            )
            plus, minus = expectation.shot_losses(batch, total_shots // 2, shot_rng)
            gradient = (plus - minus) / (2 * perturbation) * direction
        else:
            raise ValueError(f"Unknown training method {method!r}")
        parameters = parameters - learning_rate * gradient
        exact = float(expectation.exact_losses(parameters[None, :])[0])
        exact_losses.append(exact)
        logger.info(
            "TRAIN_ITERATION_COMPLETED | method=%s | seed=%d | iteration=%d/%d | "
            "exact_loss=%.8g | grad_norm=%.6g | lr=%.6g",
            method,
            seed,
            iteration,
            iterations,
            exact,
            float(np.linalg.norm(gradient)),
            learning_rate,
        )
        if record_rows is not None:
            record_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "iteration": iteration,
                    "exact_loss": exact,
                }
            )
        if trajectory_rows is not None:
            trajectory_rows.extend(
                {
                    "method": method,
                    "seed": seed,
                    "iteration": iteration,
                    "parameter": int(index),
                    "value": float(np.angle(np.exp(1j * parameters[index]))),
                }
                for index in tracked
            )
    steps = np.diff(exact_losses)
    return {
        "method": method,
        "seed": seed,
        "learning_rate": learning_rate,
        "parameters": len(parameters),
        "gpsr_order": gradient_order,
        "shots_per_shift": shots,
        "shots_per_iteration": total_shots,
        "initial_exact_loss": exact_losses[0],
        "final_exact_loss": exact_losses[-1],
        "improvement": exact_losses[0] - exact_losses[-1],
        "fraction_decreasing_steps": float(np.mean(steps < 0)),
    }


def _select_learning_rate(cfg: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    seed = int(cfg["selection_seed"])
    rows = []
    for learning_rate in cfg["learning_rate_grid"]:
        result = _train(cfg, "gpsr_exact", seed, float(learning_rate), None, None)
        rows.append({"learning_rate": float(learning_rate), **result})
        logger.info(
            "SELECTION_CANDIDATE_COMPLETED | lr=%.6g | final_exact_loss=%.8g",
            float(learning_rate),
            result["final_exact_loss"],
        )
    return select_learning_rate(rows), rows


def select_learning_rate(rows: list[dict[str, Any]]) -> float:
    """Apply the predeclared rule: lowest finite final exact loss, ties to the smaller rate.

    Raises
    ------
    ValueError
        If no candidate has a finite final loss.
    """
    finite = [row for row in rows if np.isfinite(row["final_exact_loss"])]
    if not finite:
        raise ValueError("Every learning-rate candidate diverged")
    best = min(row["final_exact_loss"] for row in finite)
    tied = [row for row in finite if row["final_exact_loss"] - best <= TIE_TOLERANCE]
    return min(row["learning_rate"] for row in tied)


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Train every method on every seed; see the module docstring.

    Parameters
    ----------
    cfg : dict
        Needs ``modes``, ``input``, ``circuit``, ``observable``, ``iterations``,
        ``shots``, ``shot_interpretation``, ``methods``, ``seeds``,
        ``device``, and either ``learning_rate`` or ``learning_rate_grid`` with
        ``selection_seed``; ``spsa_perturbation`` when SPSA runs; optional
        ``gpsr_order`` (default: the observable order).
    run_dir : pathlib.Path
        Destination of the tables.

    Returns
    -------
    dict
        Per-run results and per-method aggregates.
    """
    seeds = [int(seed) for seed in cfg["seeds"]]
    logger.info(
        "DATASET_READY | ensemble=synthetic_training_problems | modes=%s | seeds=%s | "
        "circuit=%s | observable=%s",
        cfg["modes"],
        seeds,
        cfg["circuit"]["type"],
        cfg["observable"]["type"],
    )
    selection_rows: list[dict[str, Any]] = []
    logger.info(
        "TRAINING_STARTED | methods=%s | seeds=%s | iterations=%s",
        cfg["methods"],
        seeds,
        cfg["iterations"],
    )
    if "learning_rate_grid" in cfg:
        learning_rate, selection_rows = _select_learning_rate(cfg)
        logger.info("LEARNING_RATE_SELECTED | lr=%.6g", learning_rate)
    else:
        learning_rate = float(cfg["learning_rate"])
    loss_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    results = []
    for seed in seeds:
        for method in cfg["methods"]:
            result = _train(
                cfg,
                method,
                seed,
                learning_rate,
                loss_rows,
                trajectory_rows if seed == seeds[0] else None,
            )
            results.append(result)
    logger.info("EVALUATION_STARTED | metric=final_exact_loss")
    aggregates = {}
    for method in cfg["methods"]:
        rows = [result for result in results if result["method"] == method]
        improvements = [row["improvement"] for row in rows]
        aggregates[method] = {
            "runs": len(rows),
            "improvement_mean": float(np.mean(improvements)),
            "improvement_std": float(np.std(improvements, ddof=1))
            if len(rows) > 1
            else None,
            "final_exact_loss_mean": float(
                np.mean([row["final_exact_loss"] for row in rows])
            ),
            "fraction_decreasing_steps_mean": float(
                np.mean([row["fraction_decreasing_steps"] for row in rows])
            ),
        }
        logger.info(
            "EVALUATION_COMPLETED | method=%s | improvement_mean=%.6g | runs=%d",
            method,
            aggregates[method]["improvement_mean"],
            len(rows),
        )
    for name, rows in (
        (LOSS_TABLE, loss_rows),
        (TRAJECTORY_TABLE, trajectory_rows),
        (SELECTION_TABLE, selection_rows),
    ):
        if not rows:
            continue
        with (run_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("ARTIFACT_WRITTEN | path=%s | rows=%d", name, len(rows))
    return {
        "experiment": "training",
        "learning_rate": learning_rate,
        "learning_rate_selected": "learning_rate_grid" in cfg,
        "results": results,
        "aggregates": aggregates,
        "artifacts": {
            "losses": LOSS_TABLE,
            "trajectories": TRAJECTORY_TABLE,
            "selection": SELECTION_TABLE if selection_rows else None,
        },
    }
