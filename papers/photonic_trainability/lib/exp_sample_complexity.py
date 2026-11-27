"""E10 (paper Prop. 1, App. A and Eq. 144): does ``N ~ Var_samp / Var_circ`` suffice?

Part A, loss ordering. For each case, draw ``pairs`` independent pairs of Haar
interferometers ``(U, U')``. With ``N`` shots per circuit, estimate both losses
and record whether the estimated difference has the sign of the true
difference. ``N`` runs over multiples of the exact variance ratio
``r = Var_samp / Var_circ``. Reported per ``N``:

* success over all pairs, and over "resolvable" pairs with
  ``|D| > 2 epsilon sqrt(Var_circ)`` (Prop. 1);
* Prop. 1's guarantee for resolvable pairs with ``t`` chosen optimally,
  ``1 - 4 sqrt(r / (N epsilon^2))`` (from ``1 - 2/t - 2 delta`` with
  ``N = t r / (delta epsilon^2)``);
* the fraction of pairs with ``|D| >= sqrt(Var_circ)``, which App. A.4(ii)
  puts at about 0.48 for Gaussian loss differences.

Part B, gradient variance. On Clements meshes at Haar-random initialisation,
the exact variance of an ``N``-shot GPSR gradient component is
``sum_j c_j^2 Var[O | theta + kappa_j e_k] / N``; its average over parameters
and initialisations is compared with paper Eq. 144,
``(2R^2 + 1) / 6 * Var_samp / N``, which assumes equal sample variance at the
shifted points.

Artifact: ``ordering_table.csv`` (one row per case and N).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import clements
from .data import build_dataloader
from .fock import fock_basis, input_occupation
from .photonic_sim import output_distribution
from .specs import build_observable
from .training import (
    PhotonicExpectation,
    gpsr_shifts_and_coefficients,
    shifted_parameter_batch,
)

logger = logging.getLogger(__name__)

TABLE_NAME = "ordering_table.csv"


def describe_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Descriptor for ``run_status.json``."""
    return {
        "name": "haar_random_interferometers and Clements meshes (synthetic, from seed)",
        "splits": {
            "pairs": int(cfg["pairs"]),
            "gradient_inits": int(cfg["gradient"]["inits"]),
        },
        "preprocessing": "none",
        "subset": [case["name"] for case in cfg["cases"]],
    }


def _ordering_case(cfg, case, case_index, rng):
    modes = int(case["modes"])
    occupation = input_occupation(modes, case["input"])
    observable = build_observable(case["observable"], modes, rng)
    exact = {k: float(v) for k, v in observable.exact_moments(occupation).items()}
    ratio = exact["sample_variance"] / exact["circuit_variance"]
    pairs = int(cfg["pairs"])
    loader = build_dataloader(
        {
            **cfg,
            "modes": modes,
            "case_index": case_index,
            "monte_carlo": {"unitaries": 2 * pairs, "batch_size": 2 * pairs},
        },
        split="ordering",
    )
    unitaries = next(iter(loader))
    basis = fock_basis(sum(occupation), modes)
    values = observable.evaluate(basis)
    distributions = np.array(
        [output_distribution(unitary, occupation, basis)[1] for unitary in unitaries]
    )
    distributions = np.clip(distributions, 0.0, None)
    distributions /= distributions.sum(axis=1, keepdims=True)
    losses = distributions @ values
    differences = losses[0::2] - losses[1::2]
    epsilon = float(cfg["epsilon"])
    scale = np.sqrt(exact["circuit_variance"])
    resolvable = np.abs(differences) > 2 * epsilon * scale
    rows = []
    for factor in cfg["shot_factors"]:
        shots = max(1, int(round(float(factor) * ratio)))
        estimates = np.array(
            [
                rng.multinomial(shots, distribution) @ values / shots
                for distribution in distributions
            ]
        )
        estimated_differences = estimates[0::2] - estimates[1::2]
        correct = np.sign(estimated_differences) == np.sign(differences)
        guarantee = 1.0 - 4.0 * np.sqrt(ratio / (shots * epsilon**2))
        rows.append(
            {
                "case": case["name"],
                "modes": modes,
                "photons": sum(occupation),
                "variance_ratio": ratio,
                "shot_factor": float(factor),
                "shots": shots,
                "success_all_pairs": float(correct.mean()),
                "success_resolvable_pairs": float(correct[resolvable].mean())
                if resolvable.any()
                else None,
                "resolvable_fraction": float(resolvable.mean()),
                "prop1_guarantee_resolvable": float(max(guarantee, 0.0)),
                "fraction_abs_difference_above_sqrt_var_circ": float(
                    np.mean(np.abs(differences) >= scale)
                ),
                "monte_carlo_circuit_variance": float(np.var(losses, ddof=1)),
                "exact_circuit_variance": exact["circuit_variance"],
            }
        )
        logger.info(
            "EVALUATION_POINT | case=%s | N=%d | N/ratio=%.3g | success_all=%.4f | "
            "success_resolvable=%s | guarantee=%.4f",
            case["name"],
            shots,
            float(factor),
            rows[-1]["success_all_pairs"],
            rows[-1]["success_resolvable_pairs"],
            rows[-1]["prop1_guarantee_resolvable"],
        )
    return rows


def _gradient_variance_case(cfg, rng):
    gradient_cfg = cfg["gradient"]
    modes = int(gradient_cfg["modes"])
    occupation = input_occupation(modes, gradient_cfg["input"])
    observable = build_observable(gradient_cfg["observable"], modes, rng)
    order = observable.order
    exact = observable.exact_moments(occupation)
    _, coefficients = gpsr_shifts_and_coefficients(order)
    loader = build_dataloader(
        {
            **cfg,
            "modes": modes,
            "monte_carlo": {
                "unitaries": int(gradient_cfg["inits"]),
                "batch_size": int(gradient_cfg["inits"]),
            },
        },
        split="gradient",
    )
    weighted, unshifted = [], []
    expectation = None
    for unitary in next(iter(loader)):
        mesh = clements.decompose(unitary)
        if expectation is None:
            expectation = PhotonicExpectation(
                clements.perceval_circuit(mesh), occupation, observable.evaluate
            )
        parameters = mesh.parameters()
        variances = expectation.shot_variances(
            shifted_parameter_batch(parameters, order)
        )
        per_parameter = (
            variances.reshape(len(parameters), len(coefficients)) @ coefficients**2
        )
        weighted.append(per_parameter)
        unshifted.append(expectation.shot_variances(parameters[None, :])[0])
    weighted = np.concatenate(weighted)
    predicted = (2 * order**2 + 1) / 6 * float(exact["sample_variance"])
    return {
        "observable": observable.description,
        "modes": modes,
        "gpsr_order": order,
        "inits": int(gradient_cfg["inits"]),
        "sum_coefficients_squared": float(np.sum(coefficients**2)),
        "mean_single_shot_gradient_variance": float(weighted.mean()),
        "eq144_prediction": predicted,
        "ratio_measured_to_eq144": float(weighted.mean() / predicted),
        "mean_unshifted_sample_variance": float(np.mean(unshifted)),
        "exact_haar_sample_variance": float(exact["sample_variance"]),
    }


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run parts A and B; see the module docstring.

    Parameters
    ----------
    cfg : dict
        Needs ``seed``, ``pairs``, ``epsilon``, ``shot_factors``, ``cases`` and
        ``gradient`` (``modes``, ``input``, ``observable``, ``inits``).
    run_dir : pathlib.Path
        Destination of ``ordering_table.csv``.

    Returns
    -------
    dict
        Metrics record.
    """
    rng = np.random.default_rng(int(cfg["seed"]))
    logger.info(
        "DATASET_READY | ensemble=haar_pairs_and_meshes | pairs=%d | cases=%d",
        int(cfg["pairs"]),
        len(cfg["cases"]),
    )
    rows = []
    for case_index, case in enumerate(cfg["cases"]):
        logger.info("EVALUATION_STARTED | part=ordering | case=%s", case["name"])
        case_rows = _ordering_case(cfg, case, case_index, rng)
        rows.extend(case_rows)
        logger.info(
            "EVALUATION_COMPLETED | part=ordering | case=%s | ratio=%.6g",
            case["name"],
            case_rows[0]["variance_ratio"],
        )
    logger.info("EVALUATION_STARTED | part=gradient_variance")
    gradient = _gradient_variance_case(cfg, rng)
    logger.info(
        "EVALUATION_COMPLETED | part=gradient_variance | measured/eq144=%.4f",
        gradient["ratio_measured_to_eq144"],
    )
    with (run_dir / TABLE_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("ARTIFACT_WRITTEN | path=%s | rows=%d", TABLE_NAME, len(rows))
    return {
        "experiment": "sample_complexity",
        "ordering": rows,
        "gradient_variance": gradient,
        "artifacts": {"table": TABLE_NAME},
    }
