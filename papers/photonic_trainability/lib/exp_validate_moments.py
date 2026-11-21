"""E2: validate the exact Haar-moment code against Haar Monte Carlo.

Not a paper figure. Every variance curve in the paper (Figs. 3-5, 10) comes
from the analytic formulas, so this is the first control: for each configured
case it compares

1. the exact first moment with a brute-force average over the maximally mixed
   n-photon state (paper Theorem 3), which must agree exactly;
2. the irrep decomposition identities ``sum_k d_k = |H|^2`` and
   ``sum_k ||P_k(O)||^2 = tr(O^2)``, and the truncation ``P_k(O) = 0`` for
   ``k > order`` that the fast second-moment path relies on;
3. the exact ``mu1, mu2, Var_samp, Var_circ`` with Haar Monte Carlo estimates
   (Perceval SLOS output distributions), via z-scores against batch-means
   standard errors.

A case passes when (1) and (2) hold exactly and every |z| is below the
predeclared ``z_threshold``.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from . import moments
from .combinatorics import fock_dimension
from .data import build_dataloader
from .fock import fock_basis, input_occupation
from .haar_mc import moment_estimates, per_unitary_moments
from .specs import build_observable

logger = logging.getLogger(__name__)

COMPARED_QUANTITIES = ("mu1", "mu2", "sample_variance", "circuit_variance")


def describe_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Synthetic-ensemble descriptor for ``run_status.json``."""
    monte_carlo = cfg["monte_carlo"]
    return {
        "name": "haar_random_interferometers (synthetic, generated from seed)",
        "splits": {"monte_carlo": int(monte_carlo["unitaries"])},
        "preprocessing": "none",
        "subset": [case["name"] for case in cfg["cases"]],
    }


def _exact_identity_checks(case, occupation: tuple[int, ...]) -> dict[str, Any]:
    photons, modes = sum(occupation), len(occupation)
    diagonal = case.diagonal_values(photons, modes)
    first_moment_bruteforce = sum(diagonal.values(), Fraction(0)) / fock_dimension(
        photons, modes
    )
    exact = case.exact_moments(occupation)
    full_g = moments.g_values_diagonal(diagonal, photons, modes, photons)
    norms = moments.irrep_norms(full_g, photons, modes)
    trace_square = sum((value * value for value in diagonal.values()), Fraction(0))
    dimension_sum = sum(moments.irrep_dimension(k, modes) for k in range(photons + 1))
    beyond_order = norms[case.order + 1 :] if case.order < photons else []
    full_mu2 = moments.second_moment_from_g(
        moments.g_values_fock_state(occupation, photons), full_g, photons, modes
    )
    return {
        "first_moment_matches_bruteforce": exact["mu1"] == first_moment_bruteforce,
        "irrep_dimensions_sum_to_dim_squared": dimension_sum
        == fock_dimension(photons, modes) ** 2,
        "irrep_norms_sum_to_trace_square": sum(norms, Fraction(0)) == trace_square,
        "irrep_norms_vanish_beyond_order": all(norm == 0 for norm in beyond_order),
        "truncated_mu2_matches_full_sum": exact["mu2"] == full_mu2,
    }


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run every configured validation case and return the metrics record.

    Parameters
    ----------
    cfg : dict
        Resolved config with ``cases``, ``monte_carlo`` and ``z_threshold``.
    run_dir : pathlib.Path
        Run directory (unused; no extra artifacts beyond metrics).

    Returns
    -------
    dict
        Per-case results and an ``all_passed`` flag.
    """
    z_threshold = float(cfg["z_threshold"])
    batches = int(cfg["monte_carlo"]["batches"])
    rng = np.random.default_rng(int(cfg["seed"]))
    logger.info(
        "DATASET_READY | ensemble=haar_unitaries | cases=%d | unitaries_per_case=%d",
        len(cfg["cases"]),
        int(cfg["monte_carlo"]["unitaries"]),
    )
    cases = []
    for case_index, case_cfg in enumerate(cfg["cases"]):
        modes = int(case_cfg["modes"])
        occupation = input_occupation(modes, case_cfg["input"])
        observable = build_observable(case_cfg["observable"], modes, rng)
        loader = build_dataloader(
            {**cfg, "modes": modes, "case_index": case_index}, split="monte_carlo"
        )
        unitaries = [unitary for batch in loader for unitary in batch]
        logger.info(
            "CASE_ENSEMBLE_READY | case=%s | modes=%d | photons=%d | unitaries=%d | "
            "fock_dim=%d | observable=%s",
            case_cfg["name"],
            modes,
            sum(occupation),
            len(unitaries),
            len(fock_basis(sum(occupation), modes)),
            observable.description,
        )
        logger.info("EVALUATION_STARTED | case=%s", case_cfg["name"])
        identity_checks = _exact_identity_checks(observable, occupation)
        exact = {
            name: float(value)
            for name, value in observable.exact_moments(occupation).items()
        }
        means, second_moments = per_unitary_moments(
            observable.evaluate, occupation, unitaries
        )
        estimates = moment_estimates(means, second_moments, batches)
        comparisons = {}
        for name in COMPARED_QUANTITIES:
            estimate = estimates[name]
            z_score = (estimate["estimate"] - exact[name]) / estimate["standard_error"]
            comparisons[name] = {
                "exact": exact[name],
                "monte_carlo": estimate["estimate"],
                "standard_error": estimate["standard_error"],
                "z_score": z_score,
                "relative_error": abs(estimate["estimate"] - exact[name])
                / abs(exact[name])
                if exact[name] != 0
                else None,
            }
        passed = all(identity_checks.values()) and all(
            abs(entry["z_score"]) < z_threshold for entry in comparisons.values()
        )
        logger.info(
            "EVALUATION_COMPLETED | case=%s | passed=%s | max_abs_z=%.3f | "
            "var_circ_exact=%.6g | var_circ_mc=%.6g | var_samp_exact=%.6g | "
            "var_samp_mc=%.6g | identities=%s",
            case_cfg["name"],
            passed,
            max(abs(entry["z_score"]) for entry in comparisons.values()),
            exact["circuit_variance"],
            estimates["circuit_variance"]["estimate"],
            exact["sample_variance"],
            estimates["sample_variance"]["estimate"],
            all(identity_checks.values()),
        )
        cases.append(
            {
                "name": case_cfg["name"],
                "modes": modes,
                "photons": sum(occupation),
                "input": list(occupation),
                "observable": observable.description,
                "order": observable.order,
                "unitaries": len(unitaries),
                "identity_checks": identity_checks,
                "comparisons": comparisons,
                "passed": passed,
            }
        )
    return {
        "experiment": "validate_moments",
        "z_threshold": z_threshold,
        "cases": cases,
        "all_passed": all(case["passed"] for case in cases),
    }
