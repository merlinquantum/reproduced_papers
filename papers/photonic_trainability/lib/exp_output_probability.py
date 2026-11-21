"""E6 (paper Sec. 3.2.4, App. G): trainability of output-probability observables.

For each configured ``(M, input)``:

* Single-outcome projectors ``|s><s|`` (claim C7). The exact second moment of
  every output pattern is computed with the irrep formula and averaged over
  ``s``; the averaged variances are compared with paper Eqs. 21-22 evaluated
  with ``P2(M, n)`` from Mhiri et al. Eq. 40, and the variance ratio with the
  paper's stated scaling ``Theta(|H| / (n + n/M + 1))`` and with Mhiri et al.'s
  saturated-regime ``P2 ~ M/n + 1``. The paper's formulas assume a
  collision-free input, so inputs must have at most one photon per mode.
* Constant-fraction coarse-grained bins ``O_S = sum_{s in S} |s><s|`` with
  ``|S| = |H| / K`` (claim C8). For random bins the exact identity
  ``Var_samp + Var_circ = (1/K)(1 - 1/K)`` (Eq. 25) is checked, and the
  bin-averaged circuit variance is compared with the exact average of Eq. 152
  and the approximation of Eq. 26. A structured bin (``n_0 >= 1``, at least one
  photon in mode 0) is reported as the control the paper says can escape
  concentration. Haar Monte Carlo cross-checks the random-bin circuit variance.

Artifacts: ``projector_table.csv`` and ``bin_table.csv``.
"""

from __future__ import annotations

import csv
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

logger = logging.getLogger(__name__)

PROJECTOR_TABLE = "projector_table.csv"
BIN_TABLE = "bin_table.csv"


def describe_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Descriptor for ``run_status.json``."""
    return {
        "name": "exact Haar averages plus Haar Monte Carlo (synthetic, from seed)",
        "splits": {"monte_carlo": int(cfg["monte_carlo"]["unitaries"])},
        "preprocessing": "none",
        "subset": [case["name"] for case in cfg["cases"]],
    }


def _diagonal_variances(
    pattern_set: list[tuple[int, ...]], occupation: tuple[int, ...]
) -> dict[str, Fraction]:
    photons, modes = sum(occupation), len(occupation)
    diagonal = {pattern: Fraction(1) for pattern in pattern_set}
    mu1 = Fraction(len(pattern_set), fock_dimension(photons, modes))
    mu2 = moments.second_moment_from_g(
        moments.g_values_fock_state(occupation, photons),
        moments.g_values_diagonal(diagonal, photons, modes, photons),
        photons,
        modes,
    )
    # O_S is a projector, so mu1(O_S^2) = mu1(O_S).
    return {
        "mu1": mu1,
        "mu2": mu2,
        "sample_variance": mu1 - mu2,
        "circuit_variance": mu2 - mu1 * mu1,
    }


def _projector_case(occupation: tuple[int, ...]) -> dict[str, Any]:
    photons, modes = sum(occupation), len(occupation)
    dimension = fock_dimension(photons, modes)
    input_g = moments.g_values_fock_state(occupation, photons)
    second_moments = [
        moments.second_moment_from_g(
            input_g,
            moments.g_values_fock_state(tuple(int(v) for v in row), photons),
            photons,
            modes,
        )
        for row in fock_basis(photons, modes)
    ]
    average_mu2 = sum(second_moments, Fraction(0)) / dimension
    mu1 = Fraction(1, dimension)
    collision = moments.collision_probability(modes, photons)
    exact_circuit = average_mu2 - mu1 * mu1
    exact_sample = mu1 - average_mu2
    eq21_circuit = (collision - 1) / dimension**2
    eq22_sample = Fraction(1, dimension) * (1 - collision / dimension)
    ratio = exact_sample / exact_circuit
    return {
        "modes": modes,
        "photons": photons,
        "hilbert_dimension": dimension,
        "collision_probability_P2": float(collision),
        "average_mu2_times_dim_squared": float(average_mu2 * dimension**2),
        "circuit_variance_exact": float(exact_circuit),
        "circuit_variance_eq21": float(eq21_circuit),
        "sample_variance_exact": float(exact_sample),
        "sample_variance_eq22": float(eq22_sample),
        "eq21_matches_exactly": exact_circuit == eq21_circuit,
        "eq22_matches_exactly": exact_sample == eq22_sample,
        "variance_ratio": float(ratio),
        "paper_scaling_dim_over_n_plus_n_over_M_plus_1": dimension
        / (photons + photons / modes + 1),
        "mhiri_scaling_dim_over_M_over_n_plus_1": dimension / (modes / photons + 1),
        "min_mu2_over_patterns_times_dim_squared": float(
            min(second_moments) * dimension**2
        ),
        "max_mu2_over_patterns_times_dim_squared": float(
            max(second_moments) * dimension**2
        ),
    }


def _bin_case(
    occupation: tuple[int, ...],
    fraction_denominator: int,
    bins: int,
    rng: np.random.Generator,
    unitaries: list[np.ndarray],
    batches: int,
) -> list[dict[str, Any]]:
    photons, modes = sum(occupation), len(occupation)
    basis = [tuple(int(v) for v in row) for row in fock_basis(photons, modes)]
    dimension = len(basis)
    if dimension % fraction_denominator:
        raise ValueError(
            f"|H| = {dimension} is not divisible by K = {fraction_denominator}"
        )
    bin_size = dimension // fraction_denominator
    target_total = Fraction(1, fraction_denominator) * (
        1 - Fraction(1, fraction_denominator)
    )
    collision = moments.collision_probability(modes, photons)
    # Eq. 152 (exact average over random bins) with sum_s Var P(s) = (P2 - 1)/|H|.
    eq152 = (
        Fraction(bin_size, dimension)
        * (1 - Fraction(bin_size - 1, dimension - 1))
        * (collision - 1)
        / dimension
    )
    eq26 = target_total * (collision - 1) / dimension
    rows = []
    for bin_index in range(bins):
        members = [
            basis[i] for i in sorted(rng.choice(dimension, bin_size, replace=False))
        ]
        record = _diagonal_variances(members, occupation)
        member_array = np.array(members)

        def indicator(patterns: np.ndarray, member_array=member_array) -> np.ndarray:
            matches = (patterns[:, None, :] == member_array[None, :, :]).all(axis=-1)
            return matches.any(axis=1).astype(np.float64)

        means, second = per_unitary_moments(indicator, occupation, unitaries)
        estimate = moment_estimates(means, second, batches)["circuit_variance"]
        rows.append(
            {
                "bin": f"random_{bin_index}",
                "bin_size": bin_size,
                "K": fraction_denominator,
                "circuit_variance_exact": float(record["circuit_variance"]),
                "sample_variance_exact": float(record["sample_variance"]),
                "total_variance_matches_eq25": record["circuit_variance"]
                + record["sample_variance"]
                == target_total,
                "circuit_variance_monte_carlo": estimate["estimate"],
                "circuit_variance_monte_carlo_se": estimate["standard_error"],
                "variance_ratio": float(
                    record["sample_variance"] / record["circuit_variance"]
                ),
                "eq152_bin_average": float(eq152),
                "eq26_approximation": float(eq26),
            }
        )
    structured = [pattern for pattern in basis if pattern[0] >= 1]
    record = _diagonal_variances(structured, occupation)
    rows.append(
        {
            "bin": "structured n_0>=1",
            "bin_size": len(structured),
            "K": None,
            "circuit_variance_exact": float(record["circuit_variance"]),
            "sample_variance_exact": float(record["sample_variance"]),
            "total_variance_matches_eq25": None,
            "circuit_variance_monte_carlo": None,
            "circuit_variance_monte_carlo_se": None,
            "variance_ratio": float(
                record["sample_variance"] / record["circuit_variance"]
            ),
            "eq152_bin_average": None,
            "eq26_approximation": None,
        }
    )
    return rows


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Compute projector and bin variances; see the module docstring.

    Parameters
    ----------
    cfg : dict
        Needs ``cases`` (``name``, ``modes``, ``input``, ``bin_fraction_K``,
        ``random_bins``), ``monte_carlo`` and ``seed``.
    run_dir : pathlib.Path
        Destination of the two tables.

    Returns
    -------
    dict
        Metrics record.
    """
    rng = np.random.default_rng(int(cfg["seed"]))
    batches = int(cfg["monte_carlo"]["batches"])
    logger.info(
        "DATASET_READY | ensemble=exact_haar_plus_mc | cases=%d | unitaries=%d",
        len(cfg["cases"]),
        int(cfg["monte_carlo"]["unitaries"]),
    )
    projector_rows, bin_rows, summaries = [], [], []
    for case_index, case in enumerate(cfg["cases"]):
        modes = int(case["modes"])
        occupation = input_occupation(modes, case["input"])
        if max(occupation) > 1:
            raise ValueError("Paper Eqs. 21-26 assume a collision-free input")
        logger.info("EVALUATION_STARTED | case=%s", case["name"])
        projector = {"case": case["name"], **_projector_case(occupation)}
        projector_rows.append(projector)
        loader = build_dataloader(
            {**cfg, "modes": modes, "case_index": case_index}, split="monte_carlo"
        )
        unitaries = [unitary for batch in loader for unitary in batch]
        case_bins = [
            {"case": case["name"], "modes": modes, "photons": sum(occupation), **row}
            for row in _bin_case(
                occupation,
                int(case["bin_fraction_K"]),
                int(case["random_bins"]),
                rng,
                unitaries,
                batches,
            )
        ]
        bin_rows.extend(case_bins)
        random_rows = [row for row in case_bins if row["bin"].startswith("random")]
        structured = next(
            row for row in case_bins if row["bin"].startswith("structured")
        )
        summary = {
            "case": case["name"],
            "hilbert_dimension": projector["hilbert_dimension"],
            "projector_ratio": projector["variance_ratio"],
            "projector_eqs21_22_exact": projector["eq21_matches_exactly"]
            and projector["eq22_matches_exactly"],
            "random_bin_mean_circuit_variance": float(
                np.mean([row["circuit_variance_exact"] for row in random_rows])
            ),
            "random_bin_eq152": random_rows[0]["eq152_bin_average"],
            "random_bin_eq25_all_exact": all(
                row["total_variance_matches_eq25"] for row in random_rows
            ),
            "random_bin_max_abs_z_mc": float(
                max(
                    abs(
                        row["circuit_variance_monte_carlo"]
                        - row["circuit_variance_exact"]
                    )
                    / row["circuit_variance_monte_carlo_se"]
                    for row in random_rows
                )
            ),
            "structured_bin_circuit_variance": structured["circuit_variance_exact"],
            "structured_bin_ratio": structured["variance_ratio"],
        }
        summaries.append(summary)
        logger.info(
            "EVALUATION_COMPLETED | case=%s | |H|=%d | projector_ratio=%.6g | "
            "eqs21_22_exact=%s | random_bin_var_circ=%.6g | eq152=%.6g | "
            "structured_var_circ=%.6g",
            case["name"],
            summary["hilbert_dimension"],
            summary["projector_ratio"],
            summary["projector_eqs21_22_exact"],
            summary["random_bin_mean_circuit_variance"],
            summary["random_bin_eq152"],
            summary["structured_bin_circuit_variance"],
        )
    _write_table(run_dir / PROJECTOR_TABLE, projector_rows)
    _write_table(run_dir / BIN_TABLE, bin_rows)
    logger.info("ARTIFACT_WRITTEN | paths=%s,%s", PROJECTOR_TABLE, BIN_TABLE)
    return {
        "experiment": "output_probability",
        "cases": summaries,
        "artifacts": {"projectors": PROJECTOR_TABLE, "bins": BIN_TABLE},
    }
