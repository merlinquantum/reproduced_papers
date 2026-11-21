"""E13 / Phase 05 comparison for claim C9: samples to estimate a PNP loss.

Protocol frozen in LOG.md ("Phase 5 Comparison Plan"). For each Haar unitary U
(Fock input ``|1,0,1,0,...>``) and each PNP ``C = sum_alpha c_alpha M_alpha``
drawn from a fixed pool of PNMs:

* Quantum sampling: ``N_Q = Var[C | U] / eps^2`` with the exact single-shot
  variance from the exact output distribution (term covariance matrix, so each
  PNP costs ``c^T Cov c``).
* B1, term-wise Lim-Oh with Neyman allocation:
  ``N_LO = (sum_alpha |c_alpha| sigma_alpha)^2 / eps^2``, with ``sigma_alpha``
  the standard deviation of the Lim-Oh estimator for term ``alpha`` at U
  (``lib/lim_oh.py``), measured from ``lim_oh_samples`` draws; each draw's
  mean is checked against the exact expectation (z-score).
* B2, exact classical evaluation: wall-clock of
  ``correlators.polynomial_expectation`` (permanents of <= order-sized
  submatrices), checked against the exact distribution.

Slopes of log N versus log |A| are fitted per (U, coefficient family) after
averaging over the PNP draws at each |A|.

Artifacts: ``estimator_cases.csv`` (one row per U, family, |A|, draw),
``lim_oh_terms.csv`` (per U and pool term) and the metrics record.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .correlators import polynomial_expectation
from .data import build_dataloader
from .fock import fock_basis, input_occupation
from .lim_oh import monomial_samples
from .observables import all_monomials, evaluate_monomial
from .photonic_sim import output_distribution

logger = logging.getLogger(__name__)

CASES_TABLE = "estimator_cases.csv"
TERMS_TABLE = "lim_oh_terms.csv"
CHUNK = 50_000


def describe_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Descriptor for ``run_status.json``."""
    return {
        "name": "haar_random_interferometers and random PNPs (synthetic, from seed)",
        "splits": {
            "unitaries": int(cfg["unitaries"]),
            "draws_per_size": int(cfg["draws_per_size"]),
        },
        "preprocessing": "none",
        "subset": {"modes": cfg["modes"], "term_counts": cfg["term_counts"]},
    }


def _term_moments(pool, patterns, probabilities):
    """Exact means and covariance matrix of the pool terms under ``probabilities``."""
    count = len(pool)
    means = np.zeros(count)
    second = np.zeros((count, count))
    for start in range(0, len(patterns), CHUNK):
        block = patterns[start : start + CHUNK]
        weights = probabilities[start : start + CHUNK]
        values = np.stack([evaluate_monomial(term, block) for term in pool], axis=1)
        means += weights @ values
        second += (values * weights[:, None]).T @ values
    return means, second - np.outer(means, means)


def neyman_samples(
    coefficients: np.ndarray, sigmas: np.ndarray, epsilon: float
) -> float:
    """Samples for a term-wise estimator with Neyman allocation.

    Allocating ``N_alpha`` proportional to ``|c_alpha| sigma_alpha`` minimises
    ``sum c^2 sigma^2 / N_alpha`` at fixed ``sum N_alpha``, giving
    ``N = (sum |c| sigma)^2 / eps^2`` (never more than equal allocation,
    ``|A| sum c^2 sigma^2 / eps^2``, by Cauchy-Schwarz).
    """
    return float(np.abs(coefficients) @ sigmas) ** 2 / epsilon**2


def _slope(term_counts, values):
    slope, _ = np.polyfit(np.log(term_counts), np.log(values), 1)
    return float(slope)


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run the C9 comparison; see the module docstring.

    Parameters
    ----------
    cfg : dict
        Needs ``seed``, ``modes``, ``input``, ``pool_orders`` ([min, max]),
        ``term_counts``, ``draws_per_size``, ``unitaries``, ``families``
        (name -> coefficient range), ``lim_oh_samples``, ``epsilon``,
        ``unbiased_z_threshold``, ``monte_carlo.batch_size``.
    run_dir : pathlib.Path
        Destination of the two tables.

    Returns
    -------
    dict
        Metrics record with per-unitary slopes and the predeclared decision.
    """
    modes = int(cfg["modes"])
    occupation = input_occupation(modes, cfg["input"])
    photons = sum(occupation)
    pool = all_monomials(modes, *[int(v) for v in cfg["pool_orders"]])
    term_counts = [int(v) for v in cfg["term_counts"]]
    if max(term_counts) > len(pool):
        raise ValueError(f"term_counts exceed the pool of {len(pool)} monomials")
    epsilon = float(cfg["epsilon"])
    samples = int(cfg["lim_oh_samples"])
    loader = build_dataloader(
        {
            **cfg,
            "monte_carlo": {
                "unitaries": int(cfg["unitaries"]),
                "batch_size": int(cfg["unitaries"]),
            },
        },
        split="estimator_scaling",
    )
    unitaries = next(iter(loader))
    basis = fock_basis(photons, modes)
    logger.info(
        "DATASET_READY | unitaries=%d | modes=%d | photons=%d | fock_dim=%d | pool=%d | term_counts=%s",
        len(unitaries),
        modes,
        photons,
        len(basis),
        len(pool),
        term_counts,
    )
    case_rows, term_rows, per_unitary = [], [], []
    for unitary_index, unitary in enumerate(unitaries):
        logger.info("EVALUATION_STARTED | unitary=%d", unitary_index)
        started = time.perf_counter()
        _, probabilities = output_distribution(unitary, occupation, basis)
        exact_means, covariance = _term_moments(pool, basis, probabilities)
        distribution_seconds = time.perf_counter() - started
        lim_oh_rng = np.random.default_rng([int(cfg["seed"]), unitary_index, 1])
        sigmas = np.zeros(len(pool))
        samplers: dict[int, Any] = {}
        lim_oh_started = time.perf_counter()
        max_z = 0.0
        for term_index, term in enumerate(pool):
            draws = monomial_samples(
                unitary, occupation, term, samples, lim_oh_rng, samplers
            )
            sigmas[term_index] = float(draws.std(ddof=1))
            z = (draws.mean() - exact_means[term_index]) / (
                sigmas[term_index] / np.sqrt(samples)
            )
            max_z = max(max_z, abs(float(z)))
            term_rows.append(
                {
                    "unitary": unitary_index,
                    "term": str(dict(term)),
                    "order": sum(e for _, e in term),
                    "exact_mean": float(exact_means[term_index]),
                    "lim_oh_mean": float(draws.mean()),
                    "lim_oh_sigma": sigmas[term_index],
                    "z": float(z),
                    "quantum_single_shot_sigma": float(
                        np.sqrt(max(covariance[term_index, term_index], 0.0))
                    ),
                }
            )
        lim_oh_seconds = time.perf_counter() - lim_oh_started
        seconds_per_sample = lim_oh_seconds / (samples * len(pool))
        draw_rng = np.random.default_rng([int(cfg["seed"]), unitary_index, 2])
        slopes = {}
        for family, (low, high) in cfg["families"].items():
            means_q, means_lo = [], []
            for size in term_counts:
                quantum_counts, lim_oh_counts = [], []
                for draw in range(int(cfg["draws_per_size"])):
                    chosen = draw_rng.choice(len(pool), size=size, replace=False)
                    coefficients = draw_rng.uniform(float(low), float(high), size=size)
                    variance = float(
                        coefficients @ covariance[np.ix_(chosen, chosen)] @ coefficients
                    )
                    n_quantum = variance / epsilon**2
                    n_lim_oh = (
                        float(np.abs(coefficients) @ sigmas[chosen]) ** 2 / epsilon**2
                    )
                    polynomial = {
                        pool[i]: float(c) for i, c in zip(chosen, coefficients)
                    }
                    b2_started = time.perf_counter()
                    exact_classical = polynomial_expectation(
                        unitary, occupation, polynomial
                    )
                    b2_seconds = time.perf_counter() - b2_started
                    exact_quantum = float(coefficients @ exact_means[chosen])
                    case_rows.append(
                        {
                            "unitary": unitary_index,
                            "family": family,
                            "terms": size,
                            "draw": draw,
                            "single_shot_variance": variance,
                            "n_quantum": n_quantum,
                            "n_lim_oh": n_lim_oh,
                            "ratio_lim_oh_to_quantum": n_lim_oh / n_quantum,
                            "lim_oh_classical_seconds": n_lim_oh * seconds_per_sample,
                            "b2_exact_seconds": b2_seconds,
                            "b2_abs_error": abs(exact_classical - exact_quantum),
                        }
                    )
                    quantum_counts.append(n_quantum)
                    lim_oh_counts.append(n_lim_oh)
                means_q.append(float(np.mean(quantum_counts)))
                means_lo.append(float(np.mean(lim_oh_counts)))
            slopes[family] = {
                "slope_quantum": _slope(term_counts, means_q),
                "slope_lim_oh": _slope(term_counts, means_lo),
                "mean_n_quantum_by_terms": dict(zip(term_counts, means_q)),
                "mean_n_lim_oh_by_terms": dict(zip(term_counts, means_lo)),
            }
        per_unitary.append(
            {
                "unitary": unitary_index,
                "distribution_seconds": distribution_seconds,
                "lim_oh_seconds_total": lim_oh_seconds,
                "lim_oh_seconds_per_sample": seconds_per_sample,
                "lim_oh_max_abs_z": max_z,
                "slopes": slopes,
            }
        )
        logger.info(
            "EVALUATION_COMPLETED | unitary=%d | lim_oh_max_abs_z=%.2f | slopes=%s",
            unitary_index,
            max_z,
            {
                f: (round(v["slope_quantum"], 3), round(v["slope_lim_oh"], 3))
                for f, v in slopes.items()
            },
        )
    decisions = {}
    for family in cfg["families"]:
        differences = [
            u["slopes"][family]["slope_lim_oh"] - u["slopes"][family]["slope_quantum"]
            for u in per_unitary
        ]
        mean_difference = float(np.mean(differences))
        if mean_difference >= 0.5 and all(d > 0.3 for d in differences):
            outcome = "SUPPORTED"
        elif mean_difference <= 0.2:
            outcome = "UNSUPPORTED"
        else:
            outcome = "PARTIALLY_SUPPORTED"
        decisions[family] = {
            "slope_differences": differences,
            "mean_difference": mean_difference,
            "std_difference": float(np.std(differences, ddof=1)),
            "slope_quantum_mean": float(
                np.mean([u["slopes"][family]["slope_quantum"] for u in per_unitary])
            ),
            "slope_lim_oh_mean": float(
                np.mean([u["slopes"][family]["slope_lim_oh"] for u in per_unitary])
            ),
            "outcome_vs_lim_oh": outcome,
        }
    for name, rows in ((CASES_TABLE, case_rows), (TERMS_TABLE, term_rows)):
        with (run_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("ARTIFACT_WRITTEN | path=%s | rows=%d", name, len(rows))
    unbiased = max(u["lim_oh_max_abs_z"] for u in per_unitary) < float(
        cfg["unbiased_z_threshold"]
    )
    logger.info(
        "EVALUATION_COMPLETED | summary | lim_oh_unbiased=%s | decisions=%s",
        unbiased,
        {f: d["outcome_vs_lim_oh"] for f, d in decisions.items()},
    )
    return {
        "experiment": "estimator_scaling",
        "epsilon": epsilon,
        "lim_oh_unbiased": unbiased,
        "per_unitary": per_unitary,
        "decisions": decisions,
        "max_b2_abs_error": float(max(row["b2_abs_error"] for row in case_rows)),
        "artifacts": {"cases": CASES_TABLE, "terms": TERMS_TABLE},
    }
