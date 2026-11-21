"""E5 (Fig. 5): exact sample and circuit variances of random photon-number polynomials.

For each mode count ``M`` the experiment draws ``instances`` random PNPs with
i.i.d. uniform coefficients and computes their exact Haar variances with
``lib/pnp_moments.py``. Term sets (``terms`` config block):

* ``{"mode": "all", "min_order": 1, "max_order": 2}`` - every PNM in the order
  range (Fig. 5a,b, ``|A| ~ M^2 / 2``);
* ``{"mode": "sampled", "min_order": 1, "max_order": 4, "terms_per_mode": 5,
  "scheme": "uniform_monomial"}`` - ``|A| = 5M`` distinct PNMs (Fig. 5c,d); the
  selection scheme is an assumption, see
  :func:`observables.sample_distinct_monomials`.

The coefficient range defaults to the paper's ``[-1, 1]``; ``[0, 1]`` gives the
positive-coefficient contrast of paper Eq. 17 (no off-diagonal cancellation).
Artifact: ``pnp_variance_table.csv`` with one row per instance.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import pnp_moments
from .fock import input_occupation
from .observables import all_monomials, random_polynomial, sample_distinct_monomials

logger = logging.getLogger(__name__)

TABLE_NAME = "pnp_variance_table.csv"


def describe_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Descriptor for ``run_status.json``."""
    return {
        "name": "random photon-number polynomials (synthetic, from seed); exact Haar moments",
        "splits": {
            "modes": list(cfg["modes"]),
            "instances_per_M": int(cfg["instances"]),
        },
        "preprocessing": "none",
        "subset": {
            "terms": cfg["terms"],
            "coefficient_range": cfg["coefficient_range"],
        },
    }


def _draw_polynomial(cfg: dict[str, Any], modes: int, rng: np.random.Generator):
    terms = cfg["terms"]
    low, high = (float(value) for value in cfg["coefficient_range"])
    if terms["mode"] == "all":
        candidates = all_monomials(
            modes, int(terms["min_order"]), int(terms["max_order"])
        )
        return random_polynomial(candidates, None, low, high, rng)
    if terms["mode"] == "sampled":
        monomials = sample_distinct_monomials(
            modes,
            int(terms["min_order"]),
            int(terms["max_order"]),
            int(terms["terms_per_mode"]) * modes,
            terms["scheme"],
            rng,
        )
        coefficients = rng.uniform(low, high, size=len(monomials))
        return {monomial: float(c) for monomial, c in zip(monomials, coefficients)}
    raise ValueError(f"Unknown term mode {terms['mode']!r}")


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Tabulate exact PNP variances; see the module docstring.

    Parameters
    ----------
    cfg : dict
        Needs ``seed``, ``input``, ``modes``, ``instances``, ``terms`` and
        ``coefficient_range``.
    run_dir : pathlib.Path
        Destination of ``pnp_variance_table.csv``.

    Returns
    -------
    dict
        Per-M averages and spreads.
    """
    instances = int(cfg["instances"])
    mode_values = [int(value) for value in cfg["modes"]]
    logger.info(
        "DATASET_READY | ensemble=random_pnp | input=%s | modes=%s | instances=%d | terms=%s",
        cfg["input"],
        mode_values,
        instances,
        cfg["terms"],
    )
    rows, summaries = [], []
    for modes in mode_values:
        # One independent stream per M keeps instances reproducible when the
        # M grid changes.
        rng = np.random.default_rng([int(cfg["seed"]), modes])
        occupation = input_occupation(modes, cfg["input"])
        logger.info("EVALUATION_STARTED | M=%d | photons=%d", modes, sum(occupation))
        started = time.perf_counter()
        sample_values, circuit_values, term_counts = [], [], []
        for instance in range(instances):
            polynomial = _draw_polynomial(cfg, modes, rng)
            record = pnp_moments.polynomial_variances(polynomial, occupation)
            row = {
                "modes": modes,
                "photons": sum(occupation),
                "instance": instance,
                "terms": len(polynomial),
                "max_order": max(sum(e for _, e in m) for m in polynomial),
                "mu1": float(record["mu1"]),
                "sample_variance": float(record["sample_variance"]),
                "circuit_variance": float(record["circuit_variance"]),
                "variance_ratio": float(
                    record["sample_variance"] / record["circuit_variance"]
                ),
            }
            rows.append(row)
            sample_values.append(row["sample_variance"])
            circuit_values.append(row["circuit_variance"])
            term_counts.append(row["terms"])
        summary = {
            "modes": modes,
            "terms_mean": float(np.mean(term_counts)),
            "sample_variance_mean": float(np.mean(sample_values)),
            "sample_variance_std": float(np.std(sample_values, ddof=1)),
            "circuit_variance_mean": float(np.mean(circuit_values)),
            "circuit_variance_std": float(np.std(circuit_values, ddof=1)),
            "ratio_of_means": float(np.mean(sample_values) / np.mean(circuit_values)),
            "seconds": time.perf_counter() - started,
        }
        summaries.append(summary)
        logger.info(
            "EVALUATION_COMPLETED | M=%d | terms=%.0f | var_samp_mean=%.6g | "
            "var_circ_mean=%.6g | seconds=%.1f",
            modes,
            summary["terms_mean"],
            summary["sample_variance_mean"],
            summary["circuit_variance_mean"],
            summary["seconds"],
        )
    with (run_dir / TABLE_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("ARTIFACT_WRITTEN | path=%s | rows=%d", TABLE_NAME, len(rows))
    return {
        "experiment": "pnp_variances",
        "input": cfg["input"],
        "terms": cfg["terms"],
        "coefficient_range": cfg["coefficient_range"],
        "per_modes": summaries,
        "artifacts": {"table": TABLE_NAME},
    }
