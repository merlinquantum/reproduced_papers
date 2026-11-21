"""E3 (Fig. 3), E4 (Fig. 4) and E7 (Eqs. 14-16): exact PNM variances versus M.

For every configured observable and mode count ``M`` this computes, exactly,
``mu1``, ``mu1(alpha)^2``, ``mu1(2 alpha)``, ``mu2``, the sample variance, the
circuit variance and their ratio, for the configured Fock input.

Observable families (``observables`` entries):

* ``{"label": "[0,0,6]", "exponents": [6]}`` - a fixed PNM; zero entries in the
  paper's labels only pad the support, so ``[0, 1, 5]`` is ``exponents [1, 5]``.
* ``{"label": "(1,...,1), K=M/2", "growing_support": {"fraction": 0.5,
  "exponent": 1}}`` - the order-growing monomial of paper Eq. 15-16 with
  support ``K = floor(fraction * M)``; ``Var_samp >= (2^K - 1) nu^{2K}`` is
  checked alongside.

Artifacts: ``variance_table.csv`` (one row per observable and M) and the
metrics record with per-observable decay-model fits (``lib/fits.py``).
"""

from __future__ import annotations

import csv
import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import moments
from .fits import compare_decay_models, local_loglog_slopes
from .fock import input_occupation

logger = logging.getLogger(__name__)

TABLE_NAME = "variance_table.csv"


def describe_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Descriptor for ``run_status.json``: exact Haar averages, no samples."""
    return {
        "name": "exact Haar averages (analytic, no sampled data)",
        "splits": {"modes": list(cfg["modes"])},
        "preprocessing": "none",
        "subset": [observable["label"] for observable in cfg["observables"]],
    }


def _exponents_for(observable: dict[str, Any], modes: int) -> tuple[int, ...]:
    if "exponents" in observable:
        exponents = tuple(int(value) for value in observable["exponents"])
    else:
        growing = observable["growing_support"]
        support = int(float(growing["fraction"]) * modes)
        exponents = tuple([int(growing["exponent"])] * support)
    if not exponents or any(value <= 0 for value in exponents):
        raise ValueError(f"Observable {observable['label']} has no positive exponents")
    if len(exponents) > modes:
        raise ValueError(
            f"Observable {observable['label']} needs more than {modes} modes"
        )
    return exponents


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Compute the variance table and decay fits; see the module docstring.

    Parameters
    ----------
    cfg : dict
        Needs ``input`` (pattern name), ``modes`` (list of M), ``observables``
        and ``fit.min_points``.
    run_dir : pathlib.Path
        Destination of ``variance_table.csv``.

    Returns
    -------
    dict
        Metrics record.
    """
    mode_values = [int(value) for value in cfg["modes"]]
    if mode_values != sorted(mode_values):
        raise ValueError("modes must be increasing")
    input_pattern = cfg["input"]
    min_points = int(cfg["fit"]["min_points"])
    logger.info(
        "DATASET_READY | ensemble=exact_haar | input=%s | modes=%s | observables=%d",
        input_pattern,
        mode_values,
        len(cfg["observables"]),
    )
    rows: list[dict[str, Any]] = []
    summaries = []
    for observable in cfg["observables"]:
        logger.info("EVALUATION_STARTED | observable=%s", observable["label"])
        curve = []
        for modes in mode_values:
            occupation = input_occupation(modes, input_pattern)
            photons = sum(occupation)
            exponents = _exponents_for(observable, modes)
            record = moments.monomial_variances(exponents, occupation)
            doubled = tuple(2 * value for value in exponents)
            density = Fraction(photons, modes)
            row = {
                "observable": observable["label"],
                "modes": modes,
                "photons": photons,
                "support": len(exponents),
                "order": sum(exponents),
                "mu1": float(record["mu1"]),
                "mu1_squared": float(record["mu1"] ** 2),
                "mu1_of_square": float(record["mu1_of_square"]),
                "mu2": float(record["mu2"]),
                "sample_variance": float(record["sample_variance"]),
                "circuit_variance": float(record["circuit_variance"]),
                "variance_ratio": float(
                    record["sample_variance"] / record["circuit_variance"]
                )
                if record["circuit_variance"] != 0
                else None,
                "sample_variance_limit": moments.first_moment_limit(
                    doubled, float(density)
                ),
            }
            if "growing_support" in observable:
                support = len(exponents)
                row["eq16_lower_bound"] = float(
                    (2**support - 1) * density ** (2 * support)
                )
            rows.append(row)
            curve.append(row)
            logger.info(
                "EVALUATION_POINT | observable=%s | M=%d | n=%d | var_samp=%.6g | "
                "var_circ=%.6g | ratio=%s",
                observable["label"],
                modes,
                photons,
                row["sample_variance"],
                row["circuit_variance"],
                row["variance_ratio"],
            )
        fit = (
            compare_decay_models(
                [point["modes"] for point in curve],
                [point["circuit_variance"] for point in curve],
                min_points,
            )
            if "exponents" in observable
            else None
        )
        positive = [point for point in curve if point["circuit_variance"] > 0]
        slopes = (
            local_loglog_slopes(
                [point["modes"] for point in positive],
                [point["circuit_variance"] for point in positive],
            )
            if len(positive) >= 2
            else []
        )
        summary = {
            "observable": observable["label"],
            "decay_fit": fit,
            "local_loglog_slopes": slopes,
            "last_local_slope": slopes[-1]["slope"] if slopes else None,
            "circuit_variance_first": curve[0]["circuit_variance"],
            "circuit_variance_last": curve[-1]["circuit_variance"],
            "sample_variance_last": curve[-1]["sample_variance"],
            "sample_variance_limit": curve[-1]["sample_variance_limit"],
            "variance_ratio_first": curve[0]["variance_ratio"],
            "variance_ratio_last": curve[-1]["variance_ratio"],
        }
        if "growing_support" in observable:
            summary["eq16_bound_holds"] = all(
                point["sample_variance"] >= point["eq16_lower_bound"] for point in curve
            )
        summaries.append(summary)
        logger.info(
            "EVALUATION_COMPLETED | observable=%s | preferred_decay=%s | "
            "ratio_first=%s | ratio_last=%s",
            observable["label"],
            None if fit is None else fit["preferred"],
            summary["variance_ratio_first"],
            summary["variance_ratio_last"],
        )
    table_path = run_dir / TABLE_NAME
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("ARTIFACT_WRITTEN | path=%s | rows=%d", TABLE_NAME, len(rows))
    return {
        "experiment": "variance_scaling",
        "input": input_pattern,
        "modes": mode_values,
        "observables": summaries,
        "artifacts": {"table": TABLE_NAME},
    }
