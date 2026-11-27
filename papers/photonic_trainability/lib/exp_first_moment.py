"""E1 (Fig. 10): PNM first moment versus total weight |alpha| (paper Thm. 1).

Fig. 10 fixes the support ``K``, spreads the weight as evenly as possible over
the ``K`` modes (``alpha_j`` as close to ``|alpha| / K`` as possible), and plots
``mu1`` at ``M = 50`` (solid, Eq. 9) and the ``M -> infinity`` limit (dotted,
Eq. 10) for several photon densities. The photon number is ``n = nu * M``,
which must be an integer.

The first moment does not depend on the input state beyond its photon number
(the Haar-averaged output is maximally mixed, Theorem 3), so no input is set.
Artifact: ``first_moment_table.csv``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import moments

logger = logging.getLogger(__name__)

TABLE_NAME = "first_moment_table.csv"


def describe_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Descriptor for ``run_status.json``."""
    return {
        "name": "exact Haar averages (analytic, no sampled data)",
        "splits": {
            "weights": list(cfg["weights"]),
            "densities": list(cfg["densities"]),
        },
        "preprocessing": "none",
        "subset": f"M={cfg['modes']}, K={cfg['support']}",
    }


def balanced_exponents(weight: int, support: int) -> tuple[int, ...]:
    """Split ``weight`` over ``support`` modes as evenly as possible."""
    if weight < support:
        raise ValueError(f"Weight {weight} cannot cover {support} modes")
    base, extra = divmod(weight, support)
    return tuple([base + 1] * extra + [base] * (support - extra))


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Tabulate ``mu1`` and its limit; see the module docstring.

    Parameters
    ----------
    cfg : dict
        Needs ``modes``, ``support``, ``weights`` and ``densities``.
    run_dir : pathlib.Path
        Destination of ``first_moment_table.csv``.

    Returns
    -------
    dict
        Metrics record with per-density growth summaries.
    """
    modes, support = int(cfg["modes"]), int(cfg["support"])
    weights = [int(value) for value in cfg["weights"]]
    logger.info(
        "DATASET_READY | ensemble=exact_haar | M=%d | K=%d | weights=%s | densities=%s",
        modes,
        support,
        weights,
        cfg["densities"],
    )
    rows = []
    summaries = []
    for density in cfg["densities"]:
        photons = float(density) * modes
        if not photons.is_integer():
            raise ValueError(f"nu * M = {photons} is not an integer")
        photons = int(photons)
        logger.info("EVALUATION_STARTED | density=%s | photons=%d", density, photons)
        values = []
        for weight in weights:
            exponents = balanced_exponents(weight, support)
            exact = float(moments.first_moment_exponents(exponents, photons, modes))
            limit = moments.first_moment_limit(exponents, float(density))
            rows.append(
                {
                    "density": float(density),
                    "modes": modes,
                    "photons": photons,
                    "weight": weight,
                    "exponents": "-".join(str(value) for value in exponents),
                    "mu1_exact": exact,
                    "mu1_limit": limit,
                    "relative_gap": abs(exact - limit) / limit,
                }
            )
            values.append(exact)
        # Balanced splitting makes successive ratios zig-zag as the extra unit
        # moves between modes, so compare the growth rate over the first and last
        # K weight steps (one full cycle each) instead of requiring monotonicity.
        ratios = [right / left for left, right in zip(values, values[1:])]
        cycle = min(support, len(ratios))
        early_rate = float(np.prod(ratios[:cycle]) ** (1 / cycle))
        late_rate = float(np.prod(ratios[-cycle:]) ** (1 / cycle))
        summary = {
            "density": float(density),
            "mu1_at_min_weight": values[0],
            "mu1_at_max_weight": values[-1],
            "mean_growth_factor_first_cycle": early_rate,
            "mean_growth_factor_last_cycle": late_rate,
            "superexponential": late_rate > early_rate,
        }
        summaries.append(summary)
        logger.info(
            "EVALUATION_COMPLETED | density=%s | mu1_max_weight=%.6g | "
            "superexponential=%s",
            density,
            values[-1],
            summary["superexponential"],
        )
    with (run_dir / TABLE_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("ARTIFACT_WRITTEN | path=%s | rows=%d", TABLE_NAME, len(rows))
    return {
        "experiment": "first_moment",
        "modes": modes,
        "support": support,
        "densities": summaries,
        "artifacts": {"table": TABLE_NAME},
    }
