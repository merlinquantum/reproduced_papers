"""E12 (paper Eq. 20, Fig. 6): a quadratic network is a photon-number polynomial.

Part A, exact expansion. For small ``M`` and ``L`` the network is expanded
symbolically (SymPy, exact rationals from the float weights), and the check
records: the maximal total degree (expected ``2^L``), the number of monomials
with non-zero coefficient (expected ``C(2^L + M, M)`` with biases, i.e. every
monomial up to order ``2^L``, constant included), and the largest difference
between the expansion and the network on random photon patterns.

Part B, cost. For larger ``M`` it tabulates the forward-pass multiply-adds
``L M^2 + M`` against the number of expanded terms ``C(2^L + M, M)``, the
quantity a term-wise classical estimator has to enumerate (App. D.4).
"""

from __future__ import annotations

import logging
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import sympy

from .observables import QuadraticNetworkObservable

logger = logging.getLogger(__name__)


def describe_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Descriptor for ``run_status.json``."""
    return {
        "name": "random quadratic networks (synthetic, from seed)",
        "splits": {"expansion_cases": len(cfg["expansion_cases"])},
        "preprocessing": "none",
        "subset": {"cost_modes": cfg["cost_modes"], "cost_layers": cfg["cost_layers"]},
    }


def _symbolic_network(network: QuadraticNetworkObservable, symbols: list[sympy.Symbol]):
    hidden = [sympy.Integer(0) + symbol for symbol in symbols]
    for layer_index, weight in enumerate(network.weights):
        bias = None if network.biases is None else network.biases[layer_index]
        outputs = []
        for row_index, row in enumerate(weight):
            value = sum(
                (sympy.Rational(Fraction(float(w))) * h for w, h in zip(row, hidden)),
                sympy.Integer(0),
            )
            if bias is not None:
                value += sympy.Rational(Fraction(float(bias[row_index])))
            if layer_index < network.hidden_layers:
                value = sympy.expand(value**2)
            outputs.append(value)
        hidden = outputs
    return sympy.Poly(
        sympy.expand(hidden[0] * sympy.Rational(Fraction(network.output_scale))),
        *symbols,
    )


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Run the expansion checks and the cost table; see the module docstring.

    Parameters
    ----------
    cfg : dict
        Needs ``seed``, ``expansion_cases`` (``modes``, ``hidden_layers``,
        ``bias``), ``test_patterns``, ``cost_modes`` and ``cost_layers``.
    run_dir : pathlib.Path
        Unused; results go to the metrics record.

    Returns
    -------
    dict
        Metrics record.
    """
    rng = np.random.default_rng(int(cfg["seed"]))
    logger.info(
        "DATASET_READY | ensemble=random_quadratic_networks | cases=%d",
        len(cfg["expansion_cases"]),
    )
    expansions = []
    for case in cfg["expansion_cases"]:
        modes, layers = int(case["modes"]), int(case["hidden_layers"])
        logger.info("EVALUATION_STARTED | part=expansion | M=%d | L=%d", modes, layers)
        network = QuadraticNetworkObservable.random(
            modes, layers, rng, with_bias=bool(case["bias"])
        )
        symbols = list(sympy.symbols(f"n0:{modes}"))
        polynomial = _symbolic_network(network, symbols)
        terms = polynomial.terms()
        patterns = rng.integers(0, 4, size=(int(cfg["test_patterns"]), modes))
        expanded_values = np.array(
            [float(polynomial.eval(tuple(int(v) for v in row))) for row in patterns]
        )
        network_values = network(patterns)
        record = {
            "modes": modes,
            "hidden_layers": layers,
            "bias": bool(case["bias"]),
            "expected_order": network.polynomial_order,
            "total_degree": int(polynomial.total_degree()),
            "nonzero_terms": len(terms),
            "all_terms_up_to_order": comb(network.polynomial_order + modes, modes),
            "max_abs_difference": float(
                np.max(np.abs(expanded_values - network_values))
            ),
            "max_abs_value": float(np.max(np.abs(network_values))),
        }
        expansions.append(record)
        logger.info(
            "EVALUATION_COMPLETED | part=expansion | M=%d | L=%d | degree=%d | terms=%d/%d | "
            "max_abs_diff=%.3g",
            modes,
            layers,
            record["total_degree"],
            record["nonzero_terms"],
            record["all_terms_up_to_order"],
            record["max_abs_difference"],
        )
    costs = [
        {
            "modes": int(modes),
            "hidden_layers": int(layers),
            "forward_multiply_adds": int(layers) * int(modes) ** 2 + int(modes),
            "expanded_terms": comb(2 ** int(layers) + int(modes), int(modes)),
        }
        for modes in cfg["cost_modes"]
        for layers in cfg["cost_layers"]
    ]
    logger.info("EVALUATION_COMPLETED | part=cost | rows=%d", len(costs))
    return {
        "experiment": "network_polynomial",
        "expansions": expansions,
        "costs": costs,
    }
