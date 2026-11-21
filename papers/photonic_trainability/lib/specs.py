"""Turn JSON observable specifications into observables and exact moments.

Supported ``observable`` entries in a config case:

* ``{"type": "monomial", "exponents_by_mode": {"0": 2, "1": 1}}``
* ``{"type": "polynomial", "terms": [{"exponents_by_mode": {...},
  "coefficient": 0.5}, ...]}``
* ``{"type": "random_polynomial", "min_order": 1, "max_order": 2,
  "term_count": null, "coefficient_range": [-1, 1]}``
* ``{"type": "projector", "pattern": [1, 0, 1, 0]}``
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from . import moments
from .fock import fock_basis
from .observables import (
    Monomial,
    Polynomial,
    all_monomials,
    evaluate_monomial,
    evaluate_polynomial,
    make_monomial,
    monomial_order,
    random_polynomial,
)


@dataclass
class ObservableCase:
    """An observable ready for both exact and Monte Carlo evaluation.

    Attributes
    ----------
    kind : str
        ``monomial``, ``polynomial`` or ``projector``.
    description : str
        Human-readable label for logs and metrics.
    order : int
        Polynomial order; for a projector, the photon number (all irreps
        contribute).
    evaluate : callable
        Maps ``(D, M)`` patterns to ``(D,)`` values.
    exact_moments : callable
        Maps a Fock input to the exact moment record of
        :func:`moments.monomial_variances`.
    diagonal_values : callable
        Maps ``(photons, modes)`` to ``{pattern: value}`` for brute-force checks.
    """

    kind: str
    description: str
    order: int
    evaluate: Callable[[np.ndarray], np.ndarray]
    exact_moments: Callable[[tuple[int, ...]], dict[str, Fraction]]
    diagonal_values: Callable[[int, int], dict[tuple[int, ...], Fraction]]


def _parse_monomial(exponents_by_mode: Mapping[str, int]) -> Monomial:
    return make_monomial(
        {int(mode): int(value) for mode, value in exponents_by_mode.items()}
    )


def _polynomial_case(polynomial: Polynomial, description: str) -> ObservableCase:
    def diagonal_values(photons: int, modes: int) -> dict[tuple[int, ...], Fraction]:
        basis = fock_basis(photons, modes)
        values: dict[tuple[int, ...], Fraction] = {}
        for row in basis:
            total = Fraction(0)
            for monomial, coefficient in polynomial.items():
                product = 1
                for mode, exponent in monomial:
                    product *= int(row[mode]) ** exponent
                total += Fraction(coefficient) * product
            values[tuple(int(value) for value in row)] = total
        return values

    return ObservableCase(
        kind="polynomial",
        description=description,
        order=max(monomial_order(monomial) for monomial in polynomial),
        evaluate=lambda patterns: evaluate_polynomial(polynomial, patterns),
        exact_moments=lambda occupation: moments.polynomial_variances(
            polynomial, occupation
        ),
        diagonal_values=diagonal_values,
    )


def build_observable(
    spec: Mapping[str, Any], modes: int, rng: np.random.Generator
) -> ObservableCase:
    """Build an :class:`ObservableCase` from a config entry.

    Parameters
    ----------
    spec : Mapping[str, Any]
        Observable specification (see the module docstring).
    modes : int
        Number of modes of the case.
    rng : numpy.random.Generator
        Used only by ``random_polynomial``.

    Returns
    -------
    ObservableCase
        The observable.

    Raises
    ------
    KeyError
        If a required key is missing.
    ValueError
        If the type is unknown or the observable does not fit the modes.
    """
    kind = spec["type"]
    if kind == "monomial":
        monomial = _parse_monomial(spec["exponents_by_mode"])
        if monomial and monomial[-1][0] >= modes:
            raise ValueError(f"Monomial {monomial} uses a mode beyond {modes}")
        exponents = tuple(exponent for _, exponent in monomial)
        case = _polynomial_case({monomial: 1.0}, f"PNM {dict(monomial)}")
        case.kind = "monomial"
        case.exact_moments = lambda occupation: moments.monomial_variances(
            exponents, occupation
        )
        case.evaluate = lambda patterns: evaluate_monomial(monomial, patterns)
        return case
    if kind == "polynomial":
        polynomial = {
            _parse_monomial(term["exponents_by_mode"]): float(term["coefficient"])
            for term in spec["terms"]
        }
        return _polynomial_case(polynomial, f"PNP with {len(polynomial)} terms")
    if kind == "random_polynomial":
        low, high = spec["coefficient_range"]
        candidates = all_monomials(
            modes, int(spec["min_order"]), int(spec["max_order"])
        )
        polynomial = random_polynomial(
            candidates, spec["term_count"], float(low), float(high), rng
        )
        return _polynomial_case(
            polynomial,
            f"random PNP |alpha| in [{spec['min_order']}, {spec['max_order']}], "
            f"{len(polynomial)} terms",
        )
    if kind == "projector":
        pattern = tuple(int(value) for value in spec["pattern"])
        if len(pattern) != modes:
            raise ValueError(f"Projector pattern {pattern} does not fit {modes} modes")

        def evaluate(patterns: np.ndarray) -> np.ndarray:
            return np.all(patterns == np.array(pattern), axis=-1).astype(np.float64)

        return ObservableCase(
            kind="projector",
            description=f"projector |{pattern}><{pattern}|",
            order=sum(pattern),
            evaluate=evaluate,
            exact_moments=lambda occupation: moments.projector_variances(
                pattern, occupation
            ),
            diagonal_values=lambda photons, modes_: {pattern: Fraction(1)},
        )
    raise ValueError(f"Unknown observable type {kind!r}")
