"""Photon-number observables: monomials (PNMs), polynomials (PNPs) and NN observables.

Conventions (paper Sec. 2.1, Eq. 6):

* A PNM ``M_alpha = prod_i n_i^{alpha_i}`` is stored as a sorted tuple of
  ``(mode, exponent)`` pairs with strictly positive exponents. The empty tuple
  is the identity. Modes are 0-based.
* ``support`` is the set of modes with a positive exponent (``K`` in the paper)
  and ``order`` is ``|alpha| = sum_i alpha_i``.
* A PNP ``C = sum_alpha c_alpha M_alpha`` is a ``dict`` mapping a PNM to its
  real coefficient.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations_with_replacement

import numpy as np

Monomial = tuple[tuple[int, int], ...]
Polynomial = dict[Monomial, float]


def make_monomial(exponents_by_mode: Mapping[int, int]) -> Monomial:
    """Build a canonical PNM key from a ``{mode: exponent}`` mapping.

    Parameters
    ----------
    exponents_by_mode : Mapping[int, int]
        Exponent of each mode. Zero exponents are dropped.

    Returns
    -------
    Monomial
        Sorted ``(mode, exponent)`` pairs.

    Raises
    ------
    ValueError
        If a mode index or exponent is negative.
    """
    pairs = []
    for mode, exponent in exponents_by_mode.items():
        if mode < 0 or exponent < 0:
            raise ValueError(f"Invalid mode/exponent pair ({mode}, {exponent})")
        if exponent > 0:
            pairs.append((int(mode), int(exponent)))
    return tuple(sorted(pairs))


def monomial_order(monomial: Monomial) -> int:
    """Return the total order ``|alpha|`` of a PNM."""
    return sum(exponent for _, exponent in monomial)


def monomial_exponents(monomial: Monomial) -> tuple[int, ...]:
    """Return the positive exponents of a PNM in mode order (its support pattern)."""
    return tuple(exponent for _, exponent in monomial)


def multiply_monomials(left: Monomial, right: Monomial) -> Monomial:
    """Return the PNM ``left * right`` (exponents add mode by mode)."""
    exponents: dict[int, int] = dict(left)
    for mode, exponent in right:
        exponents[mode] = exponents.get(mode, 0) + exponent
    return make_monomial(exponents)


def polynomial_order(polynomial: Mapping[Monomial, float]) -> int:
    """Return the largest monomial order present in a PNP."""
    if not polynomial:
        raise ValueError("Empty photon-number polynomial")
    return max(monomial_order(monomial) for monomial in polynomial)


def evaluate_monomial(monomial: Monomial, occupations: np.ndarray) -> np.ndarray:
    """Evaluate a PNM on one or more Fock patterns.

    Parameters
    ----------
    monomial : Monomial
        PNM to evaluate.
    occupations : numpy.ndarray
        Integer photon counts of shape ``(modes,)`` or ``(patterns, modes)``.

    Returns
    -------
    numpy.ndarray
        Observable value per pattern, as float64.
    """
    values = np.ones(occupations.shape[:-1], dtype=np.float64)
    for mode, exponent in monomial:
        values = values * occupations[..., mode].astype(np.float64) ** exponent
    return values


def evaluate_polynomial(
    polynomial: Mapping[Monomial, float], occupations: np.ndarray
) -> np.ndarray:
    """Evaluate a PNP on one or more Fock patterns (see :func:`evaluate_monomial`)."""
    values = np.zeros(occupations.shape[:-1], dtype=np.float64)
    for monomial, coefficient in polynomial.items():
        values = values + coefficient * evaluate_monomial(monomial, occupations)
    return values


def all_monomials(modes: int, min_order: int, max_order: int) -> list[Monomial]:
    """List every PNM on ``modes`` modes with ``min_order <= |alpha| <= max_order``.

    Parameters
    ----------
    modes : int
        Number of modes.
    min_order : int
        Smallest total order, at least 1.
    max_order : int
        Largest total order.

    Returns
    -------
    list of Monomial
        All monomials, in increasing order and then lexicographically.
    """
    if min_order < 1 or max_order < min_order:
        raise ValueError(f"Invalid order range [{min_order}, {max_order}]")
    monomials = []
    for order in range(min_order, max_order + 1):
        for chosen_modes in combinations_with_replacement(range(modes), order):
            exponents: dict[int, int] = {}
            for mode in chosen_modes:
                exponents[mode] = exponents.get(mode, 0) + 1
            monomials.append(make_monomial(exponents))
    return monomials


def random_polynomial(
    candidate_monomials: list[Monomial],
    term_count: int | None,
    coefficient_low: float,
    coefficient_high: float,
    rng: np.random.Generator,
) -> Polynomial:
    """Draw a PNP with i.i.d. uniform coefficients.

    Parameters
    ----------
    candidate_monomials : list of Monomial
        Pool of monomials to draw terms from.
    term_count : int | None
        Number of distinct monomials drawn uniformly without replacement. If
        omitted, every candidate is used (Fig. 5a,b). Default value is None.
    coefficient_low, coefficient_high : float
        Bounds of the uniform coefficient distribution (``[-1, 1]`` in Fig. 5,
        ``[-5, 5]`` in Fig. 9).
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    Polynomial
        Mapping from monomial to coefficient.
    """
    if term_count is None:
        chosen = list(candidate_monomials)
    else:
        if term_count > len(candidate_monomials):
            raise ValueError(
                f"Requested {term_count} terms from {len(candidate_monomials)} candidates"
            )
        indices = rng.choice(len(candidate_monomials), size=term_count, replace=False)
        chosen = [candidate_monomials[index] for index in sorted(indices)]
    coefficients = rng.uniform(coefficient_low, coefficient_high, size=len(chosen))
    return {
        monomial: float(coefficient)
        for monomial, coefficient in zip(chosen, coefficients)
    }


def sample_distinct_monomials(
    modes: int,
    min_order: int,
    max_order: int,
    term_count: int,
    scheme: str,
    rng: np.random.Generator,
) -> list[Monomial]:
    """Draw ``term_count`` distinct PNMs with ``min_order <= |alpha| <= max_order``.

    The paper (Fig. 5c,d) only says the terms are "randomly selected PNMs", so
    the selection scheme is a configurable assumption:

    * ``"uniform_monomial"`` - uniform over all monomials in the order range;
    * ``"uniform_order"`` - order uniform in the range, then a uniform monomial
      of that order;
    * ``"iid_modes"`` - order uniform in the range, then that many modes drawn
      independently and uniformly (repeats raise the exponent);
    * ``"uniform_pattern"`` - exponent pattern (integer partition of an order in
      the range) uniform over all such partitions, placed on distinct random
      modes;
    * ``"order_then_partition"`` - order uniform in the range, then a uniform
      partition of that order, placed on distinct random modes.

    Parameters
    ----------
    modes : int
        Number of modes.
    min_order, max_order : int
        Order range.
    term_count : int
        Number of distinct monomials.
    scheme : str
        One of the schemes above.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    list of Monomial
        Distinct monomials in draw order.
    """
    if scheme == "uniform_monomial":
        candidates = all_monomials(modes, min_order, max_order)
        indices = rng.choice(len(candidates), size=term_count, replace=False)
        return [candidates[index] for index in sorted(indices)]
    by_order = (
        {
            order: all_monomials(modes, order, order)
            for order in range(min_order, max_order + 1)
        }
        if scheme == "uniform_order"
        else None
    )
    if scheme in ("uniform_pattern", "order_then_partition"):
        return _sample_by_partition(
            modes, min_order, max_order, term_count, scheme, rng
        )
    if scheme not in ("uniform_order", "iid_modes"):
        raise ValueError(f"Unknown term sampling scheme {scheme!r}")
    chosen: dict[Monomial, None] = {}
    while len(chosen) < term_count:
        order = int(rng.integers(min_order, max_order + 1))
        if by_order is not None:
            pool = by_order[order]
            monomial = pool[int(rng.integers(len(pool)))]
        else:
            exponents: dict[int, int] = {}
            for mode in rng.integers(0, modes, size=order):
                exponents[int(mode)] = exponents.get(int(mode), 0) + 1
            monomial = make_monomial(exponents)
        chosen.setdefault(monomial, None)
    return list(chosen)


def integer_partitions(total: int) -> list[tuple[int, ...]]:
    """All partitions of ``total`` as non-increasing tuples."""
    if total == 0:
        return [()]
    result = []

    def extend(remaining: int, largest: int, prefix: tuple[int, ...]) -> None:
        if remaining == 0:
            result.append(prefix)
            return
        for part in range(min(remaining, largest), 0, -1):
            extend(remaining - part, part, (*prefix, part))

    extend(total, total, ())
    return result


def _sample_by_partition(
    modes: int,
    min_order: int,
    max_order: int,
    term_count: int,
    scheme: str,
    rng: np.random.Generator,
) -> list[Monomial]:
    by_order = {
        order: integer_partitions(order) for order in range(min_order, max_order + 1)
    }
    every_pattern = [pattern for patterns in by_order.values() for pattern in patterns]
    chosen: dict[Monomial, None] = {}
    while len(chosen) < term_count:
        if scheme == "uniform_pattern":
            pattern = every_pattern[int(rng.integers(len(every_pattern)))]
        else:
            patterns = by_order[int(rng.integers(min_order, max_order + 1))]
            pattern = patterns[int(rng.integers(len(patterns)))]
        if len(pattern) > modes:
            continue
        support = rng.choice(modes, size=len(pattern), replace=False)
        chosen.setdefault(
            make_monomial(
                {int(mode): exponent for mode, exponent in zip(support, pattern)}
            ),
            None,
        )
    return list(chosen)


def single_mode_power(mode: int, exponent: int) -> Monomial:
    """Return the single-mode PNM ``n_mode^exponent`` (Fig. 4 observables)."""
    return make_monomial({mode: exponent})


def monomials_from_exponents(
    exponent_patterns: Iterable[Iterable[int]],
) -> list[Monomial]:
    """Place exponent patterns such as ``[0, 2, 4]`` on modes ``0, 1, 2, ...``.

    Used for the Fig. 3 family ``alpha = [0, 0, 6], [0, 1, 5], ...``.
    """
    return [
        make_monomial(dict(enumerate(int(value) for value in pattern)))
        for pattern in exponent_patterns
    ]


class QuadraticNetworkObservable:
    """Neural-network observable with quadratic activations (paper Sec. 3.2.3).

    ``L`` hidden layers, each a linear map followed by a squaring activation,
    then a linear read-out to one neuron. The network is a photon-number
    polynomial of order ``2^L`` (paper Eq. 20) and costs ``O(L M^2)`` per
    sample. With biases it contains every monomial up to that order.

    The paper's wording ("L linear layers, each interleaved with a quadratic
    activation, whose final layer has a single output neuron") is ambiguous about
    the read-out. A squared output would make the observable non-negative, but
    Fig. 7 reports negative losses, and its "3-layer network" has order 8; three
    squaring hidden layers plus a linear read-out satisfy both, so that is the
    reading used here (LOG.md, Reproducibility Assessment).

    Parameters
    ----------
    weights : list of numpy.ndarray
        ``L + 1`` matrices: ``L`` hidden layers, then a one-row read-out.
    biases : list of numpy.ndarray | None
        One bias vector per matrix, or ``None``. Default value is None.
    output_scale : float
        Factor applied to the read-out (see :meth:`normalised`). Default value
        is 1.0.
    """

    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray] | None = None,
        output_scale: float = 1.0,
    ) -> None:
        if len(weights) < 2 or weights[-1].shape[0] != 1:
            raise ValueError("Need at least one hidden layer and a one-row read-out")
        if biases is not None and len(biases) != len(weights):
            raise ValueError("Need one bias vector per layer")
        self.weights = [np.asarray(weight, dtype=np.float64) for weight in weights]
        self.biases = (
            None
            if biases is None
            else [np.asarray(bias, dtype=np.float64) for bias in biases]
        )
        self.output_scale = float(output_scale)

    @property
    def hidden_layers(self) -> int:
        """Number of squaring layers ``L``."""
        return len(self.weights) - 1

    @property
    def polynomial_order(self) -> int:
        """Total order ``2^L`` of the equivalent photon-number polynomial."""
        return 2**self.hidden_layers

    def __call__(self, occupations: np.ndarray) -> np.ndarray:
        """Evaluate the network on photon-count patterns of shape ``(..., M)``."""
        hidden = np.asarray(occupations, dtype=np.float64)
        for layer_index, weight in enumerate(self.weights):
            hidden = hidden @ weight.T
            if self.biases is not None:
                hidden = hidden + self.biases[layer_index]
            if layer_index < self.hidden_layers:
                hidden = hidden**2
        return self.output_scale * hidden[..., 0]

    def normalised(self, patterns: np.ndarray) -> QuadraticNetworkObservable:
        """Copy rescaled to unit standard deviation over ``patterns``, uniformly weighted.

        With ``patterns`` the full n-photon basis this is the standard deviation
        under the Haar-averaged (maximally mixed) output state, so learning rates
        transfer between random networks. This rescaling is an assumption; the
        paper does not state an output scale.
        """
        values = self(patterns) / self.output_scale
        spread = float(np.std(values))
        if spread == 0:
            raise ValueError("Network output is constant on the given patterns")
        return QuadraticNetworkObservable(self.weights, self.biases, 1.0 / spread)

    @classmethod
    def random(
        cls,
        modes: int,
        hidden_layers: int,
        rng: np.random.Generator,
        hidden_width: int | None = None,
        with_bias: bool = True,
    ) -> QuadraticNetworkObservable:
        """Draw a network with Gaussian weights and biases of variance ``1 / fan_in``.

        The initialisation is an assumption: the paper only says the network is
        "randomly initialised" (Sec. 4.1).

        Parameters
        ----------
        modes : int
            Input width ``M``.
        hidden_layers : int
            Number of squaring layers ``L``.
        rng : numpy.random.Generator
            Source of randomness.
        hidden_width : int | None
            Hidden width; ``M`` if omitted (the paper's ``O(L M^2)`` cost assumes
            ``M x M`` layers). Default value is None.
        with_bias : bool
            Include bias vectors. Default value is True.

        Returns
        -------
        QuadraticNetworkObservable
            The random network.
        """
        width = modes if hidden_width is None else hidden_width
        sizes = [modes] + [width] * hidden_layers + [1]
        weights = [
            rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=(fan_out, fan_in))
            for fan_in, fan_out in zip(sizes[:-1], sizes[1:])
        ]
        biases = (
            [
                rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=fan_out)
                for fan_in, fan_out in zip(sizes[:-1], sizes[1:])
            ]
            if with_bias
            else None
        )
        return cls(weights, biases)
