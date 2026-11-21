"""Exact combinatorial helpers for Haar moments of linear-optical circuits.

Everything here works on Python integers or :class:`fractions.Fraction`, because
the irrep-norm formula of Mhiri et al. (arXiv:2604.14323, Corollary 1) is an
alternating sum that cancels catastrophically in floating point once the photon
number reaches a few tens.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from math import comb, factorial


@cache
def stirling2(order: int, parts: int) -> int:
    """Stirling number of the second kind ``S(order, parts)``.

    Parameters
    ----------
    order : int
        Size of the set being partitioned.
    parts : int
        Number of non-empty blocks.

    Returns
    -------
    int
        Number of partitions of ``order`` labelled items into ``parts`` blocks.
    """
    if order == parts:
        return 1
    if parts == 0 or parts > order:
        return 0
    return parts * stirling2(order - 1, parts) + stirling2(order - 1, parts - 1)


def rising_factorial(base: int, length: int) -> int:
    """Pochhammer symbol ``(base)_length = base (base + 1) ... (base + length - 1)``.

    Parameters
    ----------
    base : int
        First factor.
    length : int
        Number of factors; ``0`` gives ``1``.

    Returns
    -------
    int
        The rising factorial.
    """
    if length < 0:
        raise ValueError(f"Pochhammer length must be non-negative, got {length}")
    result = 1
    for offset in range(length):
        result *= base + offset
    return result


def fock_dimension(photons: int, modes: int) -> int:
    """Dimension ``C(n + M - 1, n)`` of the n-photon, M-mode Fock sector.

    Parameters
    ----------
    photons : int
        Total photon number ``n``.
    modes : int
        Number of modes ``M``.

    Returns
    -------
    int
        Number of Fock states with exactly ``photons`` photons.
    """
    return comb(photons + modes - 1, photons)


def compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield every tuple of ``parts`` non-negative integers summing to ``total``.

    The order is lexicographic in the first entry, then the next, and so on.
    For ``parts == 0`` the only composition of ``0`` is the empty tuple.

    Parameters
    ----------
    total : int
        Required sum.
    parts : int
        Tuple length.

    Yields
    ------
    tuple of int
        One composition at a time.
    """
    if parts == 0:
        if total == 0:
            yield ()
        return
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first, *rest)


def bounded_compositions(
    total: int, upper_bounds: tuple[int, ...]
) -> Iterator[tuple[int, ...]]:
    """Yield compositions of ``total`` with entry ``i`` at most ``upper_bounds[i]``.

    Parameters
    ----------
    total : int
        Required sum.
    upper_bounds : tuple of int
        Per-entry inclusive upper bounds.

    Yields
    ------
    tuple of int
        One bounded composition at a time.
    """
    if not upper_bounds:
        if total == 0:
            yield ()
        return
    head_bound, *tail_bounds = upper_bounds
    tail_capacity = sum(tail_bounds)
    for first in range(max(0, total - tail_capacity), min(head_bound, total) + 1):
        for rest in bounded_compositions(total - first, tuple(tail_bounds)):
            yield (first, *rest)


def falling_factorial(base: int, length: int) -> int:
    """Falling factorial ``base (base - 1) ... (base - length + 1)``.

    Parameters
    ----------
    base : int
        First factor.
    length : int
        Number of factors.

    Returns
    -------
    int
        The falling factorial; ``0`` when ``length > base >= 0``.
    """
    if length > base:
        return 0
    return factorial(base) // factorial(base - length)
