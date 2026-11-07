"""Metric definitions shared by every path.

Kept dependency-free on purpose: the GPU runner imports this and must not pull in
Perceval or MerLin through it. A test asserts that.
"""

from __future__ import annotations


def relative_error(best, optimum):
    """Percent error against the exact optimum, as the paper's tables report it.

    Parameters
    ----------
    best : float
        Best cost found.
    optimum : float
        Exact optimum for the instance.

    Returns
    -------
    float
        ``100 * |best - optimum| / |optimum|``, or NaN when the optimum is zero.
    """
    if optimum == 0:
        return float("nan")
    return 100.0 * abs(best - optimum) / abs(optimum)
