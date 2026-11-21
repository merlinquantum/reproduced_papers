"""Fock-space bookkeeping: basis enumeration and the paper's input states."""

from __future__ import annotations

import numpy as np

from .combinatorics import compositions


def fock_basis(photons: int, modes: int) -> np.ndarray:
    """Enumerate the n-photon, M-mode Fock basis ``Phi_M^n`` (paper Eq. 4).

    Parameters
    ----------
    photons : int
        Total photon number ``n``.
    modes : int
        Number of modes ``M``.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(C(n + M - 1, n), M)``, one pattern per row,
        in lexicographic order of the first mode, then the second, and so on.
    """
    return np.array(list(compositions(photons, modes)), dtype=np.int64).reshape(
        -1, modes
    )


def input_occupation(modes: int, pattern: str | list[int]) -> tuple[int, ...]:
    """Return a Fock input state as an occupation tuple.

    Parameters
    ----------
    modes : int
        Number of modes ``M``.
    pattern : str | list of int
        ``"alternating"`` gives ``|1, 0, 1, 0, ...>`` (photon density 0.5 for
        even ``M``; paper Fig. 4 and Sec. 4.1), ``"all_ones"`` gives
        ``|1, ..., 1>`` (density 1, inferred for Fig. 3c,d), ``"all_twos"`` gives
        ``|2, ..., 2>`` (density 2, the oversaturated regime of Eq. 15), and an
        explicit list is used as given.

    Returns
    -------
    tuple of int
        Photon count per mode.

    Raises
    ------
    ValueError
        If the pattern is unknown or an explicit list has the wrong length.
    """
    if pattern == "alternating":
        return tuple(1 if mode % 2 == 0 else 0 for mode in range(modes))
    if pattern == "all_ones":
        return tuple(1 for _ in range(modes))
    if pattern == "all_twos":
        return tuple(2 for _ in range(modes))
    if isinstance(pattern, list):
        if len(pattern) != modes or any(count < 0 for count in pattern):
            raise ValueError(f"Explicit input {pattern} does not fit {modes} modes")
        return tuple(int(count) for count in pattern)
    raise ValueError(f"Unknown input pattern {pattern!r}")
