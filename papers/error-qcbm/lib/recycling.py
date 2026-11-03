"""Adapt native MerLin threshold counts to Perceval's photon recycling API.

Only input validation, SDK conversion and fixed-bin alignment live here.
Perceval performs the sector expansion, fitting, clipping and normalization.
Fewer clicks can also arise from collisions; they are not identified losses.
"""

import warnings
from collections.abc import Mapping
from importlib.metadata import version
from numbers import Integral

import numpy as np
import perceval as pcvl

from .data import collision_free_states


def recycling_provenance() -> dict:
    """Return the SDK estimator identity used in reports and resume signatures.

    Returns
    -------
    dict
        Official entry point and installed Perceval distribution version.
    """
    return {
        "implementation": "perceval.photon_recycling",
        "perceval_version": version("perceval-quandela"),
    }


def recycling_mitigation(
    counts: Mapping[tuple[int, ...], int], modes: int, photons: int
) -> dict:
    """Apply official Perceval recycling to one complete native count ledger.

    Parameters
    ----------
    counts : mapping of tuple of int to int
        Binary threshold occupations and nonnegative integer initial-shot counts.
        All click sectors are retained in the ledger passed to Perceval.
    modes : int
        Number of optical modes.
    photons : int
        Ideal input photon count, at least two. The standalone Perceval function
        accepts the two-photon smoke; its automatic Computer layer requires three.

    Returns
    -------
    dict
        Fixed ``states``, SDK ``probabilities``, ``postselected`` probabilities,
        measured ``sector_counts`` for zero/one/two missing clicks, unused
        ``ignored_count``, and SDK provenance/warnings in ``diagnostics``.
        Perceval does not expose pre-clipping values or the fitted decay rate.

    Raises
    ------
    ValueError
        If counts are invalid or a required sector has no positive counts.
        No extra shots, smoothing or replacement estimator is used.
    RuntimeError
        If the SDK returns an invalid or incompatible distribution.
    """
    states = collision_free_states(modes, photons)
    if photons < 2:
        raise ValueError("Recycling requires at least two ideal photons")
    sector_counts = [0, 0, 0]
    ignored_count = 0
    sdk_counts = pcvl.BSCount()
    for state, count in counts.items():
        if len(state) != modes or any(
            not isinstance(value, Integral) or value not in (0, 1) for value in state
        ):
            raise ValueError(
                "Recycling requires binary threshold states of length modes"
            )
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("Recycling counts must be nonnegative integers")
        missing_clicks = photons - sum(state)
        if missing_clicks < 0:
            raise ValueError("Observed click count exceeds the ideal photon count")
        if missing_clicks > 2:
            ignored_count += int(count)
        else:
            sector_counts[missing_clicks] += int(count)
        if count:
            sdk_counts[pcvl.BasicState([int(value) for value in state])] = int(count)
    if any(count == 0 for count in sector_counts):
        raise ValueError(
            "Recycling needs nonempty zero/one/two missing-click sectors; "
            f"observed sector_counts={sector_counts}, ignored_count={ignored_count}"
        )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        mitigated = pcvl.photon_recycling(sdk_counts, ideal_photon_count=photons)
    state_support = set(states)
    if any(tuple(state) not in state_support for state in mitigated):
        raise RuntimeError(
            "Perceval recycling returned states outside the target support"
        )
    # SDK iteration order is not the paper's fixed lexicographic bin order.
    probabilities = np.array(
        [mitigated.get(pcvl.BasicState(list(state)), 0.0) for state in states],
        dtype=np.float64,
    )
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-10, rtol=0)
    ):
        raise RuntimeError("Perceval recycling returned invalid probabilities")
    postselected = np.array([counts.get(state, 0) for state in states], dtype=float)
    postselected /= postselected.sum()
    return {
        "states": states,
        "probabilities": probabilities,
        "postselected": postselected,
        "sector_counts": sector_counts,
        "ignored_count": ignored_count,
        "diagnostics": {
            **recycling_provenance(),
            "fit_warnings": [str(item.message) for item in captured],
        },
    }
