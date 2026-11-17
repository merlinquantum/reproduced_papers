"""Adaptive Boson Sampling (ABS) circuits of Hoch et al. (2025), platforms A-B3.

Conventions (0-indexed modes; the paper numbers modes 1..m):

- Every crossing drawn in Figs. 3-4 is a Mach-Zehnder interferometer
  ``MZI(theta) = B . diag(e^{i theta}, 1) . B`` with ``B`` a 50:50 beam splitter
  ``[[c, i s], [i s, c]]`` (c = s = 1/sqrt(2)). The paper's "BS splitting ratio
  theta" is this internal phase; rectangles are external phase shifters.
  This convention was selected because it reproduces the platform-A theory kernel
  printed in Fig. 3b (0.298 / 0.966 / 0.331) to 1e-3, while single-BS conventions
  do not (see ``tests/test_circuits.py`` and LOG.md).
- An adaptive circuit is emulated in post-selection: the whole mesh is programmed
  with the angles that outcome ``p`` would trigger, and only events with outcome
  ``p`` on the detected modes are kept (paper, "Experimental apparatus").
- ``state_table[c][k]`` is the residual state when outcome ``c`` is observed while
  the circuit for outcome ``k`` is programmed. The paper's kernel uses the
  diagonal ``c == k``; off-diagonal entries give the alternative kernels of SI
  Note 4.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import unitary_group

from .fock import conditional_state

QUARTER_PI = math.pi / 4

# An operation is ("ps", mode, phase) or ("mzi", top_mode, internal_phase).
Operation = tuple[str, int, float]


def balanced_beam_splitter() -> np.ndarray:
    """Return the symmetric 50:50 beam-splitter matrix used inside every MZI."""
    return np.array([[1.0, 1.0j], [1.0j, 1.0]], dtype=np.complex128) / math.sqrt(2)


def mzi_matrix(internal_phase: float) -> np.ndarray:
    """Return the 2x2 MZI transfer matrix ``B . diag(e^{i theta}, 1) . B``."""
    splitter = balanced_beam_splitter()
    phase = np.diag([np.exp(1j * internal_phase), 1.0])
    return splitter @ phase @ splitter


def unitary_from_operations(
    n_modes: int,
    operations: Sequence[Operation],
    prefix: np.ndarray | None = None,
) -> np.ndarray:
    """Compose phase shifters and MZIs, applied in list order, into one unitary.

    Parameters
    ----------
    n_modes : int
        Number of optical modes.
    operations : sequence of Operation
        ``("ps", mode, phase)`` or ``("mzi", top_mode, theta)``; an MZI acts on
        ``(top_mode, top_mode + 1)``.
    prefix : numpy.ndarray | None
        Fixed unitary applied before the operations (the random ``U_0`` of B2/B3).
        Default value is None (identity).

    Returns
    -------
    numpy.ndarray
        ``(n_modes, n_modes)`` complex unitary.
    """
    unitary = np.eye(n_modes, dtype=np.complex128) if prefix is None else prefix.copy()
    for kind, mode, value in operations:
        step = np.eye(n_modes, dtype=np.complex128)
        if kind == "ps":
            step[mode, mode] = np.exp(1j * value)
        elif kind == "mzi":
            step[np.ix_([mode, mode + 1], [mode, mode + 1])] = mzi_matrix(value)
        else:
            raise ValueError(f"unknown operation kind {kind!r}")
        unitary = step @ unitary
    return unitary


@dataclass(frozen=True)
class AbsPlatform:
    """One ABS feature map: [m, n, d, D] in the paper's notation.

    Attributes
    ----------
    name : str
        Paper label (A, B1, B2, B3).
    n_modes : int
        m.
    input_modes : tuple of int
        Modes receiving one photon each (n = len).
    detected_modes : tuple of int
        Adaptive detectors in the order o_0, o_1, ...
    residual_modes : tuple of int
        Modes carrying the encoded state |psi_p>.
    outcomes : tuple of tuple of int
        Detected-count strings p, in the kernel index order used by the paper.
    operations_for : Callable[[tuple of int], list of Operation]
        Mesh programmed for an outcome string.
    needs_random_u0 : bool
        True for B2/B3, whose fixed U_0 is a random interferometer (unpublished).
    """

    name: str
    n_modes: int
    input_modes: tuple[int, ...]
    detected_modes: tuple[int, ...]
    residual_modes: tuple[int, ...]
    outcomes: tuple[tuple[int, ...], ...]
    operations_for: Callable[[tuple[int, ...]], list[Operation]]
    needs_random_u0: bool

    @property
    def output_dimension(self) -> int:
        """Return d = C(m - k + n - r - 1, n - r)."""
        residual_photons = len(self.input_modes) - sum(self.outcomes[0])
        return math.comb(
            len(self.residual_modes) + residual_photons - 1, residual_photons
        )


def cascaded_angles(outcome: Sequence[int], n_adaptive: int) -> list[float]:
    """Return theta_i = (pi/2) sum_{j<i} o_j for i = 1..n_adaptive (Fig. 3a/3d)."""
    return [(math.pi / 2) * sum(outcome[:i]) for i in range(1, n_adaptive + 1)]


def _platform_a_operations(outcome: tuple[int, ...]) -> list[Operation]:
    # Fig. 3a mesh: U_0 = two MZI columns, then U_1 (theta_1), U_2 (theta_2),
    # U_3 (theta_3). Detectors o_0..o_3 sit on paper modes 1, 2, 3, 6.
    theta_1, theta_2, theta_3 = cascaded_angles(outcome, 3)
    q = QUARTER_PI
    return [
        ("ps", 0, q), ("ps", 2, q), ("ps", 4, q),
        ("mzi", 0, q), ("mzi", 2, q), ("mzi", 4, q),
        ("ps", 1, q), ("ps", 3, q),
        ("mzi", 1, q), ("mzi", 3, q),
        ("ps", 2, q), ("ps", 4, q),
        ("mzi", 2, theta_1), ("mzi", 4, q),
        ("ps", 3, q), ("mzi", 3, theta_2),
        ("ps", 4, q), ("mzi", 4, theta_3),
    ]  # fmt: skip


def _platform_b1_operations(outcome: tuple[int, ...]) -> list[Operation]:
    # Fig. 3d mesh: U_0 = two MZI columns, then five cascaded adaptive blocks.
    # Detectors o_0..o_5 sit on paper modes 1, 2, 3, 4, 5, 8; qubit on 6-7.
    theta = cascaded_angles(outcome, 5)
    q = QUARTER_PI
    return [
        ("ps", 0, q), ("ps", 2, q), ("ps", 4, q), ("ps", 6, q),
        ("mzi", 0, q), ("mzi", 2, q), ("mzi", 4, q), ("mzi", 6, q),
        ("ps", 1, q), ("ps", 3, q), ("ps", 5, q),
        ("mzi", 1, q), ("mzi", 3, q), ("mzi", 5, q),
        ("ps", 2, q), ("ps", 4, q), ("ps", 6, q),
        ("mzi", 2, theta[0]), ("mzi", 4, q), ("mzi", 6, q),
        ("ps", 3, q), ("ps", 5, q),
        ("mzi", 3, theta[1]), ("mzi", 5, q),
        ("ps", 4, q), ("ps", 6, q),
        ("mzi", 4, theta[2]), ("mzi", 6, q),
        ("ps", 5, q), ("mzi", 5, theta[3]),
        ("ps", 6, q), ("mzi", 6, theta[4]),
    ]  # fmt: skip


def pair_outcomes(n_detectors: int, allow_bunching: bool) -> list[tuple[int, ...]]:
    """Return two-photon detection strings in the paper's index order.

    Non-bunched pairs (a < b) come first in lexicographic order, then bunched
    pairs (both photons on detector j) in increasing j. This matches the index
    i = b + sum_{t=1}^{a} (k - 1 - t) of the B2/B3 angle rules (k detectors), which
    runs 1..C(k, 2), and i = C(k, 2) + 1 + j for bunched outcomes (B3).
    """
    outcomes = []
    for first, second in itertools.combinations(range(n_detectors), 2):
        counts = [0] * n_detectors
        counts[first] = counts[second] = 1
        outcomes.append(tuple(counts))
    if allow_bunching:
        for detector in range(n_detectors):
            counts = [0] * n_detectors
            counts[detector] = 2
            outcomes.append(tuple(counts))
    return outcomes


def gaussian_like_index(outcome: Sequence[int]) -> int:
    """Return the 1-based adaptive index i of a B2/B3 outcome (Sec. B2, B3).

    Paper formulas, with (k, j) the detector indices of the two photons and
    k > j: B2 uses i = k + sum_{t=1}^{j} (5 - t) over 6 detectors; B3 uses
    i = k + sum_{t=1}^{j} (4 - t) over 5 detectors and i = 11 + j when both photons
    land on detector j. Reading (k, j) as (second, first) photon is inferred: it is
    the only reading that gives 15 distinct indices.
    """
    n_detectors = len(outcome)
    occupied = [mode for mode, count in enumerate(outcome) for _ in range(count)]
    if len(occupied) != 2:
        raise ValueError(f"expected a two-photon outcome, got {outcome}")
    first, second = occupied
    if first == second:
        return math.comb(n_detectors, 2) + 1 + first
    return second + sum(n_detectors - 1 - t for t in range(1, first + 1))


def _platform_b2_operations(outcome: tuple[int, ...]) -> list[Operation]:
    # Fig. 4a: 2-mode adaptive block on paper modes 2-3 after the random U_0.
    # phi_i = theta_i = i / 5 (radians).
    angle = gaussian_like_index(outcome) / 5
    return [("ps", 1, angle), ("mzi", 1, angle), ("ps", 2, QUARTER_PI)]


def _platform_b3_operations(outcome: tuple[int, ...]) -> list[Operation]:
    # Fig. 4b: 3-mode adaptive block on paper modes 1-3 with two programmable MZIs.
    # Which phase shifters carry phi_i is not stated; we put phi_i before each
    # adaptive MZI on its top mode and keep the others at pi/4 (inferred).
    angle = gaussian_like_index(outcome) / 5
    return [
        ("ps", 2, QUARTER_PI),
        ("ps", 0, angle), ("mzi", 0, angle),
        ("ps", 1, angle), ("mzi", 1, angle),
        ("ps", 0, QUARTER_PI), ("ps", 2, QUARTER_PI),
    ]  # fmt: skip


def get_platform(name: str) -> AbsPlatform:
    """Return the ABS platform definition for a paper label.

    Parameters
    ----------
    name : str
        One of ``"A"``, ``"B1"``, ``"B2"``, ``"B3"``.

    Returns
    -------
    AbsPlatform
        Circuit definition with outcomes in kernel index order.

    Raises
    ------
    ValueError
        For an unknown label.
    """
    if name == "A":
        # Paper index 0, 1, 2 of the Fig. 3b kernel corresponds to the photon on
        # o_3, o_2, o_1 (paper modes 6, 3, 2). This order is inferred from the
        # printed values; o_0 (mode 1) is unreachable from inputs 3 and 6.
        return AbsPlatform(
            name="A",
            n_modes=6,
            input_modes=(2, 5),
            detected_modes=(0, 1, 2, 5),
            residual_modes=(3, 4),
            outcomes=((0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0)),
            operations_for=_platform_a_operations,
            needs_random_u0=False,
        )
    if name == "B1":
        return AbsPlatform(
            name="B1",
            n_modes=8,
            input_modes=(0, 2, 4),
            detected_modes=(0, 1, 2, 3, 4, 7),
            residual_modes=(5, 6),
            outcomes=tuple(pair_outcomes(6, allow_bunching=False)),
            operations_for=_platform_b1_operations,
            needs_random_u0=False,
        )
    if name == "B2":
        return AbsPlatform(
            name="B2",
            n_modes=8,
            input_modes=(2, 3, 4),
            detected_modes=(0, 3, 4, 5, 6, 7),
            residual_modes=(1, 2),
            outcomes=tuple(pair_outcomes(6, allow_bunching=False)),
            operations_for=_platform_b2_operations,
            needs_random_u0=True,
        )
    if name == "B3":
        return AbsPlatform(
            name="B3",
            n_modes=8,
            input_modes=(3, 4, 5),
            detected_modes=(3, 4, 5, 6, 7),
            residual_modes=(0, 1, 2),
            outcomes=tuple(pair_outcomes(5, allow_bunching=True)),
            operations_for=_platform_b3_operations,
            needs_random_u0=True,
        )
    raise ValueError(f"unknown platform {name!r}; expected A, B1, B2 or B3")


def random_u0(n_modes: int, seed: int) -> np.ndarray:
    """Return a Haar-random ``U_0``; the paper's B2/B3 matrices are unpublished."""
    return unitary_group.rvs(n_modes, random_state=np.random.default_rng(seed))


def programmed_unitary(
    platform: AbsPlatform, outcome: tuple[int, ...], u0: np.ndarray | None
) -> np.ndarray:
    """Return the full mesh unitary programmed for ``outcome``.

    Raises
    ------
    ValueError
        If ``u0`` is missing for B2/B3 or given for A/B1.
    """
    if platform.needs_random_u0 and u0 is None:
        raise ValueError(f"platform {platform.name} needs a U_0 matrix")
    if not platform.needs_random_u0 and u0 is not None:
        raise ValueError(f"platform {platform.name} has a fixed mesh; u0 must be None")
    return unitary_from_operations(
        platform.n_modes, platform.operations_for(outcome), prefix=u0
    )


def state_table(
    platform: AbsPlatform, u0: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return residual states for every (observed outcome, programmed circuit) pair.

    Returns
    -------
    states : numpy.ndarray
        Complex array ``(D, D, d)``; ``states[c, k]`` is the normalised residual
        state when outcome ``c`` is observed with the circuit for outcome ``k``.
        Entries with zero post-selection probability are NaN.
    probabilities : numpy.ndarray
        ``(D, D)`` post-selection probabilities of outcome ``c`` under circuit
        ``k``.
    """
    n_outcomes = len(platform.outcomes)
    dimension = platform.output_dimension
    states = np.full((n_outcomes, n_outcomes, dimension), np.nan, dtype=np.complex128)
    probabilities = np.zeros((n_outcomes, n_outcomes))
    for circuit_index, programmed_outcome in enumerate(platform.outcomes):
        unitary = programmed_unitary(platform, programmed_outcome, u0)
        for observed_index, observed_outcome in enumerate(platform.outcomes):
            try:
                state, probability = conditional_state(
                    unitary,
                    platform.input_modes,
                    platform.detected_modes,
                    observed_outcome,
                    platform.residual_modes,
                )
            except ValueError as exc:
                if "zero probability" not in str(exc):
                    raise
                continue
            states[observed_index, circuit_index] = state
            probabilities[observed_index, circuit_index] = probability
    return states, probabilities
