"""Photonic (MerLin) adaptation of the paper's quantum-walk encodings.

This module is a Phase 4 adaptation, not part of the original reproduction.

Mapping
-------
The single-particle sector of the paper's XY Hamiltonian is ``2A`` (one
``XX + YY`` hop per edge), so the 1-CQRW propagator is ``U(t) = exp(-i 2A t)``.
A passive linear-optical interferometer on N modes applies exactly this
unitary to every photon. One photon injected in mode ``i`` therefore gives the
1-CQRW feature ``|U_ji|^2`` (:func:`photonic_cqrw_features`).

With n >= 2 indistinguishable photons the interferometer realises a
*non-interacting* bosonic walk; the paper's two-particle walk is interacting
(hard-core). The only nonlinearity linear optics offers here is the
measurement: threshold detectors with postselection on n clicks keep the
collision-free outcomes and renormalise them (MerLin
``ComputationSpace.UNBUNCHED``). Number-resolving detection without
postselection keeps every outcome (``ComputationSpace.FOCK``).

Feature
-------
:func:`photonic_walk_covariance` computes the statistic the paper's SRG
experiment uses for the interacting walk
(``lib.srg.two_particle_walk_correlations``): the output occupation covariance
averaged over every collision-free input (one photon per input mode),

    C_ij = mean_s E[n_i n_j | s] - mean_s E[n_i | s] E[n_j | s].

Backends
--------
``"merlin"`` builds a MerLin ``QuantumLayer`` around a Perceval ``Unitary``
component (a graph-dependent interferometer has no CircuitBuilder primitive)
and pushes every input state through it as a batch of one-hot amplitude
vectors. ``"permanent"`` evaluates the same probabilities in closed form from
2x2 permanents; it supports one or two photons and is used for the ZINC
positional encodings, where one MerLin layer per graph and time would dominate
run time. Tests and each ZINC run check the two backends against each other.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

DETECTIONS = ("threshold", "pnr")
PARTICLES = ("bosons", "distinguishable")
BACKENDS = ("merlin", "permanent")


def adjacency_matrix(num_nodes: int, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    """Build a symmetric 0/1 adjacency matrix from an edge list.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes (optical modes).
    edges : Sequence[tuple[int, int]]
        Undirected edges.

    Returns
    -------
    numpy.ndarray
        Adjacency matrix with shape ``(num_nodes, num_nodes)``.
    """
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    return adjacency


def graph_unitary(
    adjacency: np.ndarray, t: float, hamiltonian: str = "xy"
) -> np.ndarray:
    """Interferometer unitary of a graph at evolution time ``t``.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Symmetric adjacency matrix with shape ``(N, N)``.
    t : float
        Evolution time.
    hamiltonian : str
        ``"xy"`` uses ``exp(-i 2A t)``, the paper's single-particle XY
        propagator; ``"adj"`` uses ``exp(-i A t)``. Default value is ``"xy"``.

    Returns
    -------
    numpy.ndarray
        Complex unitary with shape ``(N, N)``; symmetric because ``A`` is.

    Raises
    ------
    ValueError
        If ``hamiltonian`` is unknown.
    """
    if hamiltonian == "xy":
        generator = 2.0 * adjacency
    elif hamiltonian == "adj":
        generator = adjacency
    else:
        raise ValueError(f"unknown hamiltonian: {hamiltonian}")
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    return (eigenvectors * np.exp(-1j * t * eigenvalues)) @ eigenvectors.conj().T


def photonic_cqrw_features(
    adjacency: np.ndarray, k: int, times: Sequence[float]
) -> np.ndarray:
    """Single-photon interferometer walk, identical to the paper's 1-CQRW.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Adjacency matrix with shape ``(N, N)``.
    k : int
        Photon number; only 1 is defined here (the multi-photon statistic is
        :func:`photonic_walk_covariance`).
    times : Sequence[float]
        Evolution times.

    Returns
    -------
    numpy.ndarray
        ``P[t, i, j] = |U(t)_{ji}|^2`` with shape ``(len(times), N, N)``.

    Raises
    ------
    ValueError
        If ``k`` is not 1.
    """
    if k != 1:
        raise ValueError(
            "photonic_cqrw_features is the single-photon walk; use "
            "photonic_walk_covariance for two or more photons"
        )
    return np.stack(
        [(np.abs(graph_unitary(adjacency, float(t))) ** 2).T for t in times], axis=0
    )


def _check_options(
    n_photons: int, detection: str, particles: str, backend: str
) -> None:
    if detection not in DETECTIONS:
        raise ValueError(f"detection must be one of {DETECTIONS}, got {detection!r}")
    if particles not in PARTICLES:
        raise ValueError(f"particles must be one of {PARTICLES}, got {particles!r}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if n_photons < 1:
        raise ValueError("n_photons must be positive")
    if backend == "permanent" and n_photons > 2:
        raise ValueError("the permanent backend supports one or two photons")
    if backend == "merlin" and particles != "bosons":
        raise ValueError(
            "MerLin simulates indistinguishable photons only; use the permanent "
            "backend for distinguishable photons"
        )


def _covariance_from_distribution(
    occupation: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Average occupation covariance over inputs.

    ``occupation`` has shape (outcomes, modes); ``probabilities`` has shape
    (outcomes, inputs) with one column per input state.
    """
    marginals = occupation.T @ probabilities  # E[n_i | s], shape (modes, inputs)
    mean_outcome = probabilities.mean(axis=1)
    second_moment = (occupation.T * mean_outcome) @ occupation
    return second_moment - marginals @ marginals.T / probabilities.shape[1]


def _permanent_distribution(
    unitary: np.ndarray, n_photons: int, detection: str, particles: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form outcome distribution for every collision-free input.

    Returns ``(occupation, probabilities, success)`` where ``success`` is the
    collision-free probability of each input (1 for number-resolving
    detection, which keeps every outcome).
    """
    num_modes = unitary.shape[0]
    if n_photons == 1:
        probabilities = np.abs(unitary) ** 2  # rows: output mode, cols: input mode
        return np.eye(num_modes), probabilities, np.ones(num_modes)
    pairs = np.asarray(list(combinations(range(num_modes), 2)))
    first, second = pairs[:, 0], pairs[:, 1]
    # Rows index the output pair (i, j), columns the input pair (a, b).
    u_ia = unitary[np.ix_(first, first)]
    u_jb = unitary[np.ix_(second, second)]
    u_ib = unitary[np.ix_(first, second)]
    u_ja = unitary[np.ix_(second, first)]
    # ``bunched[m, s]`` is the probability that both photons of input s leave
    # in mode m.
    if particles == "bosons":
        collision_free = np.abs(u_ia * u_jb + u_ib * u_ja) ** 2
        # Amplitude of |2_m> is sqrt(2) U_ma U_mb, so P = 2 |U_ma U_mb|^2.
        bunched = 2.0 * np.abs(unitary[:, first] * unitary[:, second]) ** 2
    else:
        collision_free = (
            np.abs(u_ia) ** 2 * np.abs(u_jb) ** 2
            + np.abs(u_ib) ** 2 * np.abs(u_ja) ** 2
        )
        # Photon 1 and photon 2 both in m: |U_ma|^2 |U_mb|^2.
        bunched = np.abs(unitary[:, first]) ** 2 * np.abs(unitary[:, second]) ** 2
    pair_occupation = np.zeros((len(pairs), num_modes))
    pair_occupation[np.arange(len(pairs)), first] = 1.0
    pair_occupation[np.arange(len(pairs)), second] = 1.0
    success = collision_free.sum(axis=0)
    if detection == "threshold":
        return pair_occupation, collision_free / success, success
    occupation = np.vstack([pair_occupation, 2.0 * np.eye(num_modes)])
    probabilities = np.vstack([collision_free, bunched])
    return occupation, probabilities, np.ones(len(pairs))


def _merlin_probabilities(
    unitary: np.ndarray, n_photons: int, space_name: str
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Push every collision-free input through a MerLin layer of ``unitary``.

    Returns the outcome keys and the probabilities with shape
    (outcomes, inputs), inputs in ``itertools.combinations`` order.
    """
    import merlin as ml
    import perceval as pcvl
    import torch

    num_modes = unitary.shape[0]
    circuit = pcvl.Circuit(num_modes)
    circuit.add(0, pcvl.Unitary(pcvl.Matrix(unitary)))
    layer = ml.QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        input_state=[1] * n_photons + [0] * (num_modes - n_photons),
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=getattr(ml.ComputationSpace, space_name)
        ),
        dtype=torch.float64,
    )
    keys = [tuple(int(count) for count in key) for key in layer.output_keys]
    key_index = {key: index for index, key in enumerate(keys)}
    inputs = [
        tuple(1 if mode in chosen else 0 for mode in range(num_modes))
        for chosen in combinations(range(num_modes), n_photons)
    ]
    # Batched one-hot complex amplitudes (cookbook Pattern C, Form 2): each
    # row is one collision-free input state in the layer's own basis.
    amplitudes = torch.zeros(len(inputs), len(keys), dtype=torch.complex128)
    for row, state in enumerate(inputs):
        amplitudes[row, key_index[state]] = 1.0
    with torch.no_grad():
        probabilities = layer(amplitudes).double().numpy().T
    return keys, probabilities


def _merlin_distribution(
    unitary: np.ndarray, n_photons: int, detection: str, need_success: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MerLin outcome distribution for every collision-free input state."""
    if detection == "pnr":
        keys, probabilities = _merlin_probabilities(unitary, n_photons, "FOCK")
        return (
            np.asarray(keys, dtype=np.float64),
            probabilities,
            np.ones(probabilities.shape[1]),
        )
    # UNBUNCHED is threshold detection postselected on n clicks: the
    # collision-free outcomes, renormalised per input.
    keys, probabilities = _merlin_probabilities(unitary, n_photons, "UNBUNCHED")
    success = np.full(probabilities.shape[1], np.nan)
    if need_success:
        # The postselection success rate is the collision-free mass of the
        # unpostselected (FOCK) distribution.
        fock_keys, fock_probabilities = _merlin_probabilities(
            unitary, n_photons, "FOCK"
        )
        collision_free = np.asarray([max(key) <= 1 for key in fock_keys])
        success = fock_probabilities[collision_free].sum(axis=0)
    return np.asarray(keys, dtype=np.float64), probabilities, success


def photonic_walk_covariance(
    adjacency: np.ndarray,
    t: float,
    n_photons: int = 2,
    detection: str = "threshold",
    particles: str = "bosons",
    backend: str = "merlin",
    return_success: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Occupation covariance of an n-photon walk through the graph interferometer.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Symmetric adjacency matrix with shape ``(N, N)`` (one mode per node).
    t : float
        Evolution time of ``U(t) = exp(-i 2A t)``.
    n_photons : int
        Photons injected one per mode, in every collision-free input. Default
        value is 2.
    detection : str
        ``"threshold"``: threshold detectors, postselected on ``n_photons``
        clicks (MerLin ``UNBUNCHED``). ``"pnr"``: number-resolving detectors,
        no postselection (MerLin ``FOCK``). Default value is ``"threshold"``.
    particles : str
        ``"bosons"`` (indistinguishable photons) or ``"distinguishable"`` (a
        no-interference control, permanent backend only). Default value is
        ``"bosons"``.
    backend : str
        ``"merlin"`` or ``"permanent"`` (closed form, one or two photons).
        Default value is ``"merlin"``.
    return_success : bool
        Also return the mean postselection success rate over inputs (1 for
        number-resolving detection). With the MerLin threshold backend this
        runs an extra FOCK pass. Default value is False.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, float]
        Covariance matrix with shape ``(N, N)``, and the mean success rate when
        ``return_success`` is True.

    Raises
    ------
    ValueError
        If an option is unknown or unsupported by the chosen backend, or the
        graph has fewer modes than photons.
    """
    _check_options(n_photons, detection, particles, backend)
    if adjacency.shape[0] < n_photons:
        raise ValueError("the graph needs at least one mode per photon")
    unitary = graph_unitary(adjacency, float(t))
    if backend == "merlin":
        occupation, probabilities, success = _merlin_distribution(
            unitary, n_photons, detection, need_success=return_success
        )
    else:
        occupation, probabilities, success = _permanent_distribution(
            unitary, n_photons, detection, particles
        )
    covariance = _covariance_from_distribution(occupation, probabilities)
    if return_success:
        return covariance, float(np.mean(success))
    return covariance


def photonic_walk_features(
    adjacency: np.ndarray,
    times: Sequence[float],
    n_photons: int = 2,
    detection: str = "threshold",
    backend: str = "permanent",
) -> np.ndarray:
    """Stack photonic walk covariances over evolution times.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Adjacency matrix with shape ``(N, N)``.
    times : Sequence[float]
        Evolution times, one channel each.
    n_photons : int
        Photon number. Default value is 2.
    detection : str
        ``"threshold"`` or ``"pnr"``. Default value is ``"threshold"``.
    backend : str
        ``"permanent"`` or ``"merlin"``. Default value is ``"permanent"``.

    Returns
    -------
    numpy.ndarray
        Feature tensor with shape ``(len(times), N, N)``.
    """
    return np.stack(
        [
            photonic_walk_covariance(
                adjacency, t, n_photons=n_photons, detection=detection, backend=backend
            )
            for t in times
        ],
        axis=0,
    )


__all__ = [
    "BACKENDS",
    "DETECTIONS",
    "PARTICLES",
    "adjacency_matrix",
    "graph_unitary",
    "photonic_cqrw_features",
    "photonic_walk_covariance",
    "photonic_walk_features",
]
