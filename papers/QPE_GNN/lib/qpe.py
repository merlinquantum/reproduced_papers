"""Quantum positional encodings (QPEs) for graphs.

Implements the three encoding families from Thabet et al., "Quantum Positional
Encodings for Graph Neural Networks" (arXiv:2406.06547):

- Relative random-walk probabilities (RRWP) — classical baseline (Ma et al. 2023).
- k-particle continuous-time quantum random walk (k-CQRW), computed by
  diagonalizing the XY hamiltonian restricted to the k-particle subspace.
- k-particle quantum-inspired random walk (k-QiRW), computed by iterating a
  row-normalised power of the same XY hamiltonian.
- Eigenvectors of the Ising-correlation matrix on the ground state.

The XY hamiltonian H_XY = sum_{(i,j) in E} X_i X_j + Y_i Y_j preserves the
k-particle (Hamming weight = k) subspace. Restricted to that subspace its
matrix elements are 2 * A_{ij} between single-hop k-tuples — i.e. the adjacency
matrix of the "k-occupation graph". For k=1 this reduces to 2A on the original
graph; for k=2 it is the adjacency matrix of the line-graph-like 2-particle
hopping graph (one walker moves to a neighbour, the other stays).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Classical reference: random-walk probabilities (RRWP, Ma et al. 2023).
# ---------------------------------------------------------------------------


def adjacency_from_edges(
    num_nodes: int, edges: Sequence[tuple[int, int]]
) -> np.ndarray:
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def rrwp(A: np.ndarray, K: int) -> np.ndarray:
    """RRWP tensor P[k, i, j] = (D^{-1} A)^k for k = 0..K-1."""
    deg = A.sum(axis=1)
    deg_inv = np.divide(
        1.0,
        deg,
        out=np.zeros_like(deg, dtype=np.float64),
        where=deg > 0,
    )
    M = deg_inv[:, None] * A  # row-normalised transition matrix
    out = np.zeros((K, A.shape[0], A.shape[1]), dtype=np.float64)
    out[0] = np.eye(A.shape[0])
    cur = np.eye(A.shape[0])
    for k in range(1, K):
        cur = cur @ M
        out[k] = cur
    return out


# ---------------------------------------------------------------------------
# k-particle subspace bookkeeping.
# ---------------------------------------------------------------------------


def k_particle_basis(num_nodes: int, k: int) -> list[tuple[int, ...]]:
    """Ordered basis of the k-particle (Hamming weight k) subspace.

    Each element is a tuple of k strictly increasing integers naming the
    occupied modes.
    """
    return list(combinations(range(num_nodes), k))


def xy_in_k_subspace(A: np.ndarray, k: int) -> np.ndarray:
    """Matrix of H_XY restricted to the k-particle subspace.

    The XY hamiltonian acts as a hopping term: it moves a single walker between
    neighbouring modes. The resulting matrix is the adjacency matrix of the
    "k-occupation" graph (Henry et al. 2021).
    """
    basis = k_particle_basis(A.shape[0], k)
    index = {b: i for i, b in enumerate(basis)}
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=np.float64)
    for a, state_a in enumerate(basis):
        occ = set(state_a)
        for site in state_a:
            for nbr in np.nonzero(A[site])[0]:
                if nbr in occ:
                    continue
                new = tuple(sorted((occ - {site}) | {int(nbr)}))
                b = index[new]
                # Each XX+YY term contributes a factor of 2 between
                # configurations that differ by a single hop.
                H[a, b] += 2.0
    return H


# ---------------------------------------------------------------------------
# Continuous-time quantum random walk (CQRW).
# ---------------------------------------------------------------------------


def cqrw_features(
    A: np.ndarray,
    k: int,
    times: Sequence[float],
    initial: str = "localised",
) -> np.ndarray:
    """k-CQRW feature tensor P[t, i, j] from Eq. 4 / Sec. 3.2.2 of the paper.

    Parameters
    ----------
    A : (N, N) adjacency matrix of the graph.
    k : number of walkers (1 or 2).
    times : evolution times t_1, ..., t_K (in units of 1/||H||).
    initial : "localised" → for k=1 P[t, i, j] = |<j|e^{-iH_1^XY t}|i>|^2.
              For k=2, "edges" uses the edge-supported uniform superposition
              described after Eq. 4. "uniform" uses the pair-uniform state.

    Returns
    -------
    P : (K, N, N) tensor of probabilities. For k>=2 we collapse the
        (i, j) -> 2-particle configuration mapping to an (N, N) pair tensor,
        symmetric in (i, j).
    """
    N = A.shape[0]
    H = xy_in_k_subspace(A, k)
    eigvals, eigvecs = np.linalg.eigh(H)

    K = len(times)
    out = np.zeros((K, N, N), dtype=np.float64)
    basis = k_particle_basis(N, k)
    index = {b: i for i, b in enumerate(basis)}

    if k == 1:
        # Initial state |i> for each i.
        for ti, t in enumerate(times):
            # U(t) = V @ diag(exp(-i t lambda)) @ V^dagger
            phase = np.exp(-1j * t * eigvals)
            U = (eigvecs * phase) @ eigvecs.conj().T
            out[ti] = np.abs(U) ** 2  # |<j|U|i>|^2, indexed (j, i)
            out[ti] = out[ti].T  # make P[t, i, j] = P(i -> j)
    elif k == 2:
        # Pre-pick initial states. For initial="localised", each (i, j) gives
        # its own initial state |ij> (one walker per node); the output is
        # symmetrised.
        if initial == "localised":
            for ti, t in enumerate(times):
                phase = np.exp(-1j * t * eigvals)
                U = (eigvecs * phase) @ eigvecs.conj().T  # shape (D, D)
                # For each ordered pair (i, j) of distinct nodes we look at
                # |<{i,j}|U|{i,j}>|^2 — the self-return probability — plus the
                # marginal probability that the 2-particle state has support
                # on node j after starting at {i, ?}. The simplest sufficient
                # statistic for graph-level features is the (i, j) marginal
                # probability that one walker is at j after starting both at
                # site i and any neighbour. We use the simplest definition
                # consistent with the paper text: P[t, i, j] is the average
                # over the initial 2-particle states {i, m} (m != i) of the
                # probability of finding one walker at j.
                D = U.shape[0]
                # Marginal: for each pair-state basis index, the probability
                # of node j being occupied = sum over pair-states containing j
                # of |<pair|U|init>|^2.
                p_pair = np.abs(U) ** 2  # rows: final pairs, cols: init pairs
                # Build (pair -> node-occupation) indicator (D, N).
                node_indicator = np.zeros((D, N), dtype=np.float64)
                for pi, pair in enumerate(basis):
                    for site in pair:
                        node_indicator[pi, site] = 1.0
                # For initial pair p_init = {i, m}, probability of being at j
                # is sum_pf node_indicator[pf, j] * p_pair[pf, p_init].
                # Aggregate by averaging over all initial pair-states that
                # contain i: gives an (N, N) tensor.
                per_init = node_indicator.T @ p_pair  # (N, D)
                pair_contains_i = np.zeros((N, D), dtype=np.float64)
                for pi, pair in enumerate(basis):
                    for site in pair:
                        pair_contains_i[site, pi] = 1.0
                # normalise so that each row sums to a count
                counts = pair_contains_i.sum(axis=1, keepdims=True)
                counts = np.where(counts > 0, counts, 1.0)
                # P[t, i, j] = (sum_{m: pair contains i} prob walker at j) / count.
                out[ti] = (pair_contains_i @ per_init.T) / counts
        elif initial == "edges":
            # Uniform superposition over edges (i, j) in E.
            edges = list(zip(*np.nonzero(np.triu(A))))
            init_vec = np.zeros(len(basis), dtype=np.complex128)
            for i, j in edges:
                init_vec[index[(i, j)]] = 1.0
            n = np.linalg.norm(init_vec)
            if n > 0:
                init_vec = init_vec / n
            for ti, t in enumerate(times):
                phase = np.exp(-1j * t * eigvals)
                U = (eigvecs * phase) @ eigvecs.conj().T
                psi_t = U @ init_vec
                p_pair = np.abs(psi_t) ** 2
                # Marginal node distribution; broadcast across all i.
                node_prob = np.zeros(N, dtype=np.float64)
                for pi, pair in enumerate(basis):
                    for site in pair:
                        node_prob[site] += p_pair[pi]
                out[ti] = np.outer(np.ones(N), node_prob)
        else:
            raise ValueError(f"unknown initial: {initial}")
    else:
        raise NotImplementedError("only k=1 and k=2 are supported")
    return out


# ---------------------------------------------------------------------------
# Quantum-inspired random walk (QiRW).
# ---------------------------------------------------------------------------


def _uniform_edge_state(A: np.ndarray, basis: list[tuple[int, ...]]) -> np.ndarray:
    """Construct the paper's two-particle distribution over graph edges."""
    basis_indices = {state: index for index, state in enumerate(basis)}
    edge_indices = [
        basis_indices[(i, j)]
        for i in range(A.shape[0])
        for j in range(i + 1, A.shape[0])
        if A[i, j] != 0 or A[j, i] != 0
    ]
    if not edge_indices:
        raise ValueError("2-QiRW requires a graph with at least one edge")
    initial_state = np.zeros(len(basis), dtype=np.float64)
    initial_state[edge_indices] = 1.0 / len(edge_indices)
    return initial_state


def qirw_features(A: np.ndarray, k: int, num_features: int) -> np.ndarray:
    """Compute discrete quantum-inspired random-walk positional encodings.

    For two particles, this implements
    ``<ij|((D_2^XY)^-1 H_2^XY)^s|psi_init>`` where ``psi_init`` is the uniform
    distribution over the original graph edges. Feature zero uses the identity
    power and is therefore ``<ij|psi_init>``. A request for ``num_features``
    returns powers ``s = 0, ..., num_features - 1``.

    Parameters
    ----------
    A : numpy.ndarray
        Graph adjacency matrix with shape ``(N, N)``.
    k : int
        Number of particles. Supported values are 1 and 2.
    num_features : int
        Total feature count, including the identity-power feature at index zero.

    Returns
    -------
    numpy.ndarray
        Positional-encoding tensor with shape ``(num_features, N, N)``.

    Raises
    ------
    ValueError
        If ``num_features`` is not positive or a two-particle graph has no edge.
    NotImplementedError
        If ``k`` is not 1 or 2.
    """
    if num_features < 1:
        raise ValueError("num_features must be positive")
    N = A.shape[0]
    H = xy_in_k_subspace(A, k)
    deg = H.sum(axis=1)
    deg_inv = np.divide(
        1.0,
        deg,
        out=np.zeros_like(deg, dtype=np.float64),
        where=deg > 0,
    )
    M = deg_inv[:, None] * H  # row-normalised; matches RRWP analogue.

    basis = k_particle_basis(N, k)
    out = np.zeros((num_features, N, N), dtype=np.float64)

    if k == 1:
        cur = np.eye(N)
        out[0] = cur
        for step in range(1, num_features):
            cur = cur @ M
            out[step] = cur
    elif k == 2:
        state = _uniform_edge_state(A, basis)
        for step in range(num_features):
            if step > 0:
                state = M @ state
            for state_index, (i, j) in enumerate(basis):
                out[step, i, j] = state[state_index]
                out[step, j, i] = state[state_index]
    else:
        raise NotImplementedError("only k=1 and k=2 are supported")
    return out


# ---------------------------------------------------------------------------
# Ising correlation matrix on the ground state.
# ---------------------------------------------------------------------------


def ising_hamiltonian(A: np.ndarray) -> np.ndarray:
    """Build the full Ising hamiltonian H_I = sum_{(i,j) in E} Z_i Z_j.

    Returns a 2^N x 2^N diagonal matrix; for N>~12 this is intentionally
    capped because the dense Hilbert space becomes infeasible.
    """
    N = A.shape[0]
    if N > 18:
        raise ValueError(f"Full Ising hamiltonian only supported up to N=18, got N={N}")
    dim = 1 << N
    diag = np.zeros(dim, dtype=np.float64)
    edges = list(zip(*np.nonzero(np.triu(A))))
    for state in range(dim):
        energy = 0.0
        for i, j in edges:
            zi = 1.0 - 2.0 * ((state >> i) & 1)
            zj = 1.0 - 2.0 * ((state >> j) & 1)
            energy += zi * zj
        diag[state] = energy
    return diag  # we only need the diagonal in the computational basis


def ising_ground_state(A: np.ndarray) -> np.ndarray:
    """Equal superposition over the Ising ground-state manifold (Sec. A.2)."""
    diag = ising_hamiltonian(A)
    e_min = diag.min()
    mask = np.isclose(diag, e_min)
    psi = np.zeros_like(diag)
    psi[mask] = 1.0
    psi = psi / np.sqrt(mask.sum())
    return psi  # real-valued amplitudes in computational basis


def correlation_matrix_on_state(psi: np.ndarray, num_nodes: int) -> np.ndarray:
    """C_{ij} = <psi| Z_i Z_j |psi> for the basis-state amplitude vector psi."""
    probs = psi**2  # psi is real here
    C = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for state, p in enumerate(probs):
        if p == 0.0:
            continue
        bits = np.array([(state >> b) & 1 for b in range(num_nodes)], dtype=np.float64)
        signs = 1.0 - 2.0 * bits  # +1 for 0, -1 for 1
        C += p * np.outer(signs, signs)
    return C


def ground_state_correlation_eigvecs(A: np.ndarray, num_features: int) -> np.ndarray:
    """Top-`num_features` eigenvectors of the Ising-ground-state correlation matrix.

    Returns an (N, num_features) array used as drop-in replacement for the
    Laplacian eigenmaps in graph transformers (Sec. 3.2.1).
    """
    N = A.shape[0]
    psi = ising_ground_state(A)
    C = correlation_matrix_on_state(psi, N)
    eigvals, eigvecs = np.linalg.eigh(C)
    # eigh returns ascending; we want the largest few — take the last columns.
    idx = np.argsort(-eigvals)
    top = idx[:num_features]
    feats = eigvecs[:, top]
    # Pad with zeros if N < num_features for tiny graphs.
    if feats.shape[1] < num_features:
        pad = np.zeros((N, num_features - feats.shape[1]), dtype=np.float64)
        feats = np.concatenate([feats, pad], axis=1)
    return feats


def ladder_ising_ground_states(
    num_nodes: int,
    edges: list[tuple[int, int]],
    maximum_states: int = 10_000,
) -> np.ndarray:
    """Compute Ising ground states of a two-rail ladder by transfer DP.

    The synthetic graphs in Appendix D.2 only contain rung edges and edges
    between consecutive rungs. Consequently, the boundary of a partial graph
    has four spin assignments, independent of ladder length. This computes the
    exact minimum-energy manifold in time linear in the number of rungs. The
    added crossing term has multiplicity two in the paper-specific energy; this
    is the convention that gives the nine type-2 states shown in Figure 7
    after identifying global spin reversal.

    Parameters
    ----------
    num_nodes : int
        Even number of ladder nodes, ordered as consecutive two-node rungs.
    edges : list[tuple[int, int]]
        Undirected ladder edges.
    maximum_states : int
        Maximum ground-state count to materialize. Default value is 10,000.

    Returns
    -------
    numpy.ndarray
        Ground-state spin vectors with shape ``(num_states, num_nodes)``.

    Raises
    ------
    ValueError
        If the graph is not a consecutive-rung ladder or the ground-state
        manifold exceeds ``maximum_states``.
    """
    if num_nodes % 2:
        raise ValueError("ladder ground-state construction requires an even node count")
    num_rungs = num_nodes // 2
    edge_set = {tuple(sorted(edge)) for edge in edges}
    intra_rung_edges: list[list[tuple[int, int]]] = [[] for _ in range(num_rungs)]
    transition_edges: list[list[tuple[int, int]]] = [
        [] for _ in range(max(0, num_rungs - 1))
    ]
    for first_node, second_node in edge_set:
        first_rung, second_rung = first_node // 2, second_node // 2
        if first_rung == second_rung:
            intra_rung_edges[first_rung].append((first_node % 2, second_node % 2))
        elif second_rung == first_rung + 1:
            transition_edges[first_rung].append((first_node % 2, second_node % 2))
        else:
            raise ValueError(
                "paper-specific ladder solver only supports edges within or "
                "between consecutive rungs"
            )

    rung_states = np.asarray([(-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=np.int8)

    def internal_energy(rung_index: int, state_index: int) -> int:
        spins = rung_states[state_index]
        return sum(
            int(spins[first_position] * spins[second_position])
            for first_position, second_position in intra_rung_edges[rung_index]
        )

    energies = np.asarray(
        [internal_energy(0, state_index) for state_index in range(4)],
        dtype=np.int64,
    )
    paths: list[list[tuple[int, ...]]] = [[(state_index,)] for state_index in range(4)]
    for rung_index in range(1, num_rungs):
        next_energies = np.full(4, np.iinfo(np.int64).max, dtype=np.int64)
        next_paths: list[list[tuple[int, ...]]] = [[] for _ in range(4)]
        for next_state_index, next_spins in enumerate(rung_states):
            candidates: list[tuple[int, int]] = []
            for previous_state_index, previous_spins in enumerate(rung_states):
                transition_energy = sum(
                    (2 if (first_position, second_position) == (0, 1) else 1)
                    * int(previous_spins[first_position] * next_spins[second_position])
                    for first_position, second_position in transition_edges[
                        rung_index - 1
                    ]
                )
                candidates.append(
                    (
                        int(energies[previous_state_index]) + transition_energy,
                        previous_state_index,
                    )
                )
            minimum_candidate = min(energy for energy, _ in candidates)
            next_energies[next_state_index] = minimum_candidate + internal_energy(
                rung_index, next_state_index
            )
            for candidate_energy, previous_state_index in candidates:
                if candidate_energy == minimum_candidate:
                    next_paths[next_state_index].extend(
                        path + (next_state_index,)
                        for path in paths[previous_state_index]
                    )
            if len(next_paths[next_state_index]) > maximum_states:
                raise ValueError(
                    "ladder Ising ground-state manifold exceeds maximum_states="
                    f"{maximum_states}"
                )
        energies, paths = next_energies, next_paths

    minimum_energy = int(energies.min())
    ground_paths = [
        path
        for state_index, state_energy in enumerate(energies)
        if state_energy == minimum_energy
        for path in paths[state_index]
    ]
    if len(ground_paths) > maximum_states:
        raise ValueError(
            "ladder Ising ground-state manifold exceeds maximum_states="
            f"{maximum_states}"
        )
    return np.asarray(
        [rung_states[np.asarray(path)].reshape(-1) for path in ground_paths],
        dtype=np.float64,
    )


def ladder_ground_state_correlation_eigvecs(
    num_nodes: int,
    edges: list[tuple[int, int]],
    num_features: int,
) -> np.ndarray:
    """Compute ladder correlation eigenvectors without forming ``2**N`` states.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the ladder graph.
    edges : list[tuple[int, int]]
        Undirected graph edges.
    num_features : int
        Number of leading correlation eigenvectors.

    Returns
    -------
    numpy.ndarray
        Node features with shape ``(num_nodes, num_features)``.
    """
    ground_states = ladder_ising_ground_states(num_nodes, edges)
    # C = S.T @ S / m. Apply C implicitly so long ladders never allocate the
    # dense N x N matrix and a moderately degenerate manifold does not trigger
    # a cubic SVD in the number of states.
    from scipy.sparse.linalg import LinearOperator, eigsh

    num_states = len(ground_states)
    correlation_operator = LinearOperator(
        shape=(num_nodes, num_nodes),
        matvec=lambda vector: ground_states.T @ (ground_states @ vector) / num_states,
        matmat=lambda matrix: ground_states.T @ (ground_states @ matrix) / num_states,
        dtype=np.float64,
    )
    available_features = min(num_features, num_nodes - 1, num_states)
    _, features = eigsh(
        correlation_operator,
        k=available_features,
        which="LM",
        v0=np.linspace(1.0, 2.0, num_nodes),
    )
    features = features[:, ::-1].copy()
    if available_features < num_features:
        features = np.pad(
            features,
            ((0, 0), (0, num_features - available_features)),
        )
    return features


def laplacian_eigenvectors(A: np.ndarray, num_features: int) -> np.ndarray:
    """Return the nontrivial normalized-Laplacian eigenvectors.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric adjacency matrix.
    num_features : int
        Requested node-feature dimension.

    Returns
    -------
    numpy.ndarray
        Eigenvectors with shape ``(num_nodes, num_features)``.
    """
    from scipy import sparse
    from scipy.sparse.linalg import eigsh

    num_nodes = A.shape[0]
    degrees = A.sum(axis=1)
    if np.any(degrees == 0):
        raise ValueError(
            "Laplacian eigenvectors require a graph without isolated nodes"
        )
    inverse_square_root_degree = sparse.diags(degrees**-0.5)
    normalized_laplacian = sparse.eye(num_nodes) - (
        inverse_square_root_degree @ sparse.csr_matrix(A) @ inverse_square_root_degree
    )
    num_eigenvectors = min(num_features + 1, num_nodes - 1)
    _, eigenvectors = eigsh(
        normalized_laplacian,
        k=num_eigenvectors,
        which="SM",
        v0=np.linspace(1.0, 2.0, num_nodes),
    )
    features = eigenvectors[:, 1 : num_features + 1]
    if features.shape[1] < num_features:
        features = np.pad(features, ((0, 0), (0, num_features - features.shape[1])))
    return features


def rrwp_return_features(A: np.ndarray, num_features: int) -> np.ndarray:
    """Return node-wise diagonal RRWP values for successive walk powers.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric adjacency matrix.
    num_features : int
        Number of walk powers, including the identity power.

    Returns
    -------
    numpy.ndarray
        Return-probability node features with shape
        ``(num_nodes, num_features)``.
    """
    from scipy import sparse

    degrees = A.sum(axis=1)
    if np.any(degrees == 0):
        raise ValueError("RRWP return features require a graph without isolated nodes")
    transition = sparse.diags(1.0 / degrees) @ sparse.csr_matrix(A)
    current_power = sparse.eye(A.shape[0], format="csr")
    features = np.empty((A.shape[0], num_features), dtype=np.float64)
    for step in range(num_features):
        features[:, step] = current_power.diagonal()
        current_power = current_power @ transition
    return features


def rrwp_edge_features(
    A: np.ndarray,
    edges: list[tuple[int, int]],
    num_features: int,
) -> np.ndarray:
    """Return full relative-walk channels on the graph's sparse edges.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric adjacency matrix.
    edges : list[tuple[int, int]]
        Edges on which the relative probabilities are consumed.
    num_features : int
        Number of walk powers, including the identity power.

    Returns
    -------
    numpy.ndarray
        Edge features with shape ``(num_edges, num_features)``.
    """
    from scipy import sparse

    degrees = A.sum(axis=1)
    if np.any(degrees == 0):
        raise ValueError("RRWP edge features require a graph without isolated nodes")
    transition = sparse.diags(1.0 / degrees) @ sparse.csr_matrix(A)
    current_power = sparse.eye(A.shape[0], format="csr")
    features = np.empty((len(edges), num_features), dtype=np.float64)
    edge_rows = np.asarray([edge[0] for edge in edges])
    edge_columns = np.asarray([edge[1] for edge in edges])
    for step in range(num_features):
        features[:, step] = np.asarray(current_power[edge_rows, edge_columns]).reshape(
            -1
        )
        current_power = current_power @ transition
    return features


# ---------------------------------------------------------------------------
# Torch helpers — convert per-graph numpy features into tensors of the shape
# expected by GNN training loops.
# ---------------------------------------------------------------------------


def stack_to_edge_features(P: np.ndarray) -> torch.Tensor:
    """Convert a (K, N, N) PE tensor into per-edge features (E, K) on the full
    adjacency (E = N*N) — the format that GRIT and our minimal model use."""
    return torch.from_numpy(P).float().permute(1, 2, 0).contiguous()  # (N, N, K)
