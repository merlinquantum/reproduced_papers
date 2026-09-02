"""Paper-scale strongly regular graph distinguishability experiment."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from .qpe import k_particle_basis, rrwp, xy_in_k_subspace

_CATALOGS = {
    "srg(25,12,5,6)": ("sr251256.g6", (25, 12, 5, 6), 15),
    "srg(26,10,3,4)": ("sr261034.g6", (26, 10, 3, 4), 10),
}


def load_srg_catalogs(data_directory: str | Path) -> dict[str, list[np.ndarray]]:
    """Load and validate the two complete SRG families used by the paper.

    Parameters
    ----------
    data_directory : str | pathlib.Path
        Directory containing the graph6 catalog files.

    Returns
    -------
    dict[str, list[numpy.ndarray]]
        Adjacency matrices grouped by SRG family name.

    Raises
    ------
    FileNotFoundError
        If a required catalog is missing.
    ValueError
        If a catalog count or SRG parameter tuple is incorrect.
    """
    data_directory = Path(data_directory)
    families: dict[str, list[np.ndarray]] = {}
    for family_name, (filename, parameters, expected_count) in _CATALOGS.items():
        catalog_path = data_directory / filename
        if not catalog_path.is_file():
            raise FileNotFoundError(f"missing SRG catalog: {catalog_path}")
        graphs = [
            nx.from_graph6_bytes(line.strip())
            for line in catalog_path.read_bytes().splitlines()
            if line.strip()
        ]
        if len(graphs) != expected_count:
            raise ValueError(
                f"{family_name} requires {expected_count} graphs, got {len(graphs)}"
            )
        adjacency_matrices = []
        for graph_index, graph in enumerate(graphs):
            actual_parameters = _strongly_regular_parameters(graph)
            if actual_parameters != parameters:
                raise ValueError(
                    f"{family_name} graph {graph_index} has parameters "
                    f"{actual_parameters}, expected {parameters}"
                )
            adjacency_matrices.append(nx.to_numpy_array(graph, dtype=np.float64))
        families[family_name] = adjacency_matrices
    return families


def _strongly_regular_parameters(graph: nx.Graph) -> tuple[int, int, int, int]:
    """Return ``(v, k, lambda, mu)`` after checking strong regularity."""
    num_nodes = graph.number_of_nodes()
    degrees = {degree for _, degree in graph.degree()}
    if len(degrees) != 1:
        raise ValueError("graph is not regular")
    degree = degrees.pop()
    adjacent_common_neighbors: set[int] = set()
    nonadjacent_common_neighbors: set[int] = set()
    neighbor_sets = {node: set(graph.neighbors(node)) for node in graph.nodes}
    for first_node in graph.nodes:
        for second_node in range(first_node + 1, num_nodes):
            common_count = len(neighbor_sets[first_node] & neighbor_sets[second_node])
            target = (
                adjacent_common_neighbors
                if graph.has_edge(first_node, second_node)
                else nonadjacent_common_neighbors
            )
            target.add(common_count)
    if len(adjacent_common_neighbors) != 1 or len(nonadjacent_common_neighbors) != 1:
        raise ValueError("graph is not strongly regular")
    return (
        num_nodes,
        degree,
        adjacent_common_neighbors.pop(),
        nonadjacent_common_neighbors.pop(),
    )


def sorted_correlation_distance(
    first_correlation: np.ndarray,
    second_correlation: np.ndarray,
) -> float:
    """Compute the paper's half-L1 distance between sorted matrix entries.

    Parameters
    ----------
    first_correlation : numpy.ndarray
        First square correlation matrix.
    second_correlation : numpy.ndarray
        Second square correlation matrix.

    Returns
    -------
    float
        Permutation-invariant sorted-correlation distance.
    """
    if first_correlation.shape != second_correlation.shape:
        raise ValueError("correlation matrices must have the same shape")
    return float(
        0.5
        * np.abs(
            np.sort(first_correlation, axis=None)
            - np.sort(second_correlation, axis=None)
        ).sum()
    )


def two_particle_walk_correlations(
    adjacency: np.ndarray,
    evolution_time: float,
) -> np.ndarray:
    """Compute interacting two-particle XY-walk occupation covariances.

    The hard-core two-particle Hamiltonian is the XY Hamiltonian restricted to
    distinct-node basis states. For each localized basis state, this computes
    the occupation covariance after continuous evolution, then averages those
    covariances. Averaging over every localization makes the matrix equivariant
    to node relabeling while retaining the interacting-walk information.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Symmetric graph adjacency matrix.
    evolution_time : float
        Continuous-walk evolution time.

    Returns
    -------
    numpy.ndarray
        Node occupation covariance matrix.
    """
    num_nodes = adjacency.shape[0]
    basis = k_particle_basis(num_nodes, 2)
    hamiltonian = xy_in_k_subspace(adjacency, 2)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    evolution = (
        eigenvectors * np.exp(-1j * evolution_time * eigenvalues)
    ) @ eigenvectors.T
    transition_probabilities = np.abs(evolution) ** 2
    occupation = np.zeros((len(basis), num_nodes), dtype=np.float64)
    for basis_index, (first_node, second_node) in enumerate(basis):
        occupation[basis_index, (first_node, second_node)] = 1.0
    marginal_by_initial_state = occupation.T @ transition_probabilities
    correlations = -(marginal_by_initial_state @ marginal_by_initial_state.T) / len(
        basis
    )
    mean_pair_occupation = 1.0 / len(basis)
    for first_node, second_node in basis:
        correlations[first_node, second_node] += mean_pair_occupation
        correlations[second_node, first_node] += mean_pair_occupation
    correlations[np.diag_indices(num_nodes)] += marginal_by_initial_state.mean(axis=1)
    return correlations


def ising_p2_linked_correlations(adjacency: np.ndarray) -> np.ndarray:
    """Compute the graph-dependent linked-cluster terms entering Ising p=2.

    Exact p=2 Ising evolution on 25--26 qubits requires the parameter vector
    and a quantum backend, neither of which the paper publishes. This matrix is
    the deterministic two-layer linked-cluster signature: each entry counts
    edges induced by the pair's shared interaction neighborhood. It is kept
    separate from the exact XY-walk result so it cannot be mistaken for a
    backend-generated Ising correlator.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Symmetric graph adjacency matrix.

    Returns
    -------
    numpy.ndarray
        Pairwise linked-neighborhood signature.
    """
    num_nodes = adjacency.shape[0]
    neighborhoods = [set(np.flatnonzero(adjacency[node])) for node in range(num_nodes)]
    correlations = np.zeros_like(adjacency, dtype=np.float64)
    for first_node in range(num_nodes):
        for second_node in range(num_nodes):
            shared_neighbors = sorted(
                neighborhoods[first_node] & neighborhoods[second_node]
            )
            correlations[first_node, second_node] = (
                0.5 * adjacency[np.ix_(shared_neighbors, shared_neighbors)].sum()
            )
    return correlations


def rrwp_correlation(adjacency: np.ndarray, num_steps: int) -> np.ndarray:
    """Flatten RRWP channels into one permutation-equivariant pair matrix."""
    features = rrwp(adjacency, num_steps)
    channel_weights = 2.0 ** -np.arange(num_steps, dtype=np.float64)
    return np.tensordot(channel_weights, features, axes=(0, 0))


def run_srg_paper_experiment(config: dict, run_directory: str | Path) -> dict:
    """Run both SRG families and their random-isomorphism controls.

    Parameters
    ----------
    config : dict
        SRG experiment configuration.
    run_directory : str | pathlib.Path
        Output directory for ``srg_metrics.json``.

    Returns
    -------
    dict
        Pairwise distance matrices, control maxima, and distinction counts.
    """
    import json

    required = {
        "srg_data_directory",
        "evolution_time",
        "rrwp_steps",
        "isomorphic_permutations",
        "tolerance",
        "seed",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing SRG config entries: {', '.join(sorted(missing))}")
    if config["isomorphic_permutations"] != 5:
        raise ValueError("the paper SRG experiment requires five isomorphic controls")

    families = load_srg_catalogs(config["srg_data_directory"])
    random_generator = np.random.default_rng(config["seed"])
    tolerance = float(config["tolerance"])
    output: dict[str, object] = {
        "seed": int(config["seed"]),
        "tolerance": tolerance,
        "families": {},
        "ising_note": (
            "Linked-cluster p=2 signature only: the paper does not publish "
            "the Ising evolution parameters needed for exact correlators."
        ),
    }
    for family_name, adjacency_matrices in families.items():
        method_features = {
            "ising_p2_linked": [
                ising_p2_linked_correlations(adjacency)
                for adjacency in adjacency_matrices
            ],
            "two_particle_xy": [
                two_particle_walk_correlations(
                    adjacency, float(config["evolution_time"])
                )
                for adjacency in adjacency_matrices
            ],
            "rrwp": [
                rrwp_correlation(adjacency, int(config["rrwp_steps"]))
                for adjacency in adjacency_matrices
            ],
        }
        family_output: dict[str, object] = {}
        for method_name, features in method_features.items():
            num_graphs = len(features)
            distances = np.zeros((num_graphs, num_graphs), dtype=np.float64)
            for first_index in range(num_graphs):
                for second_index in range(first_index):
                    distance = sorted_correlation_distance(
                        features[first_index], features[second_index]
                    )
                    distances[first_index, second_index] = distance
                    distances[second_index, first_index] = distance
            control_distances = []
            for adjacency, original_feature in zip(adjacency_matrices, features):
                for _ in range(config["isomorphic_permutations"]):
                    permutation = random_generator.permutation(adjacency.shape[0])
                    permuted_adjacency = adjacency[np.ix_(permutation, permutation)]
                    if method_name == "two_particle_xy":
                        permuted_feature = two_particle_walk_correlations(
                            permuted_adjacency, float(config["evolution_time"])
                        )
                    elif method_name == "rrwp":
                        permuted_feature = rrwp_correlation(
                            permuted_adjacency, int(config["rrwp_steps"])
                        )
                    else:
                        permuted_feature = ising_p2_linked_correlations(
                            permuted_adjacency
                        )
                    control_distances.append(
                        sorted_correlation_distance(original_feature, permuted_feature)
                    )
            upper_triangle = distances[np.triu_indices(num_graphs, k=1)]
            family_output[method_name] = {
                "distance_matrix": distances.tolist(),
                "distinguished_pairs": int((upper_triangle > tolerance).sum()),
                "total_pairs": int(len(upper_triangle)),
                "minimum_nonisomorphic_distance": float(upper_triangle.min()),
                "maximum_nonisomorphic_distance": float(upper_triangle.max()),
                "maximum_isomorphic_control_distance": float(
                    max(control_distances, default=0.0)
                ),
                "controls_pass": bool(max(control_distances, default=0.0) <= tolerance),
            }
        output["families"][family_name] = family_output

    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    output_path = run_directory / "srg_metrics.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output
