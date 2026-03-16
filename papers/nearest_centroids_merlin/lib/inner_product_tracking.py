"""
Inner product tracking for c_exp vs c_sim analysis.

This module provides functions to collect inner product values during
classification for noise model validation (Figure 13 in paper).
"""

from typing import Any

import cirq
import numpy as np

from .gates import VectorLoader
from .noise import GateNoise, MeasurementNoise


def compute_inner_products_with_tracking(
    X_test: np.ndarray,
    centroids: np.ndarray,
    repetitions: int = 1000,
    error_rate: float = 0.0,
) -> dict[str, Any]:
    """
    Compute inner products and track both simulated and experimental values.

    Returns
    -------
    dict with keys:
        - 'c_sim': List of simulated (ideal) inner product squared values
        - 'c_exp_no_mit': List of experimental values without mitigation
        - 'c_exp_mit': List of experimental values with mitigation
        - 'distances_sim': List of simulated distances
        - 'distances_exp': List of experimental distances
    """
    c_sim_list = []
    c_exp_no_mit_list = []
    c_exp_mit_list = []
    distances_sim = []
    distances_exp = []

    for x in X_test:
        for centroid in centroids:
            # Simulated (ideal) inner product
            x_norm = x / np.linalg.norm(x)
            c_norm = centroid / np.linalg.norm(centroid)
            c_sim = np.dot(x_norm, c_norm) ** 2
            c_sim_list.append(c_sim)

            # Experimental inner products
            c_exp_no_mit, c_exp_mit = _measure_inner_product(
                x, centroid, repetitions, error_rate
            )
            c_exp_no_mit_list.append(c_exp_no_mit)
            c_exp_mit_list.append(c_exp_mit)

            # Distances
            norm_x = np.linalg.norm(x)
            norm_c = np.linalg.norm(centroid)

            dist_sim = np.sqrt(
                norm_x**2 + norm_c**2 - 2 * norm_x * norm_c * np.sqrt(c_sim)
            )
            dist_exp = np.sqrt(
                norm_x**2 + norm_c**2 - 2 * norm_x * norm_c * np.sqrt(max(0, c_exp_mit))
            )

            distances_sim.append(dist_sim)
            distances_exp.append(dist_exp)

    return {
        "c_sim": np.array(c_sim_list),
        "c_exp_no_mit": np.array(c_exp_no_mit_list),
        "c_exp_mit": np.array(c_exp_mit_list),
        "distances_sim": np.array(distances_sim),
        "distances_exp": np.array(distances_exp),
    }


def _measure_inner_product(
    x: np.ndarray,
    y: np.ndarray,
    repetitions: int,
    error_rate: float,
) -> tuple[float, float]:
    """
    Measure inner product with and without error mitigation.

    Returns (c_exp_no_mitigation, c_exp_with_mitigation)
    """
    qubits = cirq.LineQubit.range(len(x))
    gates_x = cirq.decompose(VectorLoader(x)(*qubits))
    gates_y = cirq.decompose(VectorLoader(y, x_flip=False)(*qubits) ** -1)

    # Circuit with all qubits measured (for mitigation)
    gates_m = cirq.measure(*qubits, key="all")
    circuit = cirq.Circuit(gates_x, gates_y, gates_m)

    if error_rate > 0.0:
        GateNoise(error_rate).apply(circuit)
        MeasurementNoise(error_rate).apply(circuit)

    simulator = cirq.Simulator()
    measure = simulator.run(circuit, repetitions=repetitions)
    hist = measure.histogram(key="all")

    n = len(x)

    # Without mitigation: just count first qubit = 1
    # State |10...0> = 2^(n-1)
    total_no_mit = hist.get(2 ** (n - 1), 0)
    c_exp_no_mit = total_no_mit / repetitions

    # With mitigation: post-select on unary basis states
    unary_states = [2**i for i in range(n)]
    num_valid = sum(hist.get(state, 0) for state in unary_states)
    total_mit = hist.get(2 ** (n - 1), 0)
    c_exp_mit = total_mit / num_valid if num_valid > 0 else 0.0

    return c_exp_no_mit, c_exp_mit
