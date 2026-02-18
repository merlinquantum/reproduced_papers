import copy

import numpy as np
from merlin.algorithms import FeatureMap, FidelityKernel
from numpy import ndarray
from perceval import GenericInterferometer, InterferometerShape
from scipy.linalg import inv, sqrtm

from .feature_map import circuit_func
from .noise import NoisySLOSComputeGraph


def generate_data(
    data_size: int,
    reg: float,
    input_state: list = None,
    quantum_kernel: FidelityKernel = None,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Generate ad-hoc dataset.

    Args:
        data_size: Number of generated X, y datapoints.
        reg: Regularization factor for data generation algorithm
        return_kernels: Specify whether or not to return kernels used to
            construct dataset.

        quantum_kernel: Merlin FidelityKernel with respect to which to
            calculate the two kernels.

    Returns:
        np.ndarray: Features dataset
        np.ndarray: Labels dataset
        np.ndarray: Kernel matrix associated with fully
            indistinguishable photons.
        np.ndarray: Kernel matrix associated with fully distinguishable
            photons.
    """
    if input_state is None and quantum_kernel is None:
        raise ValueError(
            "Please provide either an input_state or aquantum kernel instance."
        )
    if quantum_kernel is None:
        m = len(input_state)

        # Set up feature map

        circuit = GenericInterferometer(
            m, circuit_func, shape=InterferometerShape.RECTANGLE
        )
        input_size = len(circuit.get_parameters())
        feature_map = FeatureMap(circuit, input_size, input_parameters="phi")

        # Set up quantum kernels
        quantum_kernel = FidelityKernel(
            feature_map,
            input_state,
            force_psd=False,
            no_bunching=False,
        )
    else:
        quantum_kernel = copy.copy(quantum_kernel)

    input_state = quantum_kernel.input_state

    # Force quantum kernel's SLOS graph to have 0 indistinguishability.
    original_slos_graph = quantum_kernel._slos_graph

    quantum_kernel._slos_graph = NoisySLOSComputeGraph(indistinguishability=1e-5)

    # Generate X, y values
    input_size = quantum_kernel.input_size
    X = np.random.uniform(0, 2 * np.pi, size=(data_size, input_size))

    K2 = quantum_kernel(X).detach().numpy()

    # Restore SLOS graph
    quantum_kernel._slos_graph = original_slos_graph

    K1 = quantum_kernel(X).detach().numpy()

    S = sqrtm(K1) @ inv(K2 + reg * np.eye(data_size)) @ sqrtm(K1)

    eigenvals, eigenvecs = np.linalg.eig(S)

    max_eigenval = np.max(eigenvals)
    geometric_diff = np.sqrt(max_eigenval)

    v_g = eigenvecs[:, np.argmax(eigenvals)]
    vector = sqrtm(K1) @ v_g

    # Assign labels
    y = np.where(np.real(vector) >= 0, 1, -1)

    # Data generation
    return X, y, geometric_diff, K1, K2
