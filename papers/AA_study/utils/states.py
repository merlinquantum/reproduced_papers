import numpy as np
from numpy.typing import NDArray


def mixed_state(n_qubits: int) -> NDArray:
    return np.eye(2**n_qubits) / (2**n_qubits)


def superposition_state(n_qubits: int) -> NDArray:
    return np.ones((2**n_qubits, 2**n_qubits)) / (2**n_qubits)
