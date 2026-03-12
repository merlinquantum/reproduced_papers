import numpy as np
from numpy.typing import NDArray


def mixed_state(n_qubits: int) -> NDArray:
    """
    Return the maximally mixed state density matrix.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    numpy.ndarray
        Density matrix of shape (2**n_qubits, 2**n_qubits).
    """
    return np.eye(2**n_qubits) / (2**n_qubits)


def superposition_state(n_qubits: int) -> NDArray:
    """
    Return the uniform superposition density matrix.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    numpy.ndarray
        Density matrix of shape (2**n_qubits, 2**n_qubits).
    """
    return np.ones((2**n_qubits, 2**n_qubits)) / (2**n_qubits)
