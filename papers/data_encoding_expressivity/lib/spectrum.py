"""Dense-matrix quantum models for the exact spectrum checks (experiment E4).

The paper's spectrum results (Eqs. 6-12, App. A-B) assume arbitrary trainable
unitaries ``W`` and observable ``M``. These helpers build such models directly
as dense matrices, ``f(x) = <0| U(x)^dagger M U(x) |0>`` with
``U(x) = W^(L+1) S(x) W^(L) ... S(x) W^(1)`` and ``S(x) = exp(-i x H)``, using
Haar-random ``W`` and random Hermitian ``M`` so that every coefficient allowed
by the spectrum is generically nonzero.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def haar_unitary(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a Haar-random unitary (QR of a complex Ginibre matrix with phase fix)."""
    ginibre = (
        rng.standard_normal((dimension, dimension))
        + 1j * rng.standard_normal((dimension, dimension))
    ) / np.sqrt(2)
    q_factor, r_factor = np.linalg.qr(ginibre)
    diagonal = np.diag(r_factor)
    return q_factor * (diagonal / np.abs(diagonal))


def random_hermitian(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random Hermitian observable ``(A + A^dagger) / 2``."""
    matrix = rng.standard_normal((dimension, dimension)) + 1j * rng.standard_normal(
        (dimension, dimension)
    )
    return (matrix + matrix.conj().T) / 2


def generator_with_spectrum(
    eigenvalues: Sequence[float], rotate: bool, rng: np.random.Generator
) -> np.ndarray:
    """Return a Hermitian generator ``H = V diag(eigenvalues) V^dagger``.

    Parameters
    ----------
    eigenvalues : Sequence[float]
        Eigenvalues of ``H``.
    rotate : bool
        If True, ``V`` is Haar random, so ``H`` is not diagonal. The paper
        absorbs ``V`` into ``W`` (Section II), so the spectrum must not change.
    rng : np.random.Generator
        Random generator for ``V``.

    Returns
    -------
    np.ndarray
        Generator of shape ``(d, d)``.
    """
    diagonal = np.diag(np.asarray(eigenvalues, dtype=float)).astype(complex)
    if not rotate:
        return diagonal
    basis = haar_unitary(len(eigenvalues), rng)
    return basis @ diagonal @ basis.conj().T


def encoding_unitaries(inputs: np.ndarray, generator: np.ndarray) -> np.ndarray:
    """Return ``exp(-i x H)`` for every input, shape ``(N, d, d)``."""
    eigenvalues, basis = np.linalg.eigh(generator)
    phases = np.exp(-1j * inputs[:, None] * eigenvalues[None, :])
    return np.einsum("ij,nj,kj->nik", basis, phases, basis.conj())


def layered_model_values(
    inputs: np.ndarray,
    generator: np.ndarray,
    trainable_unitaries: Sequence[np.ndarray],
    observable: np.ndarray,
) -> np.ndarray:
    """Evaluate ``f(x) = <0| U^dagger M U |0>`` for the layered model of Eq. 4.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of shape ``(N,)``.
    generator : np.ndarray
        Encoding generator ``H`` of shape ``(d, d)``, the same in every layer.
    trainable_unitaries : Sequence[np.ndarray]
        ``W^(1) ... W^(L+1)``, each of shape ``(d, d)``.
    observable : np.ndarray
        Hermitian ``M`` of shape ``(d, d)``.

    Returns
    -------
    np.ndarray
        Real model values of shape ``(N,)``.
    """
    dimension = generator.shape[0]
    encodings = encoding_unitaries(inputs, generator)
    state = np.zeros((len(inputs), dimension), dtype=complex)
    state[:, 0] = 1.0
    for trainable in trainable_unitaries[:-1]:
        state = state @ trainable.T
        state = np.einsum("nij,nj->ni", encodings, state)
    state = state @ trainable_unitaries[-1].T
    values = np.einsum("ni,ij,nj->n", state.conj(), observable, state)
    return values.real


def pauli_x_sum_generator(n_qubits: int) -> np.ndarray:
    """Return ``H = sum_q X_q / 2``, the generator of ``RX(x)`` on every wire."""
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    generator = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for qubit in range(n_qubits):
        term = np.array([[1.0]], dtype=complex)
        for wire in range(n_qubits):
            term = np.kron(term, pauli_x if wire == qubit else identity)
        generator += term / 2
    return generator


def multivariate_model_values(
    first_inputs: np.ndarray,
    second_inputs: np.ndarray,
    first_generator: np.ndarray,
    second_generator: np.ndarray,
    trainable_unitaries: Sequence[np.ndarray],
    observable: np.ndarray,
) -> np.ndarray:
    """Evaluate the ``L = 1`` two-feature model of App. A on a 2D grid.

    ``S(x_1, x_2) = exp(-i x_1 H_1) (x) exp(-i x_2 H_2)``, so the spectrum is the
    Cartesian product of the two single-feature spectra.

    Parameters
    ----------
    first_inputs, second_inputs : np.ndarray
        Grids of shape ``(N1,)`` and ``(N2,)``.
    first_generator, second_generator : np.ndarray
        Generators acting on registers of dimension ``d1`` and ``d2``.
    trainable_unitaries : Sequence[np.ndarray]
        ``W^(1)`` and ``W^(2)`` on the joint register, shape ``(d1 d2, d1 d2)``.
    observable : np.ndarray
        Hermitian ``M`` on the joint register.

    Returns
    -------
    np.ndarray
        Real values of shape ``(N1, N2)``.
    """
    first_encodings = encoding_unitaries(first_inputs, first_generator)
    second_encodings = encoding_unitaries(second_inputs, second_generator)
    first_dim = first_generator.shape[0]
    second_dim = second_generator.shape[0]
    initial = trainable_unitaries[0][:, 0].reshape(first_dim, second_dim)
    state = np.einsum("aij,bkl,jl->abik", first_encodings, second_encodings, initial)
    state = state.reshape(len(first_inputs), len(second_inputs), first_dim * second_dim)
    state = state @ trainable_unitaries[1].T
    values = np.einsum("abi,ij,abj->ab", state.conj(), observable, state)
    return values.real
