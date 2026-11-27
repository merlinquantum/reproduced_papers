"""Photonic Fermi-Dirac neuron (Phase 4 MerLin adaptation; not part of the paper's experiments).

The paper's continuous-variable neuron (Sec. V, eq. 108) applies an
activation to a quadratic bosonic Hamiltonian. For a *passive*
(number-conserving) quadratic Hamiltonian on N photons in m modes,

    H = sum_ij h_ij a_i^dag a_j,    h = U^dag diag(eps) U,

the N-photon eigenstates are Fock states of the normal modes reached by the
interferometer U, with energies eps . n. Hence

    Tr[g_T(H) rho] = sum_n p_U(n) g_T(eps . n),

where p_U(n) is the photon-number distribution of rho after U. This is one
trainable MerLin interferometer, photon-number-resolving (FOCK) detection and a
classical activation of the linear energies. The convention h = U^dag diag(eps) U
matches MerLin's mode ordering (checked against dense Fock-space numerics in
``tests/test_photonic.py``).

Fock-basis ordering everywhere is MerLin's ``layer.output_keys`` order
(reverse lexicographic occupation tuples); :func:`fock_basis` reproduces it
and :class:`MerlinPassiveNeuron` checks the two agree.

Models (all expose ``n_parameters``, ``labels``, ``outputs`` and
``loss_and_gradient`` like :mod:`lib.models`, so :func:`lib.training.train_models`
trains them unchanged):

* :class:`MerlinPassiveNeuron` - the photonic neuron (activation ``tanh``) or
  its linear counterpart Tr[H rho] (activation ``linear``, baseline B3).
* :class:`PhotonCountNeuron` - commuting classical second-order neuron on
  photon counts (counterpart of the paper's IM / FCIM and of eq. 12).
* :class:`CorrelationNeuron` - classical neuron on the one-photon correlation
  matrix <a_i^dag a_j> (counterpart of baseline B2).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import scipy.sparse
from scipy.stats import unitary_group

from .activations import make_activation

READOUTS = ("pnr", "threshold", "unbunched")
INPUT_KINDS = ("fock_first", "haar_interferometer", "haar_full")


# ---------------------------------------------------------------------------
# Fock space
# ---------------------------------------------------------------------------


@cache
def fock_basis(n_modes: int, n_photons: int) -> tuple[tuple[int, ...], ...]:
    """Return the N-photon Fock basis in MerLin's order (reverse lexicographic)."""
    if n_modes < 1 or n_photons < 1:
        raise ValueError(
            f"Need n_modes >= 1 and n_photons >= 1, got {n_modes}, {n_photons}"
        )
    states = []
    for bars in itertools.combinations(range(n_photons + n_modes - 1), n_modes - 1):
        cuts = (-1, *bars, n_photons + n_modes - 1)
        states.append(tuple(cuts[k + 1] - cuts[k] - 1 for k in range(n_modes)))
    return tuple(sorted(states, reverse=True))


def occupations(n_modes: int, n_photons: int) -> np.ndarray:
    """Return the ``(d, m)`` occupation matrix of :func:`fock_basis`."""
    return np.array(fock_basis(n_modes, n_photons), dtype=float)


def first_fock_state(n_modes: int, n_photons: int) -> tuple[int, ...]:
    """Return |1^N 0^(m-N)>: one photon in each of the first N modes.

    Raises
    ------
    ValueError
        If ``n_photons > n_modes`` (the state would need bunching).
    """
    if n_photons > n_modes:
        raise ValueError(f"n_photons={n_photons} exceeds n_modes={n_modes}")
    return tuple([1] * n_photons + [0] * (n_modes - n_photons))


@cache
def hopping_operators(n_modes: int, n_photons: int) -> tuple:
    """Return sparse ``a_i^dag a_j`` on the N-photon space, indexed ``[i][j]``."""
    basis = fock_basis(n_modes, n_photons)
    index = {state: k for k, state in enumerate(basis)}
    d = len(basis)
    operators = []
    for i in range(n_modes):
        row = []
        for j in range(n_modes):
            rows, cols, values = [], [], []
            for k, state in enumerate(basis):
                if state[j] == 0:
                    continue
                target = list(state)
                amplitude = math.sqrt(target[j])
                target[j] -= 1
                amplitude *= math.sqrt(target[i] + 1)
                target[i] += 1
                rows.append(index[tuple(target)])
                cols.append(k)
                values.append(amplitude)
            row.append(scipy.sparse.csr_matrix((values, (rows, cols)), shape=(d, d)))
        operators.append(tuple(row))
    return tuple(operators)


def fock_operator(single_particle: np.ndarray, n_photons: int) -> np.ndarray:
    """Return the dense N-photon matrix of ``sum_ij h_ij a_i^dag a_j``."""
    n_modes = single_particle.shape[0]
    operators = hopping_operators(n_modes, n_photons)
    d = len(fock_basis(n_modes, n_photons))
    total = scipy.sparse.csr_matrix((d, d), dtype=complex)
    for i in range(n_modes):
        for j in range(n_modes):
            if single_particle[i, j] != 0:
                total = total + single_particle[i, j] * operators[i][j]
    return total.toarray()


def passive_generators(n_modes: int) -> tuple[list[str], np.ndarray]:
    """Return nearest-neighbour passive generators as single-particle matrices.

    Order: ``n_i`` (m), then ``a_i^dag a_{i+1} + h.c.`` (m - 1), then
    ``i(a_i^dag a_{i+1} - h.c.)`` (m - 1); J = 3m - 2. These are the photonic
    counterparts of the paper's local Pauli term sets.
    """
    labels, matrices = [], []
    for i in range(n_modes):
        matrix = np.zeros((n_modes, n_modes), dtype=complex)
        matrix[i, i] = 1.0
        labels.append(f"n{i}")
        matrices.append(matrix)
    for i in range(n_modes - 1):
        matrix = np.zeros((n_modes, n_modes), dtype=complex)
        matrix[i, i + 1] = matrix[i + 1, i] = 1.0
        labels.append(f"X{i},{i + 1}")
        matrices.append(matrix)
    for i in range(n_modes - 1):
        matrix = np.zeros((n_modes, n_modes), dtype=complex)
        matrix[i, i + 1] = 1j
        matrix[i + 1, i] = -1j
        labels.append(f"Y{i},{i + 1}")
        matrices.append(matrix)
    return labels, np.array(matrices)


def nearest_neighbour_kerr(n_modes: int, n_photons: int) -> np.ndarray:
    """Return the diagonal of ``sum_i n_i n_{i+1}`` (cross-Kerr chain) on the N-photon space."""
    occupation = occupations(n_modes, n_photons)
    return np.sum(occupation[:, :-1] * occupation[:, 1:], axis=1)


def _permanents(matrices: np.ndarray) -> np.ndarray:
    """Ryser permanents of a batch of ``(B, k, k)`` matrices."""
    k = matrices.shape[-1]
    subsets = np.array(list(itertools.product((0, 1), repeat=k)), dtype=float)[1:]
    row_sums = matrices @ subsets.T
    signs = (-1.0) ** (k - subsets.sum(axis=1))
    return np.prod(row_sums, axis=1) @ signs


def interferometer_state(
    unitary: np.ndarray, input_state: tuple[int, ...], n_photons: int
) -> np.ndarray:
    """Return Phi(V)|n_in> on the N-photon basis, with a_j^dag -> sum_i V_ij a_i^dag.

    Amplitude <n_out|Phi(V)|n_in> = Perm(V[rows(n_out), cols(n_in)]) /
    sqrt(prod n_out! prod n_in!), where rows/cols repeat mode i n_i times.
    """
    n_modes = unitary.shape[0]
    basis = fock_basis(n_modes, n_photons)
    columns = np.repeat(np.arange(n_modes), input_state)
    rows = np.array([np.repeat(np.arange(n_modes), state) for state in basis])
    submatrices = unitary[rows[:, :, None], columns[None, None, :]]
    norms = np.array(
        [
            math.sqrt(
                math.prod(math.factorial(n) for n in state)
                * math.prod(math.factorial(n) for n in input_state)
            )
            for state in basis
        ]
    )
    return _permanents(submatrices) / norms


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class PhotonicDataset:
    """Full-batch dataset of pure N-photon states.

    Parameters
    ----------
    states : numpy.ndarray
        Complex amplitudes on :func:`fock_basis`, shape ``(M, d)``.
    targets : numpy.ndarray
        Real regression targets, shape ``(M,)``.
    """

    states: np.ndarray
    targets: np.ndarray
    cache: dict = field(default_factory=dict, repr=False)

    def torch_states(self):
        """Return the states as a ``torch.complex128`` tensor (cached)."""
        import torch

        if "torch" not in self.cache:
            self.cache["torch"] = torch.as_tensor(self.states, dtype=torch.complex128)
        return self.cache["torch"]

    def count_probabilities(self) -> np.ndarray:
        """Return |<n|psi>|^2 in the physical modes, shape ``(M, d)`` (cached)."""
        if "counts" not in self.cache:
            self.cache["counts"] = np.abs(self.states) ** 2
        return self.cache["counts"]

    def correlation_features(self, n_modes: int, n_photons: int) -> np.ndarray:
        """Return the m^2 real components of <a_i^dag a_j> (cached).

        Columns: <n_i> for every i, then <a_i^dag a_j + h.c.> = 2 Re C_ij and
        <i(a_i^dag a_j - h.c.)> = -2 Im C_ij for every pair i < j.
        """
        if "correlations" not in self.cache:
            operators = hopping_operators(n_modes, n_photons)
            conjugate = self.states.conj()

            def expectation(i, j):
                return np.sum(conjugate * (operators[i][j] @ self.states.T).T, axis=1)

            columns = [np.real(expectation(i, i)) for i in range(n_modes)]
            pairs = list(itertools.combinations(range(n_modes), 2))
            values = {pair: expectation(*pair) for pair in pairs}
            columns += [2.0 * np.real(values[pair]) for pair in pairs]
            columns += [-2.0 * np.imag(values[pair]) for pair in pairs]
            self.cache["correlations"] = np.stack(columns, axis=1)
        return self.cache["correlations"]


def haar_interferometer_states(
    n_modes: int, n_photons: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Return ``count`` states Phi(V)|1^N 0^(m-N)> with Haar-random V."""
    input_state = first_fock_state(n_modes, n_photons)
    return np.array(
        [
            interferometer_state(
                unitary_group.rvs(n_modes, random_state=rng), input_state, n_photons
            )
            for _ in range(count)
        ]
    )


def haar_full_states(
    dimension: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Return ``count`` Haar-random pure states on the whole N-photon space."""
    vectors = rng.normal(size=(count, dimension)) + 1j * rng.normal(
        size=(count, dimension)
    )
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def fock_basis_states(n_modes: int, n_photons: int) -> np.ndarray:
    """Return every Fock basis state as an amplitude vector (classical data)."""
    return np.eye(len(fock_basis(n_modes, n_photons)), dtype=complex)


def build_states(
    kind: str, n_modes: int, n_photons: int, count: int | None, rng
) -> np.ndarray:
    """Return a state set by name.

    Raises
    ------
    ValueError
        If ``kind`` is unknown, or ``count`` is given for the ``fock`` set
        (which always holds all d basis states).
    """
    if kind == "fock":
        if count is not None:
            raise ValueError("the 'fock' set holds all d basis states; omit n_states")
        return fock_basis_states(n_modes, n_photons)
    if kind == "haar_interferometer":
        return haar_interferometer_states(n_modes, n_photons, int(count), rng)
    if kind == "haar_full":
        return haar_full_states(len(fock_basis(n_modes, n_photons)), int(count), rng)
    raise ValueError(
        f"Unknown photonic state set {kind!r}; use fock, haar_interferometer or haar_full"
    )


@dataclass
class PassiveTeacher:
    """Target y = Tr[g_T(H*) rho] for a passive H* = sum_j zeta_j G_j."""

    labels: list[str]
    zeta: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    def targets(self, states: np.ndarray, temperature: float) -> np.ndarray:
        """Return targets by dense Fock-space diagonalization (independent of MerLin)."""
        rotated = states @ self.eigenvectors.conj()
        return np.abs(rotated) ** 2 @ np.tanh(self.eigenvalues / temperature)


def build_teacher(
    n_modes: int, n_photons: int, target_range: float, rng: np.random.Generator
) -> PassiveTeacher:
    """Draw zeta ~ U[-r, r] over the nearest-neighbour passive generators."""
    from .trainability import hermitian_eigh

    labels, generators = passive_generators(n_modes)
    zeta = rng.uniform(-target_range, target_range, size=len(labels))
    single_particle = np.einsum("j,jab->ab", zeta, generators)
    eigenvalues, eigenvectors = hermitian_eigh(
        fock_operator(single_particle, n_photons)
    )
    return PassiveTeacher(labels, zeta, eigenvalues, eigenvectors)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _check_squared(loss: str) -> None:
    if loss != "squared":
        raise ValueError(
            f"Photonic models implement the squared loss only, got {loss!r}"
        )


class MerlinPassiveNeuron:
    """Photonic Fermi-Dirac neuron: MerLin interferometer + PNR detection + g(eps . n).

    Parameters
    ----------
    n_modes, n_photons : int
        Modes m and photons N (N >= 2 for a scientific result).
    activation : str
        ``"tanh"`` (g_T = tanh(x / T), the Fermi-Dirac neuron) or ``"linear"``
        (output Tr[H rho], baseline B3).
    temperature : float
        T of the activation.

    Notes
    -----
    Parameters are ``[mesh phases (m(m-1)), energies eps (m)]``. The mesh is
    ``CircuitBuilder.add_entangling_layer()``, a universal MZI mesh; its m
    output phases are absent because they do not change photon counts.
    """

    def __init__(
        self, n_modes: int, n_photons: int, activation: str, temperature: float
    ) -> None:
        import merlin as ml
        import torch

        self.n_modes, self.n_photons = int(n_modes), int(n_photons)
        self.temperature = float(temperature)
        if activation not in ("tanh", "linear"):
            raise ValueError(f"activation must be tanh or linear, got {activation!r}")
        self.activation = activation
        builder = ml.CircuitBuilder(n_modes=self.n_modes)
        builder.add_entangling_layer()
        self.layer = ml.QuantumLayer(
            builder=builder,
            input_state=list(first_fock_state(self.n_modes, self.n_photons)),
            n_photons=self.n_photons,
            dtype=torch.float64,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.FOCK
            ),
        )
        keys = tuple(tuple(int(n) for n in key) for key in self.layer.output_keys)
        if keys != fock_basis(self.n_modes, self.n_photons):
            raise ValueError(
                "MerLin output_keys differ from fock_basis ordering; amplitudes would be mislabelled"
            )
        self.phase_parameters = [
            parameter
            for parameter in self.layer.parameters()
            if parameter.requires_grad
        ]
        self.n_phases = sum(parameter.numel() for parameter in self.phase_parameters)
        self.occupation = torch.as_tensor(
            occupations(self.n_modes, self.n_photons), dtype=torch.float64
        )
        self.labels = [f"phase{k}" for k in range(self.n_phases)] + [
            f"eps{i}" for i in range(self.n_modes)
        ]

    @property
    def n_parameters(self) -> int:
        """Mesh phases plus mode energies."""
        return self.n_phases + self.n_modes

    def set_phases(self, phases) -> None:
        """Copy a flat phase vector into the MerLin mesh parameters."""
        import torch

        phases = torch.as_tensor(np.asarray(phases, dtype=float), dtype=torch.float64)
        offset = 0
        with torch.no_grad():
            for parameter in self.phase_parameters:
                size = parameter.numel()
                parameter.copy_(phases[offset : offset + size].reshape(parameter.shape))
                offset += size

    def unitary(self) -> np.ndarray:
        """Return the single-photon unitary U of the current mesh."""
        converter = self.layer.computation_process.converter
        tensor = converter.to_tensor(*self.phase_parameters).detach().numpy()
        return tensor.reshape(self.n_modes, self.n_modes)

    def probabilities(self, data: PhotonicDataset):
        """Return the PNR outcome distribution after the mesh, shape ``(M, d)``."""
        return self.layer(data.torch_states())

    def _readout(self, energies, temperature: float):
        import torch

        if self.activation == "tanh":
            return torch.tanh(energies / temperature)
        return energies

    def _forward(self, theta: np.ndarray, data: PhotonicDataset, requires_grad: bool):
        import torch

        self.set_phases(theta[: self.n_phases])
        for parameter in self.phase_parameters:
            parameter.grad = None
            parameter.requires_grad_(requires_grad)
        energies_weights = torch.tensor(
            theta[self.n_phases :], dtype=torch.float64, requires_grad=requires_grad
        )
        probabilities = self.probabilities(data)
        values = self._readout(self.occupation @ energies_weights, self.temperature)
        return probabilities @ values, energies_weights

    def outputs(self, theta: np.ndarray, data: PhotonicDataset) -> np.ndarray:
        """Return sum_n p_U(n) g(eps . n) for every state."""
        import torch

        with torch.no_grad():
            predictions, _ = self._forward(theta, data, requires_grad=False)
        return predictions.numpy()

    def loss_and_gradient(
        self, theta: np.ndarray, data: PhotonicDataset, loss: str
    ) -> tuple[float, np.ndarray]:
        """Return the mean squared loss and its exact gradient (torch autograd through MerLin)."""
        import torch

        _check_squared(loss)
        predictions, energies_weights = self._forward(theta, data, requires_grad=True)
        targets = torch.as_tensor(data.targets, dtype=torch.float64)
        value = torch.mean((predictions - targets) ** 2)
        value.backward()
        phase_gradient = torch.cat(
            [parameter.grad.reshape(-1) for parameter in self.phase_parameters]
        )
        gradient = torch.cat([phase_gradient, energies_weights.grad]).numpy()
        return float(value.detach()), gradient

    def readout_outputs(
        self, theta: np.ndarray, data: PhotonicDataset, readout: str
    ) -> np.ndarray:
        """Return outputs under a detector model (hardware ablation, evaluation only).

        ``pnr``: photon-number-resolving (exact). ``threshold``: clicks
        c_k = min(n_k, 1) replace n_k in the energy. ``unbunched``: only
        collision-free outcomes are kept and renormalized (postselection).

        Raises
        ------
        ValueError
            If ``readout`` is unknown.
        """
        import torch

        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        with torch.no_grad():
            self.set_phases(theta[: self.n_phases])
            probabilities = self.probabilities(data).numpy()
        energies_weights = theta[self.n_phases :]
        occupation = self.occupation.numpy()
        if readout == "threshold":
            occupation = np.minimum(occupation, 1.0)
        values = self._readout(
            torch.as_tensor(occupation @ energies_weights), self.temperature
        ).numpy()
        if readout == "unbunched":
            collision_free = np.all(self.occupation.numpy() <= 1.0, axis=1)
            kept = probabilities[:, collision_free]
            return kept @ values[collision_free] / kept.sum(axis=1)
        return probabilities @ values

    def sampled_outputs(
        self,
        theta: np.ndarray,
        data: PhotonicDataset,
        shots: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return finite-shot estimates: mean of g(eps . n) over ``shots`` PNR samples per state."""
        import torch

        with torch.no_grad():
            self.set_phases(theta[: self.n_phases])
            probabilities = self.probabilities(data).numpy()
        values = self._readout(
            self.occupation @ torch.as_tensor(theta[self.n_phases :]), self.temperature
        ).numpy()
        estimates = np.empty(len(probabilities))
        for index, distribution in enumerate(probabilities):
            distribution = np.clip(distribution, 0.0, None)
            counts = rng.multinomial(shots, distribution / distribution.sum())
            estimates[index] = counts @ values / shots
        return estimates


class PhotonCountNeuron:
    """Commuting classical neuron on photon counts: sum_n p(n) g(w . n + sum_{i<j} W_ij n_i n_j).

    The Hamiltonian sum_i w_i n_i + sum_{i<j} W_ij n_i n_j is diagonal in the
    physical Fock basis, so the model is the classical second-order neuron
    (eq. 10) averaged over the measured photon-count distribution (eq. 12
    analogue); counterpart of the paper's IM / FCIM models.
    """

    def __init__(self, n_modes: int, n_photons: int, temperature: float) -> None:
        self.n_modes, self.n_photons = int(n_modes), int(n_photons)
        occupation = occupations(self.n_modes, self.n_photons)
        pairs = list(itertools.combinations(range(self.n_modes), 2))
        pair_columns = [occupation[:, i] * occupation[:, j] for i, j in pairs]
        self.design = np.column_stack([occupation, *pair_columns])
        self.labels = [f"w{i}" for i in range(self.n_modes)] + [
            f"W{i},{j}" for i, j in pairs
        ]
        self.activation = make_activation("tanh", temperature)

    @property
    def n_parameters(self) -> int:
        """m linear plus m(m-1)/2 pair weights."""
        return self.design.shape[1]

    def outputs(self, theta: np.ndarray, data: PhotonicDataset) -> np.ndarray:
        """Return the photon-count average of the classical neuron."""
        return data.count_probabilities() @ self.activation.value(self.design @ theta)

    def output_gradients(self, theta: np.ndarray, data: PhotonicDataset) -> np.ndarray:
        """Return d f_m / d theta, shape ``(M, P)``."""
        slopes = self.activation.derivative(self.design @ theta)
        return data.count_probabilities() @ (slopes[:, None] * self.design)

    def loss_and_gradient(
        self, theta: np.ndarray, data: PhotonicDataset, loss: str
    ) -> tuple[float, np.ndarray]:
        """Return the mean squared loss and its exact gradient."""
        _check_squared(loss)
        residuals = self.outputs(theta, data) - data.targets
        gradient = 2.0 * residuals @ self.output_gradients(theta, data) / len(residuals)
        return float(np.mean(residuals**2)), gradient


class CorrelationNeuron:
    """Classical neuron tanh(sum_k theta_k F_k / T) on one-photon correlations (B2 counterpart).

    F are the m^2 real components of <a_i^dag a_j>, i.e. the expectations of
    every passive generator; m^2 parameters, as many as
    :class:`MerlinPassiveNeuron`.
    """

    def __init__(self, n_modes: int, n_photons: int, temperature: float) -> None:
        self.n_modes, self.n_photons = int(n_modes), int(n_photons)
        pairs = list(itertools.combinations(range(self.n_modes), 2))
        self.labels = (
            [f"n{i}" for i in range(self.n_modes)]
            + [f"X{i},{j}" for i, j in pairs]
            + [f"Y{i},{j}" for i, j in pairs]
        )
        self.activation = make_activation("tanh", temperature)

    @property
    def n_parameters(self) -> int:
        """m^2 correlation weights."""
        return len(self.labels)

    def outputs(self, theta: np.ndarray, data: PhotonicDataset) -> np.ndarray:
        """Return tanh(F theta / T) for every state."""
        features = data.correlation_features(self.n_modes, self.n_photons)
        return self.activation.value(features @ theta)

    def loss_and_gradient(
        self, theta: np.ndarray, data: PhotonicDataset, loss: str
    ) -> tuple[float, np.ndarray]:
        """Return the mean squared loss and its exact gradient."""
        _check_squared(loss)
        features = data.correlation_features(self.n_modes, self.n_photons)
        pre_activation = features @ theta
        residuals = self.activation.value(pre_activation) - data.targets
        weights = 2.0 * residuals * self.activation.derivative(pre_activation)
        return float(np.mean(residuals**2)), features.T @ weights / len(residuals)


def build_photonic_model(
    model_cfg: dict, n_modes: int, n_photons: int, temperature: float
):
    """Return the model named by ``model_cfg["kind"]``.

    Raises
    ------
    ValueError
        If the kind is unknown.
    """
    kind = model_cfg["kind"]
    if kind == "merlin_passive":
        return MerlinPassiveNeuron(
            n_modes, n_photons, model_cfg["activation"], temperature
        )
    if kind == "photon_count":
        return PhotonCountNeuron(n_modes, n_photons, temperature)
    if kind == "correlation":
        return CorrelationNeuron(n_modes, n_photons, temperature)
    raise ValueError(
        f"Unknown photonic model kind {kind!r}; use merlin_passive, photon_count or correlation"
    )
