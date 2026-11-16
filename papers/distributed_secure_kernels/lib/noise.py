"""The authors' depolarizing noise model and an exact noisy swap-test probability.

Noise model (``library.py::setup_noise_model``): ``depolarizing_error(rate, 1)``
on ``u1, u2, u3`` and ``depolarizing_error(rate, 2)`` on ``cx``; level 1 is
rate 0.001, level 2 is rate 0.01 (paper Sec. 7.2). Aer applies each error
after its gate; measurements, resets and ``initialize`` are noiseless. When
this noise model is attached, transpiling against the simulator decomposes
``h``, ``x``, ``z`` and ``cswap`` into ``u1/u2/u3/cx`` (so they become noisy),
while ``initialize`` stays a native, noiseless instruction.

Exact noisy probability (replaces the per-shot simulation of every kernel
entry):

1. Each per-qubit teleportation touches only its own (data, helper, server)
   triple, so the teleportation of an n-qubit state is a product channel
   ``T^{(x)n}``. ``T`` is obtained by simulating the transpiled 3-qubit
   teleportation (with its mid-circuit measurements and conditioned gates)
   on the four inputs ``|i><j|``.
2. The swap test is a fixed channel followed by a projective measurement of
   the ancilla, so ``p0 = Tr[O (rho_A (x) rho_B)]`` for one observable ``O``,
   obtained once by evolving ``|0><0|_ancilla (x) I`` backwards (Heisenberg
   picture) through the transpiled, noisy swap-test gates.

Then for every pair ``p0(a, b) = Tr[O (T^n(|a><a|) (x) T^n(|b><b|))]``, a
bilinear contraction over all pairs. ``tests/test_noise.py`` checks the
construction against per-shot Aer runs of the full distributed circuit.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Paper Sec. 7.2 / library.py: noise level code -> depolarizing rate.
NOISE_LEVEL_RATES = {0: 0.0, 1: 0.001, 2: 0.01}
NOISY_ONE_QUBIT_GATES = ("u1", "u2", "u3")
NOISY_TWO_QUBIT_GATES = ("cx",)
NOISELESS_INSTRUCTIONS = ("measure", "reset", "barrier", "initialize")


def rate_for_level(noise_level: int) -> float:
    """Depolarizing rate of an authors' noise level (0, 1 or 2)."""

    if noise_level not in NOISE_LEVEL_RATES:
        raise ValueError(
            f"noise level must be one of {sorted(NOISE_LEVEL_RATES)}, got {noise_level}"
        )
    return NOISE_LEVEL_RATES[noise_level]


def depolarizing_noise_model(rate: float) -> NoiseModel | None:
    """The authors' noise model for an arbitrary depolarizing rate (None if ``rate == 0``)."""

    if rate == 0:
        return None
    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(rate, 1), list(NOISY_ONE_QUBIT_GATES)
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error(rate, 2), list(NOISY_TWO_QUBIT_GATES)
    )
    return model


def authors_noise_model(noise_level: int) -> NoiseModel | None:
    """The authors' ``setup_noise_model(noise_level)`` (None for level 0)."""

    return depolarizing_noise_model(rate_for_level(noise_level))


def _transpiled(circuit: QuantumCircuit, rate: float) -> QuantumCircuit:
    model = depolarizing_noise_model(rate)
    simulator = AerSimulator() if model is None else AerSimulator(noise_model=model)
    return transpile(circuit, simulator)


def _error_channel(name: str, rate: float) -> Kraus | None:
    if rate == 0:
        return None
    if name in NOISY_ONE_QUBIT_GATES:
        return Kraus(depolarizing_error(rate, 1).to_quantumchannel())
    if name in NOISY_TWO_QUBIT_GATES:
        return Kraus(depolarizing_error(rate, 2).to_quantumchannel())
    return None


def _check_supported(name: str, rate: float) -> None:
    if rate == 0:
        return  # without noise every gate is exact, whatever the transpiled basis
    known = NOISY_ONE_QUBIT_GATES + NOISY_TWO_QUBIT_GATES + NOISELESS_INSTRUCTIONS
    if name not in known + ("if_else",):
        # Any other gate would be noiseless in Aer too; refuse it so that a change
        # of transpiled basis cannot silently remove noise from the model.
        raise ValueError(f"Unexpected transpiled instruction {name!r}")


# --------------------------------------------------------------------------- #
# 1. Per-qubit teleportation channel
# --------------------------------------------------------------------------- #


def _teleport_circuit() -> QuantumCircuit:
    """One per-qubit teleportation of the authors' circuit: data 0, helper 1, server 2."""

    from .protocol import _teleport_qubit

    circuit = QuantumCircuit(3, 2)
    _teleport_qubit(circuit, data=0, helper=1, server=2, bit_data=0, bit_helper=1)
    return circuit


def _evolve_branches(
    branches: dict[tuple[int, ...], DensityMatrix],
    circuit: QuantumCircuit,
    qubit_map: list[int],
    clbit_map: list[int],
    rate: float,
) -> dict[tuple[int, ...], DensityMatrix]:
    """Evolve a classical-quantum state ``{clbit values: unnormalised rho}`` through ``circuit``."""

    for instruction in circuit.data:
        operation = instruction.operation
        name = operation.name
        _check_supported(name, rate)
        qargs = [qubit_map[circuit.find_bit(q).index] for q in instruction.qubits]
        cargs = [clbit_map[circuit.find_bit(c).index] for c in instruction.clbits]
        if name == "barrier":
            continue
        if name == "measure":
            (qubit,), (clbit,) = qargs, cargs
            projectors = [Operator(np.diag([1.0, 0.0])), Operator(np.diag([0.0, 1.0]))]
            updated: dict[tuple[int, ...], DensityMatrix] = {}
            for bits, rho in branches.items():
                for outcome, projector in enumerate(projectors):
                    new_bits = list(bits)
                    new_bits[clbit] = outcome
                    part = DensityMatrix(rho.evolve(projector, [qubit]).data)
                    key = tuple(new_bits)
                    updated[key] = (
                        part
                        if key not in updated
                        else DensityMatrix(updated[key].data + part.data)
                    )
            branches = updated
        elif name == "if_else":
            (clbit_ref, value) = operation.condition
            clbit = clbit_map[circuit.find_bit(clbit_ref).index]
            true_body, false_body = (
                operation.blocks[0],
                (operation.blocks[1] if len(operation.blocks) > 1 else None),
            )
            updated = {}
            for bits, rho in branches.items():
                body = true_body if bits[clbit] == value else false_body
                if body is None:
                    updated[bits] = rho
                    continue
                body_qubits = qargs
                body_clbits = cargs
                updated.update(
                    _evolve_branches({bits: rho}, body, body_qubits, body_clbits, rate)
                )
            branches = updated
        elif name == "reset":
            raise ValueError("The teleportation channel is computed before the resets")
        else:
            unitary = Operator(operation)
            error = _error_channel(name, rate)
            evolved = {}
            for bits, rho in branches.items():
                rho = rho.evolve(unitary, qargs)
                if error is not None:
                    rho = rho.evolve(error, qargs)
                evolved[bits] = rho
            branches = evolved
    return branches


@cache
def teleport_transfer_tensor(rate: float) -> np.ndarray:
    """Single-qubit teleportation channel ``T`` as ``T[a, b, i, j]``.

    ``rho_out[a, b] = sum_ij T[a, b, i, j] rho_in[i, j]``, where ``rho_in`` is the
    client's data qubit and ``rho_out`` the server qubit after the corrections.
    Without noise ``T`` is conjugation by ``H`` (the authors' extra Hadamard).
    """

    compiled = _transpiled(_teleport_circuit(), rate)
    tensor = np.zeros((2, 2, 2, 2), dtype=complex)
    zero = np.diag([1.0, 0.0])
    for i in range(2):
        for j in range(2):
            data_input = np.zeros((2, 2))
            data_input[i, j] = 1.0
            # Qiskit little-endian: qubit 0 (data) is the rightmost Kronecker factor.
            start = DensityMatrix(np.kron(np.kron(zero, zero), data_input))
            branches = _evolve_branches(
                {(0, 0): start},
                compiled,
                list(range(compiled.num_qubits)),
                list(range(compiled.num_clbits)),
                rate,
            )
            total = sum(rho.data for rho in branches.values())
            server = partial_trace(DensityMatrix(total), [0, 1]).data
            tensor[:, :, i, j] = server
    return tensor


def apply_teleport_channel(states: np.ndarray, rate: float) -> np.ndarray:
    """Teleported density matrices ``T^{(x)n}(rho)`` for each sample.

    Parameters
    ----------
    states : numpy.ndarray
        Encoded amplitude vectors, shape ``(n_samples, 2**n)``, or density
        matrices, shape ``(n_samples, 2**n, 2**n)``.
    rate : float
        Depolarizing rate.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, 2**n, 2**n)``.
    """

    n_samples, dimension = states.shape[:2]
    n = int(np.log2(dimension))
    tensor = teleport_transfer_tensor(rate)
    if states.ndim == 2:
        rho = np.einsum("si,sj->sij", states, states.conj()).astype(complex)
    else:
        rho = states.astype(complex)
    rho = rho.reshape((n_samples,) + (2,) * (2 * n))
    for qubit in range(n):
        # Axis of qubit q in a little-endian reshaped matrix: rows n-q, columns 2n-q
        # (axis 0 is the sample index).
        row_axis, col_axis = n - qubit, 2 * n - qubit
        rho = np.moveaxis(rho, (row_axis, col_axis), (-2, -1))
        rho = np.einsum("abij,...ij->...ab", tensor, rho)
        rho = np.moveaxis(rho, (-2, -1), (row_axis, col_axis))
    return rho.reshape(n_samples, dimension, dimension)


# --------------------------------------------------------------------------- #
# 2. Swap-test observable
# --------------------------------------------------------------------------- #


def _swap_test_circuit(n: int) -> QuantumCircuit:
    """Server part of the distributed circuit on 2n + 1 qubits.

    Qubits 0..n-1 hold Bob's teleported state (the authors' register 0),
    n..2n-1 Alice's (register 2), 2n the ancilla.
    """

    circuit = QuantumCircuit(2 * n + 1)
    ancilla = 2 * n
    circuit.h(ancilla)
    for k in range(n):
        circuit.cswap(ancilla, n + k, k)
    circuit.h(ancilla)
    return circuit


@cache
def swap_test_observable(n: int, rate: float) -> np.ndarray:
    """Observable ``O`` with ``p0 = Tr[O (rho_Alice (x) rho_Bob)]`` (dimension ``4**n``).

    Obtained by evolving ``|0><0|_ancilla (x) I`` backwards through the transpiled
    swap-test gates and their depolarizing errors, then keeping the ancilla
    ``<0| . |0>`` block (the ancilla starts in ``|0>``).
    """

    compiled = _transpiled(_swap_test_circuit(n), rate)
    total_qubits = 2 * n + 1
    projector = np.kron(
        np.diag([1.0, 0.0]), np.eye(2 ** (2 * n))
    )  # ancilla is the top qubit
    heisenberg = DensityMatrix(projector.astype(complex))
    for instruction in reversed(compiled.data):
        operation = instruction.operation
        _check_supported(operation.name, rate)
        if operation.name in ("barrier", "measure"):
            continue
        qargs = [compiled.find_bit(q).index for q in instruction.qubits]
        error = _error_channel(operation.name, rate)
        if error is not None:
            heisenberg = heisenberg.evolve(error.adjoint(), qargs)
        heisenberg = heisenberg.evolve(Operator(operation).adjoint(), qargs)
    block = 2 ** (2 * n)
    observable = heisenberg.data[:block, :block]
    if total_qubits != compiled.num_qubits:
        raise ValueError("Transpilation changed the number of qubits")
    return observable


def noisy_prepared_states(states: np.ndarray, rate: float) -> np.ndarray:
    """Density matrices of ``initialize(state)`` decomposed into the noisy basis.

    Diagnostic variant (not the authors' setting under Qiskit 2.5, where
    ``initialize`` stays native and noiseless): an older Qiskit may have
    decomposed state preparation into ``u1/u2/u3/cx`` once a noise model was
    attached, so preparation itself would be noisy. Each sample's preparation
    circuit is transpiled to that basis and simulated as a density matrix with
    the authors' depolarizing errors.
    """

    basis = list(NOISY_ONE_QUBIT_GATES + NOISY_TWO_QUBIT_GATES) + ["reset"]
    n = int(np.log2(states.shape[1]))
    reset_kraus = Kraus([np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]])])
    prepared = np.empty((len(states), 2**n, 2**n), dtype=complex)
    for index, state in enumerate(states):
        circuit = QuantumCircuit(n)
        circuit.initialize(state, list(range(n)))
        compiled = transpile(circuit, basis_gates=basis)
        rho = DensityMatrix.from_label("0" * n)
        for instruction in compiled.data:
            operation = instruction.operation
            qargs = [compiled.find_bit(q).index for q in instruction.qubits]
            if operation.name == "barrier":
                continue
            if operation.name == "reset":
                rho = rho.evolve(reset_kraus, qargs)
                continue
            _check_supported(operation.name, rate)
            rho = rho.evolve(Operator(operation), qargs)
            error = _error_channel(operation.name, rate)
            if error is not None:
                rho = rho.evolve(error, qargs)
        prepared[index] = rho.data
    return prepared


def noisy_zero_probability(
    alice_states: np.ndarray,
    bob_states: np.ndarray,
    rate: float,
    teleport: bool = True,
    noisy_initialize: bool = False,
) -> np.ndarray:
    """Exact swap-test ``p0`` of the authors' circuit for every pair.

    Parameters
    ----------
    alice_states : numpy.ndarray
        Encoded rows sent by Alice (kernel rows), shape ``(n_a, 2**n)``.
    bob_states : numpy.ndarray
        Encoded rows sent by Bob (kernel columns), shape ``(n_b, 2**n)``.
    rate : float
        Depolarizing rate; 0 gives ``1/2 + 1/2 |<a|b>|^2``.
    teleport : bool
        True for the distributed circuit; False for the centralised swap test
        (states prepared on the server, no teleportation). Default value is True.
    noisy_initialize : bool
        Diagnostic variant with noisy, decomposed state preparation (see
        :func:`noisy_prepared_states`). Default value is False.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_a, n_b)``.
    """

    dimension = alice_states.shape[1]
    n = int(np.log2(dimension))
    observable = swap_test_observable(n, rate).reshape(
        dimension, dimension, dimension, dimension
    )
    if noisy_initialize:
        same = alice_states.shape == bob_states.shape and np.array_equal(
            alice_states, bob_states
        )
        alice_states = noisy_prepared_states(alice_states, rate)
        bob_states = alice_states if same else noisy_prepared_states(bob_states, rate)
        if not teleport:
            raise ValueError(
                "noisy_initialize is only modelled for the distributed circuit"
            )
    if teleport:
        rho_alice = apply_teleport_channel(alice_states, rate)
        rho_bob = apply_teleport_channel(bob_states, rate)
    else:
        rho_alice = np.einsum("si,sj->sij", alice_states, alice_states.conj()).astype(
            complex
        )
        rho_bob = np.einsum("si,sj->sij", bob_states, bob_states.conj()).astype(complex)
    # O[(p, q), (r, s)] with p, r Alice indices and q, s Bob indices:
    # p0 = sum O[p, q, r, s] rho_A[r, p] rho_B[s, q].
    partial = np.einsum("pqrs,arp->aqs", observable, rho_alice, optimize=True)
    probability = np.einsum("aqs,bsq->ab", partial, rho_bob, optimize=True)
    if np.max(np.abs(probability.imag)) > 1e-9:
        raise ValueError("Swap-test probability has a non-negligible imaginary part")
    return probability.real
