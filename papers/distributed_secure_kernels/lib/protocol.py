"""The authors' circuits (``library.py``), ported to Qiskit 2.x, and per-shot Aer runs.

Distributed circuit (``teleport_and_inner_product``), 4n + 1 qubits:

- register 0 (qubits 0..n-1): Alice's encoded sample; reused for Bob's teleported state;
- register 1 (n..2n-1): helper half of each Bell pair;
- register 2 (2n..3n-1): server half; receives Alice's state;
- register 3 (3n..4n-1): Bob's encoded sample;
- qubit 4n: server ancilla of the swap test (measured into clbit 4n).

Per qubit k, Alice's teleportation is the authors' gate list
``h(H) cx(H,S) h(A) cx(A,H) h(A) measure(A), measure(H), Z^{m_A} X^{m_H} on S``.
The first ``h(A)`` is not part of standard teleportation: it teleports
``H|psi>`` instead of ``|psi>``. Bob's teleportation has the same extra gate,
so the overlap is unchanged without noise; it adds noisy gates with noise.

The paper (Sec. 6.2, App. A) describes 6n + 1 qubits and 2n Bell pairs; the
code reuses registers instead. ``c_if`` (removed in Qiskit 2.0) is replaced
by ``if_test`` blocks.

Centralised circuit (``only_inner_product``): both samples encoded directly on
2n qubits plus the ancilla, swap test only.
"""

from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

from .noise import depolarizing_noise_model


def _teleport_qubit(
    circuit: QuantumCircuit,
    data: int,
    helper: int,
    server: int,
    bit_data: int,
    bit_helper: int,
) -> None:
    """Append one teleportation of the authors' circuit (gate order of ``library.py``)."""

    circuit.h(helper)
    circuit.cx(helper, server)
    circuit.h(data)  # authors' extra Hadamard (see module docstring)
    circuit.cx(data, helper)
    circuit.h(data)
    circuit.measure(data, bit_data)
    circuit.measure(helper, bit_helper)
    with circuit.if_test((circuit.clbits[bit_data], 1)):
        circuit.z(server)
    with circuit.if_test((circuit.clbits[bit_helper], 1)):
        circuit.x(server)


def distributed_circuit(
    alice_state: np.ndarray, bob_state: np.ndarray
) -> QuantumCircuit:
    """Authors' distributed teleportation + swap-test circuit for one pair.

    Parameters
    ----------
    alice_state, bob_state : numpy.ndarray
        Encoded amplitude vectors of length ``2**n`` (see :mod:`lib.encoding`).

    Returns
    -------
    qiskit.QuantumCircuit
        4n + 1 qubits, 4n + 1 clbits; the ancilla is measured into the last clbit.
    """

    n = int(np.log2(len(alice_state)))
    total = 4 * n + 1
    circuit = QuantumCircuit(total, total)
    circuit.initialize(alice_state, list(range(n)))
    circuit.initialize(bob_state, list(range(3 * n, 4 * n)))
    for k in range(n):
        # Alice: register 0 -> register 2 through the Bell pair (register 1, register 2).
        _teleport_qubit(circuit, k, n + k, 2 * n + k, k, n + k)
        circuit.reset(k)
        circuit.reset(n + k)
    for k in range(n):
        # Bob: register 3 -> register 0 through the Bell pair (register 1, register 0).
        _teleport_qubit(circuit, 3 * n + k, n + k, k, 2 * n + k, 3 * n + k)
    ancilla = total - 1
    circuit.h(ancilla)
    for k in range(n):
        circuit.cswap(ancilla, 2 * n + k, k)
    circuit.h(ancilla)
    circuit.measure(ancilla, total - 1)
    return circuit


def centralised_circuit(
    left_state: np.ndarray, right_state: np.ndarray
) -> QuantumCircuit:
    """Authors' centralised swap test (``only_inner_product``): 2n + 1 qubits, 1 clbit."""

    n = int(np.log2(len(left_state)))
    total = 2 * n + 1
    circuit = QuantumCircuit(total, 1)
    circuit.initialize(left_state, list(range(n)))
    circuit.initialize(right_state, list(range(n, 2 * n)))
    ancilla = total - 1
    circuit.h(ancilla)
    for k in range(n):
        circuit.cswap(ancilla, n + k, k)
    circuit.h(ancilla)
    circuit.measure(ancilla, 0)
    return circuit


def figure2_circuit(alice_state: np.ndarray, bob_state: np.ndarray) -> QuantumCircuit:
    """The paper's Fig. 2 as drawn: separate registers, 6n + 1 qubits, textbook teleportation.

    Unlike :func:`distributed_circuit` (the authors' code), there is no register
    reuse and no extra Hadamard. Registers: Alice ``A``, helper halves ``HA`` /
    ``HB``, server halves ``SA`` / ``SB``, Bob ``B``, ancilla ``a``; the ancilla
    is measured into the last clbit, as :func:`run_ancilla_zeros` expects.

    Parameters
    ----------
    alice_state, bob_state : numpy.ndarray
        Encoded amplitude vectors of length ``2**n``.

    Returns
    -------
    qiskit.QuantumCircuit
        6n + 1 qubits, 4n + 1 clbits.
    """

    n = int(np.log2(len(alice_state)))
    a, ha, sa, sb, hb, b = (
        QuantumRegister(n, name) for name in ("A", "HA", "SA", "SB", "HB", "B")
    )
    ancilla = QuantumRegister(1, "a")
    m_a, m_ha, m_hb, m_b = (
        ClassicalRegister(n, name) for name in ("mA", "mHA", "mHB", "mB")
    )
    out = ClassicalRegister(1, "out")
    circuit = QuantumCircuit(a, ha, sa, sb, hb, b, ancilla, m_a, m_ha, m_hb, m_b, out)
    circuit.initialize(alice_state, a)
    circuit.initialize(bob_state, b)
    for k in range(n):
        # Helper: Bell pairs (HA, SA) and (HB, SB).
        circuit.h(ha[k])
        circuit.cx(ha[k], sa[k])
        circuit.h(hb[k])
        circuit.cx(hb[k], sb[k])
    for data, half, target, m_data, m_half in (
        (a, ha, sa, m_a, m_ha),
        (b, hb, sb, m_b, m_hb),
    ):
        for k in range(n):
            # Client: CNOT onto its Bell half, H, measure both; server: X then Z corrections.
            circuit.cx(data[k], half[k])
            circuit.h(data[k])
            circuit.measure(data[k], m_data[k])
            circuit.measure(half[k], m_half[k])
            with circuit.if_test((m_half[k], 1)):
                circuit.x(target[k])
            with circuit.if_test((m_data[k], 1)):
                circuit.z(target[k])
    circuit.h(ancilla[0])
    for k in range(n):
        circuit.cswap(ancilla[0], sa[k], sb[k])
    circuit.h(ancilla[0])
    circuit.measure(ancilla[0], out[0])
    return circuit


# Aer seeds shot i of a run with ``seed_simulator + i``, so two runs whose seeds
# differ by less than the shot count reuse almost the same random numbers
# (measured: seeds 7, 8, 9, ... give a count variance 0.035x the binomial one).
# Independent runs therefore need seeds at least ``shots`` apart.
SEED_SPACING = 100_000


def independent_seed(check: int, index: int) -> int:
    """Aer seed for run ``index`` of check ``check``, spaced by :data:`SEED_SPACING`."""

    return (check * 1_000 + index) * SEED_SPACING


def simulator_for(rate: float, seed: int | None = None) -> AerSimulator:
    """``AerSimulator`` with the authors' depolarizing model at ``rate`` (none for 0)."""

    noise_model = depolarizing_noise_model(rate)
    options = {} if seed is None else {"seed_simulator": seed}
    if noise_model is None:
        return AerSimulator(**options)
    return AerSimulator(noise_model=noise_model, **options)


def run_ancilla_zeros(
    circuit: QuantumCircuit, shots: int, rate: float, seed: int
) -> int:
    """Run one circuit per shot on Aer and count ancilla outcomes 0.

    The ancilla is the circuit's last clbit, i.e. the leftmost character of each
    Qiskit count key; this is the authors' ``key.startswith('1')`` test.
    """

    simulator = simulator_for(rate, seed)
    # Transpile against the simulator that carries the noise model, as the authors
    # do: the noise model's basis gates decide which gates are decomposed.
    compiled = transpile(circuit, simulator)
    counts = simulator.run(compiled, shots=shots).result().get_counts()
    ones = sum(value for key, value in counts.items() if key.replace(" ", "")[0] == "1")
    return shots - ones


def client_bit_counts(
    circuit: QuantumCircuit, shots: int, seed: int
) -> dict[str, np.ndarray]:
    """Frequencies of each client measurement bit (noiseless), for the security check.

    Returns
    -------
    dict
        ``{"alice": (2n,) array, "bob": (2n,) array}`` with the fraction of shots in
        which each of the client's classical messages was 1. Teleportation makes
        these bits uniform, whatever the encoded data.
    """

    simulator = simulator_for(0.0, seed)
    counts = (
        simulator.run(transpile(circuit, simulator), shots=shots).result().get_counts()
    )
    n = (circuit.num_clbits - 1) // 4
    alice_bits = list(range(0, n)) + list(range(n, 2 * n))
    bob_bits = list(range(2 * n, 3 * n)) + list(range(3 * n, 4 * n))
    ones = np.zeros(circuit.num_clbits)
    for key, value in counts.items():
        bits = key.replace(" ", "")[::-1]  # bits[i] is clbit i
        ones += value * np.array([int(b) for b in bits])
    fraction = ones / shots
    return {"alice": fraction[alice_bits], "bob": fraction[bob_bits]}
