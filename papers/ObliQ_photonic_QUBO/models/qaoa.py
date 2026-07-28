"""QAOA baseline (Qiskit): local statevector sampling, IBM, or Quantum Inspire.

**Reproducibility.** ``SamplingVQE`` draws its initial point from
``qiskit_algorithms``' global generator, and ``StatevectorSampler`` draws shots
from its own. Both are seeded from the ``seed`` argument, so a local QAOA run is
reproducible; hardware backends remain stochastic.
"""

from __future__ import annotations

import os
import time
from multiprocessing import AuthenticationError

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.primitives import BackendSamplerV2, StatevectorSampler
from qiskit.providers.backend import Backend
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    OptimizationResultStatus,
)
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems import QuadraticProgram

HUB = "ibm-q"
GROUP = "open"
PROJECT = "main"

#: Provider names that mean "no hardware, sample locally".
LOCAL_PROVIDERS = {"local simulator", "qasm simulator", "statevector simulator"}


def initialize_backend(
    provider: str | None,
    backend: str | None,
) -> Backend | None:
    """Initialize an optional backend for the IBM or Quantum Inspire providers.

    Args:
        provider: hardware provider name (``"ibm"``, ``"qi"``, or a local alias).
        backend: specific backend name, required for hardware providers.

    Returns:
        A ``Backend`` for hardware providers; ``None`` for local sampling.

    Raises:
        AuthenticationError: if authentication with IBM or QI failed.
        ValueError: if the provider or backend is not among the supported options.
    """
    if provider is None or provider.lower() in LOCAL_PROVIDERS:
        return None

    provider_lower = provider.lower()

    if provider_lower == "ibm":
        if backend is None:
            raise ValueError("Backend name must be provided when using IBM provider.")
        try:
            runtime_service = QiskitRuntimeService(instance=f"{HUB}/{GROUP}/{PROJECT}")
        except Exception as exc:
            raise AuthenticationError("Authentication with IBM failed.") from exc
        try:
            return runtime_service.backend(backend)
        except QiskitBackendNotFoundError as exc:
            raise ValueError(f"Backend {backend} not among possible options.") from exc

    if provider_lower == "qi":
        if backend is None:
            raise ValueError("Backend name must be provided when using QI provider.")
        try:
            from quantuminspire import credentials as qicredentials
            from quantuminspire.qiskit import QI

            token = qicredentials.load_account()
            qi_authentication = qicredentials.get_token_authentication(token)
            qi_url = os.getenv("API_URL", "https://api.quantum-inspire.com/")
            project_name = f"Q-score {int(time.time())}"

            QI.set_authentication(qi_authentication, qi_url, project_name=project_name)
            return QI.get_backend(backend)
        except Exception as exc:
            raise AuthenticationError("Authentication with QI failed.") from exc

    raise ValueError(
        f"Provider {provider} and backend {backend} not among possible options."
    )


def _extract_ising_terms(
    operator: SparsePauliOp,
) -> tuple[list[tuple[int, float]], list[tuple[int, int, float]]]:
    """Split an Ising operator into single- and two-qubit Z terms."""
    single_z_terms: list[tuple[int, float]] = []
    zz_terms: list[tuple[int, int, float]] = []

    for coeff, pauli in zip(operator.coeffs, operator.paulis, strict=True):
        if abs(coeff.imag) > 1e-8:
            raise ValueError("Found complex coefficient in Ising operator.")

        label = pauli.to_label()
        qubits = [len(label) - 1 - idx for idx, char in enumerate(label) if char == "Z"]

        if not qubits:
            continue  # constant term

        value = float(coeff.real)
        if len(qubits) == 1:
            single_z_terms.append((qubits[0], value))
        elif len(qubits) == 2:
            zz_terms.append((qubits[0], qubits[1], value))
        else:
            raise ValueError(
                "Higher-order interactions are not supported by this QAOA ansatz."
            )

    return single_z_terms, zz_terms


def _build_qaoa_ansatz(
    num_qubits: int,
    single_z_terms: list[tuple[int, float]],
    zz_terms: list[tuple[int, int, float]],
    reps: int = 1,
) -> QuantumCircuit:
    """Construct a parameterized QAOA circuit without deprecated n-local helpers."""
    beta_params = ParameterVector("beta", reps)
    gamma_params = ParameterVector("gamma", reps)
    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))

    for layer in range(reps):
        gamma = gamma_params[layer]
        for qubit, coeff in single_z_terms:
            if coeff != 0:
                circuit.rz(2 * gamma * coeff, qubit)
        for qubit_i, qubit_j, coeff in zz_terms:
            if coeff != 0:
                circuit.rzz(2 * gamma * coeff, qubit_i, qubit_j)
        beta = beta_params[layer]
        for qubit in range(num_qubits):
            circuit.rx(2 * beta, qubit)

    return circuit


def run_QAOA(
    qp: QuadraticProgram,
    provider: str | None = None,
    backend: str | None = None,
    number_of_shots: int | None = None,
    maxiter: int = 100,
    reps: int = 1,
    max_attempts: int | None = 10,
    seed: int | None = None,
) -> list[int]:
    """Solve a Q-score instance with QAOA.

    Args:
        qp: quadratic program of the Q-score instance.
        provider: hardware provider, or ``None`` for local statevector sampling.
        backend: backend name for hardware providers.
        number_of_shots: shots per circuit evaluation.
        maxiter: maximum iterations for the COBYLA optimizer.
        reps: number of QAOA ansatz layers (p).
        max_attempts: maximum solve attempts before giving up.
        seed: seeds the local sampler and the variational initial point.

    Returns:
        Bitstring of the best assignment found.

    Raises:
        ValueError: if no feasible solution was found within ``max_attempts``.
    """
    if number_of_shots is None:
        number_of_shots = 1024

    if seed is not None:
        # SamplingVQE picks its initial point from this generator when none is
        # supplied, so fixing it fixes the optimizer's trajectory.
        from qiskit_algorithms.utils import algorithm_globals

        algorithm_globals.random_seed = int(seed)

    backend_instance = initialize_backend(provider, backend)
    if backend_instance is None:
        # Local: seeding the sampler makes the shot draws reproducible too.
        sampler = StatevectorSampler(
            default_shots=number_of_shots, seed=None if seed is None else int(seed)
        )
    else:
        sampler = BackendSamplerV2(
            backend=backend_instance,
            options={"default_shots": number_of_shots},
        )

    converter = QuadraticProgramToQubo()
    qubo_problem = converter.convert(qp)
    operator, _ = qubo_problem.to_ising()
    single_z_terms, zz_terms = _extract_ising_terms(operator)
    ansatz = _build_qaoa_ansatz(
        operator.num_qubits, single_z_terms, zz_terms, reps=reps
    )
    optimizer = COBYLA(maxiter=maxiter)
    qaoa_mes = SamplingVQE(ansatz=ansatz, sampler=sampler, optimizer=optimizer)

    qaoa = MinimumEigenOptimizer(qaoa_mes)
    for _ in range(max_attempts):
        qaoa_result = qaoa.solve(qubo_problem)
        if qaoa_result.status == OptimizationResultStatus.SUCCESS:
            return [round(value) for value in qaoa_result.x]
    raise ValueError("Could not find feasible solution")
